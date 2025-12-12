#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional, Any, Union, Dict
from datetime import datetime

import pandas as pd
import pyarrow as pa

import comet_ml
from comet_ml.messages import (
    MetricMessage,
    SystemDetailsMessage,
)

from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
from neptune_exporter.loaders.loader import DataLoader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def is_image(filename: Union[str, Path]) -> bool:
    """Check if a file is an image based on its extension.

    Args:
        filename: File path or name to check

    Returns:
        True if the file extension is a recognized image format, False otherwise
    """
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


class CometLoader(DataLoader):
    """Loads Neptune data from parquet files into a Comet installation."""

    def __init__(
        self,
        workspace: str,
        api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ):
        """
        Initialize Comet loader.

        Args:
            workspace: Comet workspace name
            api_key: Optional Comet API key for authentication
            name_prefix: Optional prefix for project and run names
            show_client_logs: Enable verbose logging from Comet client library
        """
        self.workspace = workspace
        self.name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)
        self._comet_experiment: Optional[comet_ml.Experiment] = None
        self._comet_api_key = api_key
        self._comet_data: Dict[str, Any] = {}
        self._comet_system_info: Dict[str, Any] = {}

        # Configure Comet logging - suppress INFO and WARNING messages
        # The logger name is "comet_ml" (not "COMET" - that's just the formatter prefix)
        if show_client_logs:
            logging.getLogger("comet_ml").setLevel(logging.INFO)
        else:
            logging.getLogger("comet_ml").setLevel(logging.ERROR)

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path to Comet-compatible format.

        Key constraints:
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Pattern: /^[_a-zA-Z][_a-zA-Z0-9]*$/

        Args:
            attribute_path: Original Neptune attribute path

        Returns:
            Sanitized attribute name compatible with Comet requirements
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", attribute_path)

        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Handle empty result
        if not sanitized:
            sanitized = "_attribute"

        return sanitized

    def _get_project_name(self, project_id: ProjectId) -> str:
        """Get Comet project name from Neptune project ID.

        Args:
            project_id: Neptune project ID

        Returns:
            Sanitized project name for Comet, optionally prefixed
        """
        name: str = str(project_id)

        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize project name (alphanumeric, hyphens, underscores)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to integer step.

        Args:
            step: Neptune decimal step value
            step_multiplier: Multiplier to convert decimal to integer

        Returns:
            Integer step value, or 0 if step is None
        """
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def _log_system_info(self, **kwargs: Any) -> None:
        """
        Log the system info.

        Args:
            **kwargs: System info keys can be:
                user, command, env, hostname, ip, machine, os_release,
                os_type, os, pid, processor, python_exe,
                python_version_verbose, python_version,
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        settings: Dict[str, Any] = {
            key: None
            for key in [
                "user",
                "hostname",
                "command",
                "env",
                "ip",
                "machine",
                "os_release",
                "os_type",
                "os",
                "pid",
                "processor",
                "python_exe",
                "python_version_verbose",
                "python_version",
            ]
        }
        settings.update(kwargs)
        message = SystemDetailsMessage(**settings)
        self._comet_experiment._enqueue_message(message)

    def _log_metric(self, name: str, value: float, step: int, timestamp: float) -> None:
        """
        Add a metric with timestamp to a Comet Experiment.

        Args:
            name: Metric name
            value: Metric value
            step: Step number (integer)
            timestamp: Timestamp as Unix timestamp (float)
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        message = MetricMessage(
            context=None,
            timestamp=timestamp,
        )
        message.set_metric(name, value, step=step)
        self._comet_experiment._enqueue_message(message)

    def _upload_histograms(self, run_data: pd.DataFrame) -> None:
        """
        Upload Neptune distribution data as Comet histogram_3d.

        Filters for float_series attributes that start with 'distributions/'
        and uploads them as 3D histograms to Comet.

        Args:
            run_data: DataFrame containing run data with attribute_type and attribute_path columns
        """
        # Filter for distribution data:
        filtered = run_data[
            (run_data["attribute_type"] == "float_series")
            & (run_data["attribute_path"].str.startswith("distributions/"))
        ]

        result_df = (
            filtered.groupby("attribute_path")["float_value"].apply(list).reset_index()
        )
        result_df.columns = ["attribute_path", "float_values"]

        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        # Or iterate more efficiently
        for attribute_path, float_values in zip(
            result_df["attribute_path"], result_df["float_values"]
        ):
            _, attribute_name = attribute_path.split("/", 1)
            try:
                self._comet_experiment.log_histogram_3d(
                    values=float_values,
                    name=attribute_name,
                    # step=step
                )
            except Exception:
                self._logger.error(
                    f"Failed to log histogram for {attribute_path}",
                    exc_info=True,
                )

    def _upload_string_series(
        self, step_multiplier: int, run_data: pd.DataFrame
    ) -> None:
        """
        Upload string series data to Comet as text assets.

        Creates a temporary text file containing all string series values
        with their steps and timestamps, then uploads it as an asset.

        Args:
            step_multiplier: Multiplier for converting decimal steps to integers
            run_data: DataFrame containing run data with string_series attributes
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create temporary file with text content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8"
            ) as tmp_file:
                series_step: Optional[int] = None
                for _, row in group.iterrows():
                    if pd.notna(row["string_value"]):
                        series_step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else None
                        )
                        timestamp = (
                            row["timestamp"].isoformat()
                            if pd.notna(row["timestamp"])
                            else None
                        )
                        text_line = (
                            f"{series_step}; {timestamp}; {row['string_value']}\n"
                        )
                        tmp_file.write(text_line)
                tmp_file_path = tmp_file.name
                self._comet_experiment.log_asset(
                    file_data=tmp_file_path,
                    file_name=attr_name,
                    step=series_step,
                )

    def _upload_file_series(
        self, step_multiplier: int, files_base_path: Path, run_data: pd.DataFrame
    ) -> None:
        """
        Upload file series data to Comet as assets or images.

        Processes file_series attributes, uploading each file at its respective step.
        Image files are logged as images, other files as assets, and directories
        as asset folders.

        Args:
            step_multiplier: Multiplier for converting decimal steps to integers
            files_base_path: Base directory path for file artifacts
            run_data: DataFrame containing run data with file_series attributes
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        file_series_data = run_data[run_data["attribute_type"] == "file_series"]
        for attr_path, group in file_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            for _, row in group.iterrows():
                if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                    file_path = files_base_path / row["file_value"]["path"]
                    if file_path.exists():
                        step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else 0
                        )
                        if file_path.is_file():
                            if is_image(file_path):
                                self._comet_experiment.log_image(
                                    image_data=file_path,
                                    name=attr_name,
                                    step=step,
                                )
                            else:
                                self._comet_experiment.log_asset(
                                    file_data=file_path,
                                    file_name=attr_name,
                                    step=step,
                                )
                        else:
                            self._comet_experiment.log_asset_folder(
                                folder=file_path,
                                step=step,
                            )
                    else:
                        self._logger.warning(f"File not found: {file_path}")

    def _upload_files(self, files_base_path: Path, file_data: pd.DataFrame) -> None:
        """
        Upload file data to Comet as assets or folders.

        Processes file, file_set, and artifact attributes. Files are uploaded
        as assets, directories as asset folders. Source code directories
        are handled specially.

        Args:
            files_base_path: Base directory path for file artifacts
            file_data: DataFrame containing file, file_set, or artifact attributes
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    if file_path.is_file():
                        self._comet_experiment.log_asset(
                            file_data=file_path,
                            file_name=attr_name,
                        )
                    else:
                        if "source_code" in str(file_path):
                            self._upload_source_code(attr_name, file_path)
                        else:
                            self._comet_experiment.log_asset_folder(
                                folder=file_path,
                            )
                else:
                    self._logger.warning(f"File not found: {file_path}")

    def _upload_source_code(self, attr_name: str, file_path: Path) -> None:
        """
        Upload source code files to Comet.

        Iterates through files in the given directory and uploads them as code.
        ZIP files are handled separately by extracting and uploading contents.

        Args:
            attr_name: Attribute name to use as code_name in Comet
            file_path: Path to directory containing source code files
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        # Go through the folder
        for filename in file_path.iterdir():
            if filename.is_file():
                if filename.suffix == ".zip":
                    self._upload_source_code_zip(filename)
                else:
                    # Log the file with code_name from attr_name
                    self._comet_experiment.log_code(
                        file_name=str(filename), code_name=attr_name
                    )

    def _upload_source_code_zip(self, filename: Path) -> None:
        """
        Upload source code from a zip file to Comet.

        Extracts the ZIP file to a temporary directory and uploads each
        file within it, using the file's path within the ZIP as the code_name.

        Args:
            filename: Path to the ZIP file containing source code
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")
        # Unzip the file and log each file with code_name from zipfile
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                # Get all file paths in the zip
                zip_file_paths = zip_ref.namelist()
                zip_ref.extractall(temp_dir)
                # Log each file with its path from the zip as code_name
                for zip_file_path in zip_file_paths:
                    # Skip directories
                    if not zip_file_path.endswith("/"):
                        extracted_file = Path(temp_dir) / zip_file_path
                        if extracted_file.is_file():
                            self._comet_experiment.log_code(
                                file_name=str(extracted_file),
                                code_name=zip_file_path,
                            )
                        # else?
                    # else?

    # Overloaded methods:

    def create_experiment(
        self, project_id: ProjectId, experiment_name: str
    ) -> TargetExperimentId:
        """
        Create or get an experiment in Comet.

        In Comet, the experiment_name maps directly to the experiment.
        This method returns the experiment name to use.

        Args:
            project_id: Neptune project ID
            experiment_name: Name of the experiment

        Returns:
            Target experiment ID (which is the experiment name in Comet)
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """
        Find a run by name in a Comet project.

        Args:
            project_id: Neptune project ID
            run_name: Name of the run to find
            experiment_id: Optional Comet experiment ID

        Returns:
            None, as Comet doesn't support resuming or forking runs
        """
        return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """
        Create a Comet experiment (run).

        Creates a new Comet experiment with the specified name in the project.
        Configures the experiment to disable automatic logging of code, graphs,
        parameters, and environment details.

        Args:
            project_id: Neptune project ID
            run_name: Name of the run/experiment to create
            experiment_id: Optional experiment ID (not used in Comet)
            parent_run_id: Optional parent run ID (not used in Comet)
            fork_step: Optional fork step as float (decimal). Will be converted to int using step_multiplier
            step_multiplier: Optional step multiplier for converting decimal steps to integers.
                If provided, will be used for fork_step conversion. If not provided,
                will calculate from fork_step alone as fallback.

        Returns:
            Target run ID (Comet experiment ID)

        Raises:
            Exception: If experiment creation fails
        """
        project_name = self._get_project_name(project_id)

        try:
            self._comet_experiment = comet_ml.Experiment(
                api_key=self._comet_api_key,
                workspace=self.workspace,
                project_name=project_name,
                experiment_name=run_name,
                log_code=False,
                log_graph=False,
                auto_param_logging=False,
                parse_args=False,
                auto_output_logging=None,
                log_env_details=False,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                display_summary_level=0,
            )
            self._comet_experiment.set_name(run_name)

            self._logger.info(
                f"Created Comet experiment '{run_name}' with ID {self._comet_experiment.id}"
            )
            return TargetRunId(self._comet_experiment.id)

        except Exception:
            self._logger.error(
                f"Error creating project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """Upload all data for a single run to Comet.

        Processes the run data in chunks (PyArrow tables), converting each to pandas
        and uploading parameters, metrics, and artifacts. Ends the experiment
        after all data is uploaded.

        Args:
            run_data: Generator of PyArrow tables containing run data
            run_id: Target run ID in Comet
            files_directory: Base directory path for file artifacts
            step_multiplier: Step multiplier for converting decimal steps to integers

        Raises:
            Exception: If upload fails
        """
        try:
            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()
                # Initialize run data:
                self._comet_data = {}
                self._comet_system_info = {}
                # Upload and collect data:
                self.upload_parameters(run_df, run_id)
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

                if self._comet_system_info:
                    self._log_system_info(**self._comet_system_info)

            if self._comet_experiment is not None:
                self._comet_experiment.end()
                if self._comet_data:
                    # Update items after initial experiment:
                    api_experiment = comet_ml.api.APIExperiment(
                        api_key=self._comet_api_key,
                        previous_experiment=self._comet_experiment.id,
                    )
                    for method in self._comet_data:
                        value = self._comet_data[method]
                        getattr(api_experiment, method)(value)

            self._logger.info(f"Successfully uploaded run {run_id} to Comet")

        except Exception:
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters to Comet experiment.

        Extracts parameters (float, int, string, bool, datetime, string_set) from
        the run data and uploads them to Comet. Special handling for model_summary
        which is set as model graph instead of a parameter.

        Args:
            run_data: DataFrame containing run data with parameter attributes
            run_id: Target run ID in Comet

        Raises:
            RuntimeError: If no active experiment exists
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")

        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        parameters = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Get the appropriate value based on attribute type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                parameters[attr_name] = row["float_value"]
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                parameters[attr_name] = int(row["int_value"])
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                if "model_summary" in attr_name:
                    self._comet_experiment.set_model_graph(row["string_value"])
                else:
                    parameters[attr_name] = row["string_value"]
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                parameters[attr_name] = bool(row["bool_value"])
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                parameters[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                parameters[attr_name] = list(row["string_set_value"])

        if parameters:
            # Save data for later setting:
            if "sys_owner" in parameters:
                self._comet_system_info["user"] = parameters["sys_owner"]
            if "sys_hostname" in parameters:
                self._comet_system_info["hostname"] = parameters["sys_hostname"]
            if "sys_tags" in parameters:
                try:
                    tags = parameters["sys_tags"]
                    if tags:
                        self._comet_data["add_tags"] = tags
                except Exception:
                    self._logger.error(
                        f"Unable to convert sys_tags: {parameters['sys_tags']}",
                        exc_info=True,
                    )
            if "sys_creation_time" in parameters:
                try:
                    datetime_string = parameters["sys_creation_time"]
                    dt = datetime.fromisoformat(datetime_string)
                    ms = int(dt.timestamp() * 1000)
                    self._comet_data["set_start_time"] = ms
                except Exception:
                    self._logger.error(
                        f"Unable to convert sys_creation_time: {parameters['sys_creation_time']}",
                        exc_info=True,
                    )
            if "sys_modification_time" in parameters:
                try:
                    datetime_string = parameters["sys_modification_time"]
                    dt = datetime.fromisoformat(datetime_string)
                    ms = int(dt.timestamp() * 1000)
                    self._comet_data["set_end_time"] = ms
                except Exception:
                    self._logger.error(
                        f"Unable to convert sys_modification_time: {parameters['sys_modification_time']}",
                        exc_info=True,
                    )

            self._comet_experiment.log_parameters(parameters)
            self._logger.info(f"Uploaded {len(parameters)} parameters for run {run_id}")

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """Upload metrics (float series) to Comet experiment.

        Processes float_series attributes and uploads them as metrics with
        their corresponding steps and timestamps.

        Args:
            run_data: DataFrame containing run data with float_series attributes
            run_id: Target run ID in Comet
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)

        Raises:
            RuntimeError: If no active experiment exists
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")

        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Use global step multiplier (calculated from all series + fork_step)
        # Group by step to log all metrics at each step together
        for row in metrics_data.itertuples():
            step_value = row.step
            timestamp = row.timestamp.timestamp()
            if pd.notna(step_value):
                step = self._convert_step_to_int(step_value, step_multiplier)
                if pd.notna(row.float_value):
                    metric_name = self._sanitize_attribute_name(row.attribute_path)
                    value = row.float_value
                    self._log_metric(metric_name, value, step=step, timestamp=timestamp)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as assets to Comet experiment.

        Uploads files, file series, string series, and histograms to Comet.
        Handles different asset types including images, files, folders, and code.

        Args:
            run_data: DataFrame containing run data with file and series attributes
            run_id: Target run ID in Comet
            files_base_path: Base directory path for file artifacts
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)

        Raises:
            RuntimeError: If no active run exists
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active run")

        file_data = run_data[
            run_data["attribute_type"].isin(["file", "file_set", "artifact"])
        ]
        self._upload_files(files_base_path, file_data)
        self._upload_file_series(step_multiplier, files_base_path, run_data)
        self._upload_string_series(step_multiplier, run_data)
        self._upload_histograms(run_data)

        self._logger.info(f"Uploaded assets for run {run_id}")
