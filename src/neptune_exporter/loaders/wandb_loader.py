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
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional, Any
import pandas as pd
import pyarrow as pa
import wandb

from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
from neptune_exporter.loaders.loader import DataLoader


class WandBLoader(DataLoader):
    """Loads Neptune data from parquet files into Weights & Biases."""

    def __init__(
        self,
        entity: str,
        api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize W&B loader.

        Args:
            entity: W&B entity (organization/username)
            api_key: Optional W&B API key for authentication
            name_prefix: Optional prefix for project and run names
            verbose: Enable verbose logging
        """
        self.entity = entity
        self.name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.ERROR)
        self._verbose = verbose
        self._active_run: Optional[wandb.Run] = None

        # Authenticate with W&B
        if api_key:
            wandb.login(key=api_key)

        # Configure W&B logging
        if not verbose:
            os.environ["WANDB_SILENT"] = "true"

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path to W&B-compatible key.

        W&B key constraints:
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Pattern: /^[_a-zA-Z][_a-zA-Z0-9]*$/
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

    def _get_project_name(self, project_id: str) -> str:
        """Get W&B project name from Neptune project ID."""
        # W&B uses entity/project structure
        # Neptune project_id maps directly to W&B project
        name = project_id

        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize for W&B project name (alphanumeric, hyphens, underscores)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to W&B integer step."""
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """
        Neptune experiment_name maps to W&B group (set in create_run).
        We return the experiment name as the group name to use.
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find a run by name in a W&B project.

        Args:
            run_name: Name of the run to find
            experiment_id: W&B group name (experiment name from Neptune)
            project_id: Neptune project ID (used to construct W&B project name)

        Returns:
            W&B run ID if found, None otherwise
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Use wandb.Api() to search for runs
            api = wandb.Api()
            project_path = f"{self.entity}/{sanitized_project}"

            # Search for runs with matching name and group
            filters = {"display_name": run_name}
            if experiment_id:
                filters["group"] = experiment_id

            runs = api.runs(project_path, filters=filters, per_page=1)

            # Get the first matching run
            for run in runs:
                return TargetRunId(run.id)

            return None
        except Exception as e:
            self._logger.error(f"Error finding run '{run_name}': {e}")
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
        """Create W&B run, with support for forked runs.

        Args:
            fork_step: Fork step as float (decimal). Will be converted to int using step_multiplier.
            step_multiplier: Step multiplier for converting decimal steps to integers.
                If provided, will be used for fork_step conversion. If not provided,
                will calculate from fork_step alone as fallback.
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Prepare init arguments
            init_kwargs: dict[str, Any] = {
                "entity": self.entity,
                "project": sanitized_project,
                "group": experiment_id,
                "name": run_name,
            }

            # Handle forking if parent exists
            if parent_run_id:
                # Convert fork_step to int using provided step_multiplier
                # step_multiplier should always be provided when fork_step is set
                if fork_step is not None:
                    if step_multiplier is None:
                        raise ValueError(
                            "step_multiplier must be provided when fork_step is set"
                        )
                    step_int = self._convert_step_to_int(
                        Decimal(str(fork_step)), step_multiplier
                    )
                else:
                    step_int = 0

                # W&B fork format: entity/project/run_id?_step=step
                fork_from = f"{self.entity}/{sanitized_project}/{parent_run_id}?_step={step_int}"
                init_kwargs["fork_from"] = fork_from
                self._logger.info(
                    f"Creating forked run '{run_name}' from parent {parent_run_id} at step {step_int}"
                )

            # Initialize the run
            run = wandb.init(**init_kwargs)
            wandb_run_id = run.id

            self._active_run = run

            self._logger.info(f"Created run '{run_name}' with W&B ID {wandb_run_id}")
            return TargetRunId(wandb_run_id)

        except Exception as e:
            self._logger.error(f"Error creating run '{run_name}': {e}")
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """Upload all data for a single run to W&B.

        Args:
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        try:
            # Note: We assume the run is already active from create_run
            # If not, we would need to resume it
            if self._active_run is None or self._active_run.id != run_id:
                self._logger.error(
                    f"Run {run_id} is not active. Call create_run first."
                )
                raise RuntimeError(f"Run {run_id} is not active")

            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()

                self.upload_parameters(run_df, run_id)
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

            # Finish the run
            self._active_run.finish()
            self._active_run = None

            self._logger.info(f"Successfully uploaded run {run_id} to W&B")

        except Exception as e:
            self._logger.error(f"Error uploading run {run_id}: {e}")
            if self._active_run:
                self._active_run.finish(exit_code=1)
                self._active_run = None
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters (configs) to W&B run."""
        if self._active_run is None:
            raise RuntimeError("No active run")

        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        config = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Get the appropriate value based on attribute type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                config[attr_name] = row["float_value"]
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                config[attr_name] = int(row["int_value"])
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                config[attr_name] = row["string_value"]
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                config[attr_name] = bool(row["bool_value"])
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                config[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                config[attr_name] = list(row["string_set_value"])

        if config:
            self._active_run.config.update(config)
            self._logger.info(f"Uploaded {len(config)} parameters for run {run_id}")

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """Upload metrics (float series) to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Use global step multiplier (calculated from all series + fork_step)
        # Group by step to log all metrics at each step together
        for step_value, group in metrics_data.groupby("step"):
            if pd.notna(step_value):
                step = self._convert_step_to_int(step_value, step_multiplier)

                metrics = {}
                for _, row in group.iterrows():
                    if pd.notna(row["float_value"]):
                        attr_name = self._sanitize_attribute_name(row["attribute_path"])
                        metrics[attr_name] = row["float_value"]

                if metrics:
                    self._active_run.log(metrics, step=step)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as artifacts to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        # Handle regular files
        file_data = run_data[
            run_data["attribute_type"].isin(["file", "file_set", "artifact"])
        ]
        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    artifact = wandb.Artifact(
                        name=attr_name, type=row["attribute_type"]
                    )
                    if file_path.is_file():
                        artifact.add_file(str(file_path))
                    else:
                        artifact.add_dir(str(file_path))
                    self._active_run.log_artifact(artifact)
                else:
                    self._logger.warning(f"File not found: {file_path}")

        # Handle file series
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
                        artifact_name = f"{attr_name}_step_{step}"
                        artifact = wandb.Artifact(
                            name=artifact_name, type="file_series"
                        )
                        if file_path.is_file():
                            artifact.add_file(str(file_path))
                        else:
                            artifact.add_dir(str(file_path))
                        self._active_run.log_artifact(artifact)
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle string series as text artifacts
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create temporary file with text content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8"
            ) as tmp_file:
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

                # Create and log W&B artifact
                artifact = wandb.Artifact(name=attr_name, type="string_series")
                artifact.add_file(tmp_file_path, name="series.txt")
                self._active_run.log_artifact(artifact)

        # Handle histogram series as W&B Histograms
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)
            # Use global step multiplier

            for _, row in group.iterrows():
                if pd.notna(row["histogram_value"]) and isinstance(
                    row["histogram_value"], dict
                ):
                    step = (
                        self._convert_step_to_int(row["step"], step_multiplier)
                        if pd.notna(row["step"])
                        else 0
                    )
                    hist = row["histogram_value"]

                    # Convert Neptune histogram to W&B Histogram
                    # Neptune format: {"type": str, "edges": list, "values": list}
                    # W&B expects histogram data as np_histogram tuple or sequence
                    try:
                        wandb_hist = wandb.Histogram(
                            np_histogram=(hist.get("values", []), hist.get("edges", []))
                        )
                        self._active_run.log({attr_name: wandb_hist}, step=step)
                    except Exception as e:
                        self._logger.warning(
                            f"Failed to log histogram for {attr_path} at step {step}: {e}"
                        )

        self._logger.info(f"Uploaded artifacts for run {run_id}")
