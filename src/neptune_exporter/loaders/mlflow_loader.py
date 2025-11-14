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
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional
from mlflow.entities.run import Run
import pandas as pd
import pyarrow as pa
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from neptune_exporter.types import TargetExperimentId, TargetRunId
from neptune_exporter.loaders.loader import DataLoader


class MLflowLoader(DataLoader):
    """Loads Neptune data from parquet files into MLflow."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        name_prefix: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize MLflow loader.

        Args:
            tracking_uri: MLflow tracking URI
            name_prefix: Optional prefix for experiment and run names (to handle org/project structure)
            verbose: Enable verbose logging
        """
        self.tracking_uri = tracking_uri
        self.name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO if verbose else logging.ERROR)
        self._verbose = verbose

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Configure MLflow logging
        if verbose:
            logging.getLogger("mlflow").setLevel(logging.INFO)
        else:
            logging.getLogger("mlflow").setLevel(logging.ERROR)
            os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "1"

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path to MLflow-compatible key.

        MLflow key constraints:
        - Only alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/)
        - Max length 250 characters
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s/]", "_", attribute_path)

        # Truncate if too long
        if len(sanitized) > 250:
            sanitized = sanitized[:250]
            self._logger.warning(
                f"Truncated attribute path '{attribute_path}' to '{sanitized}'"
            )

        return sanitized

    def _get_experiment_name(self, project_id: str, experiment_name: str) -> str:
        """Get MLflow experiment name from Neptune project ID."""
        name = f"{project_id}/{experiment_name}"

        if self.name_prefix:
            name = f"{self.name_prefix}/{name}"

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to MLflow integer step."""
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """Create or get MLflow experiment for a Neptune project."""
        target_experiment_name = self._get_experiment_name(project_id, experiment_name)

        try:
            experiment = mlflow.get_experiment_by_name(target_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(target_experiment_name)
                self._logger.info(
                    f"Created experiment '{target_experiment_name}' with ID {experiment_id}"
                )
            else:
                experiment_id = experiment.experiment_id
                self._logger.info(
                    f"Using existing experiment '{target_experiment_name}' with ID {experiment_id}"
                )

            return TargetExperimentId(experiment_id)
        except Exception as e:
            self._logger.error(
                f"Error creating/getting experiment '{target_experiment_name}': {e}"
            )
            raise

    def find_run(
        self,
        project_id: str,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find a run by name in an experiment."""
        try:
            existing_runs: list[Run] = mlflow.search_runs(
                experiment_ids=[experiment_id] if experiment_id else None,
                filter_string=f"attributes.run_name = '{run_name}'",
                output_format="list",
                max_results=1,
            )
            return TargetRunId(existing_runs[0].info.run_id) if existing_runs else None
        except Exception as e:
            self._logger.error(f"Error finding run '{run_name}': {e}")
            return None

    def create_run(
        self,
        project_id: str,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """Create MLflow run.

        Args:
            fork_step: Ignored for MLflow (parent relationships don't use step information)
            step_multiplier: Ignored for MLflow (not needed for parent relationships)
        """
        tags = {}
        if parent_run_id:
            tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

        try:
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=run_name, tags=tags
            ) as active_run:
                mlflow_run_id = active_run.info.run_id
                self._logger.info(
                    f"Created run '{run_name}' with MLflow ID {mlflow_run_id}"
                )
                return TargetRunId(mlflow_run_id)
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
        """Upload all data for a single run to MLflow.

        Args:
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        try:
            with mlflow.start_run(run_id=run_id):
                for run_data_part in run_data:
                    run_df = run_data_part.to_pandas()
                    self.upload_parameters(run_df, run_id)
                    self.upload_metrics(run_df, run_id, step_multiplier)
                    self.upload_artifacts(
                        run_df, run_id, files_directory, step_multiplier
                    )

                    self._logger.info(f"Successfully uploaded run {run_id} to MLflow")

        except Exception as e:
            self._logger.error(f"Error uploading run {run_id}: {e}")
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters (configs) to MLflow run."""
        # Filter for parameter types
        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        params = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Get the appropriate value based on attribute type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                params[attr_name] = str(row["float_value"])
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                params[attr_name] = str(row["int_value"])
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                params[attr_name] = str(row["string_value"])
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                params[attr_name] = str(row["bool_value"])
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                params[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                params[attr_name] = ",".join(row["string_set_value"])

        if params:
            mlflow.log_params(params)
            self._logger.info(f"Uploaded {len(params)} parameters for run {run_id}")

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """Upload metrics (float series) to MLflow run."""
        # Filter for float_series type
        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        mlflow_client = MlflowClient()

        # Group by attribute path and log metrics
        for attr_path, group in metrics_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Sort by step
            group = group.sort_values("step")

            metrics = []
            for _, row in group.iterrows():
                if pd.notna(row["float_value"]) and pd.notna(row["step"]):
                    step = self._convert_step_to_int(
                        row["step"], step_multiplier=step_multiplier
                    )
                    # Convert timestamp to milliseconds if available
                    timestamp = None
                    if pd.notna(row["timestamp"]):
                        timestamp = int(row["timestamp"].timestamp() * 1000)
                    metrics.append(
                        Metric(
                            key=attr_name,
                            value=row["float_value"],
                            step=step,
                            timestamp=timestamp,
                        )
                    )

            mlflow_client.log_batch(
                run_id=run_id,
                metrics=metrics,
            )

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as artifacts to MLflow run."""
        # Handle regular files
        file_data = run_data[run_data["attribute_type"].isin(["file", "artifact"])]
        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    mlflow.log_artifact(
                        local_path=str(file_path), artifact_path=attr_name
                    )
                else:
                    self._logger.warning(f"File not found: {file_path}")

        # # Handle file series
        file_series_data = run_data[run_data["attribute_type"] == "file_series"]
        for attr_path, group in file_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            for _, row in group.iterrows():
                if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                    file_path = files_base_path / row["file_value"]["path"]
                    if file_path.exists():
                        # Include step in artifact name for file series
                        step = (
                            self._convert_step_to_int(
                                row["step"], step_multiplier=step_multiplier
                            )
                            if pd.notna(row["step"])
                            else None
                        )
                        artifact_path = f"{attr_name}/step_{step}"
                        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle file sets
        file_sets_data = run_data[run_data["attribute_type"] == "file_set"]
        for _, row in file_sets_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_set_path = files_base_path / row["file_value"]["path"]
                if file_set_path.is_dir():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    mlflow.log_artifact(str(file_set_path), artifact_path=attr_name)
                else:
                    self._logger.warning(f"File set not found: {file_set_path}")

        # Handle string series as text artifacts
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create a table-like structure
            series_text = []
            for _, row in group.iterrows():
                if pd.notna(row["string_value"]) and pd.notna(row["step"]):
                    step = self._convert_step_to_int(
                        row["step"], step_multiplier=step_multiplier
                    )
                    series_text.append(f"[{step}] {row['string_value']}")

            if series_text:
                # Log as text artifact
                mlflow.log_text(
                    text="\n".join(series_text),
                    artifact_file=f"{attr_name}/series.txt",
                    run_id=run_id,
                )

        # Handle histogram series as structured data
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Prepare histogram data as JSON for better structure preservation
            histogram_data = []
            for _, row in group.iterrows():
                if pd.notna(row["histogram_value"]) and isinstance(
                    row["histogram_value"], dict
                ):
                    step = (
                        self._convert_step_to_int(
                            row["step"], step_multiplier=step_multiplier
                        )
                        if pd.notna(row["step"])
                        else None
                    )
                    hist = row["histogram_value"]
                    histogram_data.append(
                        {
                            "step": step,
                            "type": hist.get("type", ""),
                            "edges": hist.get("edges", []),
                            "values": hist.get("values", []),
                            "timestamp": row["timestamp"].isoformat()
                            if pd.notna(row["timestamp"])
                            else None,
                        }
                    )

            if histogram_data:
                # Use MLflow's log_dict for structured data
                mlflow.log_dict(
                    dictionary={"histograms": histogram_data},
                    artifact_file=f"{attr_name}/histograms.json",
                    run_id=run_id,
                )

        self._logger.info(f"Uploaded artifacts for run {run_id}")
