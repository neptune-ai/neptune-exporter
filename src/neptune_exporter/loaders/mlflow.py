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

import re
import logging
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import pyarrow as pa
import mlflow
import mlflow.tracking
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


class MLflowLoader:
    """Loads Neptune data from parquet files into MLflow."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name_prefix: Optional[str] = None,
        step_multiplier: int = 1_000_000,
    ):
        """
        Initialize MLflow loader.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_name_prefix: Optional prefix for experiment names (to handle org/project structure)
            step_multiplier: Multiplier to convert Neptune decimal steps to MLflow integer steps
        """
        self.tracking_uri = tracking_uri
        self.experiment_name_prefix = experiment_name_prefix
        self.step_multiplier = step_multiplier
        self._logger = logging.getLogger(__name__)

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

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

    def _convert_step_to_int(self, step: Decimal) -> int:
        """Convert Neptune decimal step to MLflow integer step."""
        if step is None:
            return 0
        return int(float(step) * self.step_multiplier)

    def _get_experiment_name(self, project_id: str) -> str:
        """Get MLflow experiment name from Neptune project ID."""
        if self.experiment_name_prefix:
            return f"{self.experiment_name_prefix}/{project_id}"
        return project_id

    def create_experiment(self, project_id: str) -> str:
        """Create or get MLflow experiment for a Neptune project."""
        experiment_name = self._get_experiment_name(project_id)

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self._logger.info(
                    f"Created experiment '{experiment_name}' with ID {experiment_id}"
                )
            else:
                experiment_id = experiment.experiment_id
                self._logger.info(
                    f"Using existing experiment '{experiment_name}' with ID {experiment_id}"
                )

            return experiment_id
        except Exception as e:
            self._logger.error(
                f"Error creating/getting experiment '{experiment_name}': {e}"
            )
            raise

    def create_run(
        self, experiment_id: str, run_id: str, parent_run_id: Optional[str] = None
    ) -> str:
        """Create MLflow run."""
        tags = {}
        if parent_run_id:
            tags[MLFLOW_PARENT_RUN_ID] = parent_run_id

        try:
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=run_id, tags=tags
            ):
                mlflow_run_id = mlflow.active_run().info.run_id
                self._logger.info(
                    f"Created run '{run_id}' with MLflow ID {mlflow_run_id}"
                )
                return mlflow_run_id
        except Exception as e:
            self._logger.error(f"Error creating run '{run_id}': {e}")
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: str) -> None:
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
            elif row["attribute_type"] == "string_set" and pd.notna(
                row["string_set_value"]
            ):
                # Convert list to comma-separated string
                string_set = row["string_set_value"]
                if isinstance(string_set, list):
                    params[attr_name] = ",".join(str(x) for x in string_set)
                else:
                    params[attr_name] = str(string_set)

        if params:
            mlflow.log_params(params)
            self._logger.info(f"Uploaded {len(params)} parameters for run {run_id}")

    def upload_metrics(self, run_data: pd.DataFrame, run_id: str) -> None:
        """Upload metrics (float series) to MLflow run."""
        # Filter for float_series type
        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Group by attribute path and log metrics
        for attr_path, group in metrics_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Sort by step
            group = group.sort_values("step")

            for _, row in group.iterrows():
                if pd.notna(row["float_value"]) and pd.notna(row["step"]):
                    step = self._convert_step_to_int(row["step"])
                    mlflow.log_metric(attr_name, row["float_value"], step=step)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self, run_data: pd.DataFrame, run_id: str, files_base_path: Path
    ) -> None:
        """Upload files and series as artifacts to MLflow run."""
        # Handle regular files
        file_data = run_data[run_data["attribute_type"] == "file"]
        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    mlflow.log_artifact(str(file_path), artifact_path=attr_name)
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
                        # Include step in artifact name for file series
                        step = (
                            self._convert_step_to_int(row["step"])
                            if pd.notna(row["step"])
                            else 0
                        )
                        artifact_path = f"{attr_name}/step_{step}"
                        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle string series as text artifacts
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create a table-like structure
            series_text = []
            for _, row in group.iterrows():
                if pd.notna(row["string_value"]) and pd.notna(row["step"]):
                    step = self._convert_step_to_int(row["step"])
                    series_text.append(f"Step {step}: {row['string_value']}")

            if series_text:
                # Log as text artifact
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    f.write("\n".join(series_text))
                    mlflow.log_artifact(f.name, artifact_path=f"{attr_name}/series.txt")
                    Path(f.name).unlink()  # Clean up temp file

        # Handle histogram series as table artifacts
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create CSV table
            import tempfile
            import csv

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["step", "type", "edges", "values"])

                for _, row in group.iterrows():
                    if pd.notna(row["histogram_value"]) and isinstance(
                        row["histogram_value"], dict
                    ):
                        step = (
                            self._convert_step_to_int(row["step"])
                            if pd.notna(row["step"])
                            else 0
                        )
                        hist = row["histogram_value"]
                        edges_str = ",".join(str(x) for x in hist.get("edges", []))
                        values_str = ",".join(str(x) for x in hist.get("values", []))
                        writer.writerow(
                            [step, hist.get("type", ""), edges_str, values_str]
                        )

                mlflow.log_artifact(f.name, artifact_path=f"{attr_name}/histograms.csv")
                Path(f.name).unlink()  # Clean up temp file

        self._logger.info(f"Uploaded artifacts for run {run_id}")

    def upload_run_data(
        self,
        run_data: Union[pd.DataFrame, pa.Table],
        run_id: str,
        experiment_id: str,
        files_base_path: Path,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """Upload all data for a single run to MLflow."""
        # Convert PyArrow Table to pandas DataFrame if needed
        if isinstance(run_data, pa.Table):
            if len(run_data) == 0:
                # Handle empty table case
                run_data = pd.DataFrame()
            else:
                run_data = run_data.to_pandas()

        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_id):
                # Set parent run if specified
                if parent_run_id:
                    mlflow.set_tag(MLFLOW_PARENT_RUN_ID, parent_run_id)

                # Upload different data types
                self.upload_parameters(run_data, run_id)
                self.upload_metrics(run_data, run_id)
                self.upload_artifacts(run_data, run_id, files_base_path)

                mlflow_run_id = mlflow.active_run().info.run_id
                self._logger.info(f"Successfully uploaded run {run_id} to MLflow")
                return mlflow_run_id

        except Exception as e:
            self._logger.error(f"Error uploading run {run_id}: {e}")
            raise
