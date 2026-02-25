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

import logging
import os
import re
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd
import pyarrow as pa

from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.storage.types import AnyPath
from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId

try:
    import litlogger
    from lightning_sdk import User, Organization, Teamspace

    LITLOGGER_AVAILABLE = True
except ImportError:
    LITLOGGER_AVAILABLE = False
    litlogger = None  # type: ignore
    User = None  # type: ignore
    Organization = None  # type: ignore
    Teamspace = None  # type: ignore


class LitLoggerLoader(DataLoader):
    """
    Loads Neptune data from parquet files into LitLogger (Lightning.ai).

    This loader migrates experiment data from Neptune to Lightning.ai's LitLogger.
    It handles the mapping between Neptune's data model and LitLogger's concepts:

    Neptune Concept -> LitLogger Concept
    -----------------------------------
    - Project       -> Teamspace (each Neptune project becomes a separate teamspace)
    - Run           -> Experiment (with just the run name, unique within teamspace)
    - Parameters    -> Metadata (stored as string key-value pairs)
    - Float Series  -> Metrics (logged with step information)
    - Files         -> Artifacts (uploaded with preserved directory structure)
    - String Series -> Text files (combined into .txt artifacts)
    - Histograms    -> PNG images (rendered as bar charts)

    Usage:
        loader = LitLoggerLoader()
        loader.create_run(project_id, run_name)
        loader.upload_run_data(data_generator, run_id, files_dir, step_multiplier)

    Note:
        LitLogger doesn't support run lookup, so runs will be re-created if the
        loader is executed multiple times with the same data.
    """

    def __init__(
        self,
        owner: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ):
        """
        Initialize LitLogger loader.

        Args:
            owner: Lightning.ai user or organization name where teamspaces and experiments will be created.
                If not provided, uses the authenticated user.
            api_key: Optional API key for authentication to lightning.ai.
                Can also be set via LIGHTNING_API_KEY environment variable.
            user_id: Optional user ID for authentication to lightning.ai.
                Can also be set via LIGHTNING_USER_ID environment variable.
            name_prefix: Optional prefix for experiment names. Useful for organizing
                multiple migration batches (e.g., "migration-2024-01").
            show_client_logs: Enable verbose logging from litlogger client.
                Useful for debugging connection issues.
        """

        if not LITLOGGER_AVAILABLE:
            raise RuntimeError(
                "LitLogger is not installed. Install with "
                "`pip install 'neptune-exporter[litlogger]'` to use the LitLogger loader."
            )

        # Configuration
        self.name_prefix = name_prefix
        self.show_client_logs = show_client_logs

        # Internal state
        self._logger = logging.getLogger(__name__)
        self._active_experiment: Optional[Any] = (
            None  # Current litlogger experiment instance
        )
        self._current_run_id: Optional[TargetRunId] = None  # ID of run being processed

        # Pending experiment info for deferred creation
        # We defer experiment creation until upload_run_data so we can extract
        # parameters from the first data chunk and include them as metadata
        self._pending_experiment: Optional[Dict[str, Any]] = None

        # Set authentication environment variables if provided
        # Lightning.ai SDK reads these automatically during litlogger.init()
        if api_key:
            os.environ["LIGHTNING_API_KEY"] = api_key
        if user_id:
            os.environ["LIGHTNING_USER_ID"] = user_id

        self.owner = self._validate_owner(owner_name=owner)

        # Cache for created experiment IDs (currently unused but reserved for future)
        self._experiments: Dict[str, TargetExperimentId] = {}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path to LitLogger-compatible key.

        Neptune uses hierarchical paths like "metrics/train/loss" but LitLogger
        requires valid Python identifiers for metric names.

        Transformations applied:
        - Replace slashes, dots, and special characters with underscores
        - Prepend underscore if name starts with a digit
        - Default to "_attribute" if result is empty

        Args:
            attribute_path: Neptune attribute path (e.g., "metrics/train/loss")

        Returns:
            Sanitized name (e.g., "metrics_train_loss")

        Examples:
            "metrics/loss" -> "metrics_loss"
            "1st_metric"   -> "_1st_metric"
            "train.acc"    -> "train_acc"
        """
        # Replace any non-alphanumeric character (except underscore) with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", attribute_path)

        # Python identifiers can't start with a digit - prepend underscore if needed
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Fallback for edge case of empty input
        if not sanitized:
            sanitized = "_attribute"

        return sanitized

    def _get_experiment_name(self, run_name: str, max_length: int = 64) -> str:
        """
        Generate a LitLogger experiment name from Neptune run name.

        Since each Neptune project maps to a separate teamspace, the experiment
        name only needs to contain the run name. Optionally prepends a user-defined prefix.

        Args:
            run_name: Neptune run name/ID (e.g., "RUN-123")
            max_length: Maximum length for the experiment name (default 64)

        Returns:
            Sanitized experiment name (e.g., "RUN_123"), truncated to max_length
        """
        name = run_name

        # Prepend optional prefix (useful for organizing migration batches)
        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize: keep only alphanumeric, hyphens, and underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # Truncate to max length
        if len(name) > max_length:
            name = name[:max_length]

        return name

    def _get_teamspace_name(self, project_id: str, max_length: int = 64) -> str:
        """
        Generate a LitLogger teamspace name from Neptune project ID.

        Each Neptune project maps to a separate Lightning.ai teamspace.

        Args:
            project_id: Neptune project ID (e.g., "workspace/project-name")
            max_length: Maximum length for the teamspace name (default 64)

        Returns:
            Sanitized teamspace name (e.g., "workspace_project-name"), truncated to max_length
        """
        # Sanitize: keep only alphanumeric, hyphens, and underscores
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id)

        # Truncate to max length
        if len(name) > max_length:
            name = name[:max_length]

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """
        Convert Neptune's decimal step to an integer for LitLogger.

        Neptune allows fractional steps (e.g., 0.5, 1.5) but LitLogger requires
        integers. The step_multiplier scales steps to preserve precision.

        Args:
            step: Neptune step value (Decimal, can be fractional)
            step_multiplier: Scaling factor (e.g., 100 to preserve 2 decimal places)

        Returns:
            Integer step value

        Example:
            step=1.5, multiplier=100 -> 150
        """
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def _strip_neptune_path_prefix(self, file_path: str) -> str:
        """
        Strip Neptune project/run prefix from file path.

        Neptune exports files with paths like:
        {project_path}/{run_id}/{attribute_path}

        This method strips the project_path and run_id prefix to get just
        the attribute path for cleaner organization in LitLogger.

        Args:
            file_path: Full path from Neptune file_value["path"]

        Returns:
            Path with project/run prefix stripped
        """
        if self._pending_experiment is None:
            return file_path

        # Get project_id and run_name from pending experiment
        # project_id might be like "showcase/onboarding-project"
        # run_name is the Neptune run ID like "IMG-177"
        project_id = self._pending_experiment.get("project_id", "")
        run_name = self._pending_experiment.get("run_name", "")

        # Build the prefix to strip: {project_id}/{run_id}/
        prefix = f"{project_id}/{run_name}/"

        if file_path.startswith(prefix):
            return file_path[len(prefix) :]

        # Also try with just run_name in case project_id format differs
        # Path might start with just run_id/
        run_prefix = f"{run_name}/"
        if file_path.startswith(run_prefix):
            return file_path[len(run_prefix) :]

        return file_path

    # =========================================================================
    # DataLoader Interface Implementation
    # =========================================================================
    # These methods implement the abstract DataLoader interface, providing
    # LitLogger-specific behavior for experiment/run management.

    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """
        Create or get a LitLogger experiment identifier.

        Note: In LitLogger, actual experiment creation happens during litlogger.init()
        which is called in upload_run_data. This method just prepares the identifier.

        Args:
            project_id: Neptune project ID
            experiment_name: Name for the experiment grouping

        Returns:
            Experiment identifier to be used when creating runs
        """
        # Create a unique experiment identifier from project and experiment name
        experiment_key = f"{project_id}/{experiment_name}"

        if self.name_prefix:
            experiment_key = f"{self.name_prefix}/{experiment_key}"

        return TargetExperimentId(experiment_key)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """
        Find an existing run by name in LitLogger.

        Note: LitLogger doesn't provide an API to search for existing experiments
        by name, so this always returns None. This means runs will be re-created
        if the migration is run multiple times.

        Args:
            project_id: Neptune project ID (unused)
            run_name: Name of the run to find
            experiment_id: Optional experiment identifier (unused)

        Returns:
            Always None (LitLogger doesn't support run lookup)
        """
        self._logger.debug(
            f"LitLogger doesn't support run lookup. Run '{run_name}' will be created."
        )
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
        Prepare a LitLogger experiment (run) for deferred creation.

        This method doesn't actually create the experiment yet - it stores the
        configuration for later use in upload_run_data(). We defer creation so
        we can extract parameters from the first data chunk and include them
        as experiment metadata.

        Each Neptune project maps to a separate Lightning.ai teamspace, and the
        run name becomes the experiment name within that teamspace.

        Args:
            project_id: Neptune project ID (e.g., "workspace/project")
            run_name: Name of the run (e.g., "RUN-123")
            experiment_id: Optional experiment identifier (unused, kept for interface compatibility)
            parent_run_id: Optional parent run ID (logged as warning - not supported)
            fork_step: Optional fork step (not supported by LitLogger)
            step_multiplier: Optional step multiplier (not used for run creation)

        Returns:
            Run ID (experiment name) that will be used in LitLogger
        """
        # Build experiment name from run_name only (project becomes teamspace)
        experiment_name = self._get_experiment_name(run_name)

        # Derive teamspace from project_id (each project gets its own teamspace)
        teamspace_name = self._get_teamspace_name(project_id)

        try:
            teamspace = Teamspace(
                name=teamspace_name,
                user=self.owner if isinstance(self.owner, User) else None,
                org=self.owner if isinstance(self.owner, Organization) else None,
            )
        except Exception:
            self._logger.info(f"Teamspace {teamspace_name} not found, creating it")
            teamspace = self.owner.create_teamspace(teamspace_name)

        # Warn about unsupported features
        if parent_run_id:
            self._logger.warning(
                f"LitLogger doesn't support parent-child run relationships. "
                f"Parent run '{parent_run_id}' will be ignored for run '{run_name}'."
            )

        # Store experiment info for deferred creation in upload_run_data
        # This allows us to extract parameters from data and use them as metadata
        self._pending_experiment = {
            "experiment_name": experiment_name,
            "teamspace": teamspace,
            "project_id": project_id,
            "run_name": run_name,
        }

        run_id = TargetRunId(experiment_name)
        self._current_run_id = run_id

        self._logger.info(
            f"Prepared LitLogger experiment '{experiment_name}' in teamspace '{teamspace.name}' for run '{run_name}'"
        )
        return run_id

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: AnyPath,
        step_multiplier: int,
    ) -> None:
        """
        Upload all data for a single run to LitLogger.

        This is the main entry point for uploading run data. It processes data
        in chunks (from the generator) and handles:
        1. Experiment creation (on first chunk, with parameters as metadata)
        2. Metrics upload (float_series data)
        3. Artifacts upload (files, histograms, string series)

        Args:
            run_data: PyArrow table generator yielding chunks of run data.
                Each chunk contains rows with columns like 'attribute_type',
                'attribute_path', 'float_value', 'file_value', etc.
            run_id: Run ID (experiment name) returned by create_run()
            files_directory: Base directory where exported files are stored
            step_multiplier: Multiplier for converting decimal steps to integers

        Raises:
            RuntimeError: If create_run() wasn't called first for this run_id
        """
        try:
            # Validate that create_run was called first
            if self._pending_experiment is None or self._current_run_id != run_id:
                self._logger.error(
                    f"Run {run_id} is not prepared. Call create_run first."
                )
                raise RuntimeError(f"Run {run_id} is not prepared")

            first_chunk = True
            for run_data_part in run_data:
                # Convert PyArrow table to pandas for easier manipulation
                run_df = run_data_part.to_pandas()

                # On first chunk, create the actual LitLogger experiment
                # We do this here (not in create_run) so we can extract parameters
                # from the data and include them as experiment metadata
                if first_chunk:
                    self._create_experiment_with_params(run_df)
                    first_chunk = False

                # Upload metrics and artifacts from this chunk
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

            # Finalize the experiment (required by LitLogger to complete upload)
            if self._active_experiment:
                self._active_experiment.finalize()

            # Reset state for next run
            self._active_experiment = None
            self._current_run_id = None
            self._pending_experiment = None

            self._logger.info(f"Successfully uploaded run {run_id} to LitLogger")

        except Exception:
            # Log error and clean up state on failure
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            if self._active_experiment:
                try:
                    self._active_experiment.finalize()
                except Exception:
                    pass  # Ignore finalize errors during cleanup
                self._active_experiment = None
                self._current_run_id = None
                self._pending_experiment = None
            raise

    # =========================================================================
    # Experiment Creation and Parameter Handling
    # =========================================================================

    def _create_experiment_with_params(self, run_df: pd.DataFrame) -> None:
        """
        Create the actual LitLogger experiment with parameters as metadata.

        Called on the first data chunk to initialize the experiment. Extracts
        parameter values (float, int, string, bool, datetime, string_set) from
        the data and stores them as experiment metadata.

        Each Neptune project maps to a separate Lightning.ai teamspace, derived
        from the project_id stored in the pending experiment.

        Args:
            run_df: First chunk of run data (pandas DataFrame)

        Raises:
            RuntimeError: If no pending experiment is configured
        """

        if self._pending_experiment is None:
            raise RuntimeError("No pending experiment")

        # Extract parameters from data to use as experiment metadata
        # This allows parameters to be searchable/filterable in LitLogger UI
        metadata = self._extract_parameters_as_metadata(run_df)

        # Add Neptune origin info for traceability
        metadata["neptune_project"] = self._pending_experiment["project_id"]
        metadata["neptune_run"] = self._pending_experiment["run_name"]

        # Use teamspace from project_id, or fallback to global teamspace if set
        teamspace = self._pending_experiment.get("teamspace")

        # Initialize the LitLogger experiment
        self._active_experiment = litlogger.init(
            name=self._pending_experiment["experiment_name"],
            teamspace=teamspace,
            metadata=metadata,
            store_step=True,  # Enable step tracking for metrics
            store_created_at=True,  # Store timestamps
            print_url=False,  # Don't print URL to stdout
        )

        self._logger.info(
            f"Created LitLogger experiment '{self._pending_experiment['experiment_name']}' "
            f"in teamspace '{teamspace}' with {len(metadata)} metadata fields"
        )

    def _extract_parameters_as_metadata(self, run_df: pd.DataFrame) -> Dict[str, str]:
        """
        Extract parameters from run data and convert to metadata dictionary.

        LitLogger metadata is stored as string key-value pairs. This method
        extracts Neptune parameters (scalar values) and converts them to strings.

        Supported Neptune types:
        - float  -> string representation of float
        - int    -> string representation of int
        - string -> string value directly
        - bool   -> "True" or "False"
        - datetime -> ISO format string
        - string_set -> comma-separated values

        Args:
            run_df: Run data DataFrame

        Returns:
            Dictionary of parameter names to string values
        """
        metadata: Dict[str, str] = {}

        # These Neptune types represent scalar parameters (not series/sequences)
        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_df[run_df["attribute_type"].isin(param_types)]

        if param_data.empty:
            return metadata

        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Convert each type to string, checking for NaN values
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                metadata[attr_name] = str(row["float_value"])
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                metadata[attr_name] = str(int(row["int_value"]))
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                metadata[attr_name] = str(row["string_value"])
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                metadata[attr_name] = str(bool(row["bool_value"]))
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                metadata[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                metadata[attr_name] = ",".join(row["string_set_value"])

        return metadata

    # =========================================================================
    # Metrics Upload
    # =========================================================================

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """
        Upload numeric parameters as metrics to LitLogger experiment.

        Note: This method is separate from _extract_parameters_as_metadata.
        Parameters are stored as metadata (strings) during experiment init,
        but this method also logs numeric parameters as metrics at step=0
        so they appear in the metrics view.

        Only numeric types are logged (float, int, bool as 0/1). Non-numeric
        types (string, datetime, string_set) are only stored as metadata.

        Args:
            run_data: DataFrame containing run data
            run_id: Run ID in LitLogger (for logging)

        Raises:
            RuntimeError: If no active experiment
        """
        if self._active_experiment is None:
            raise RuntimeError("No active experiment")

        # Filter to parameter types (scalar values, not series)
        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        # LitLogger metrics API only accepts numeric values
        # Convert parameters to float where possible
        numeric_params = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])
            # Prefix with 'param_' to distinguish from time-series metrics
            param_key = f"param_{attr_name}"

            # Convert to float based on type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                numeric_params[param_key] = float(row["float_value"])
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                numeric_params[param_key] = float(row["int_value"])
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                numeric_params[param_key] = 1.0 if row["bool_value"] else 0.0
            # Non-numeric types (string, datetime, string_set) are skipped
            # They're already stored as metadata in _extract_parameters_as_metadata

        if numeric_params:
            # Log all parameters at step 0 (they don't have time-series data)
            self._active_experiment.log_metrics(numeric_params, step=0)
            self._logger.info(
                f"Uploaded {len(numeric_params)} numeric parameters for run {run_id}"
            )

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """
        Upload metrics (float_series) to LitLogger experiment.

        Extracts all float_series data from the run and uploads them as
        time-series metrics with step information.

        Args:
            run_data: DataFrame containing run data
            run_id: Run ID in LitLogger (for logging)
            step_multiplier: Multiplier for converting decimal steps to integers

        Raises:
            RuntimeError: If no active experiment
        """
        if self._active_experiment is None:
            raise RuntimeError("No active experiment")

        # Filter to float_series (time-series numeric data)
        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        self._log_metrics_batch(metrics_data, step_multiplier)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def _log_metrics_batch(
        self, metrics_data: pd.DataFrame, step_multiplier: int
    ) -> None:
        """
        Batch upload all metrics in a single API call for efficiency.

        Instead of making one API call per metric value, this groups all values
        by metric name and uploads them together. This significantly reduces
        API overhead for runs with many metric values.

        Args:
            metrics_data: DataFrame containing float_series data with columns:
                - attribute_path: metric name (e.g., "metrics/train/loss")
                - step: training step (Decimal)
                - float_value: metric value
            step_multiplier: Multiplier for converting decimal steps to integers

        Format sent to LitLogger:
            {
                "metrics_train_loss": [
                    {"step": 100, "value": 0.5},
                    {"step": 200, "value": 0.3},
                    ...
                ],
                "metrics_train_accuracy": [...],
            }
        """
        metrics_dict = {}

        # Group by metric name to batch all values for each metric
        for attr_path, group in metrics_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Collect all (step, value) pairs for this metric
            values = []
            for _, row in group.iterrows():
                # Skip rows with missing values
                if pd.notna(row["float_value"]) and pd.notna(row["step"]):
                    step = self._convert_step_to_int(row["step"], step_multiplier)
                    values.append({"step": step, "value": float(row["float_value"])})

            if values:
                metrics_dict[attr_name] = values

        # Upload all metrics in a single batch call
        if metrics_dict and self._active_experiment:
            self._active_experiment.log_metrics_batch(metrics_dict)

    # =========================================================================
    # Artifacts Upload
    # =========================================================================

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: AnyPath,
        step_multiplier: int,
    ) -> None:
        """
        Upload files and artifacts to LitLogger experiment.

        Handles multiple artifact types from Neptune:
        - file, file_set, artifact: Regular files and directories
        - file_series: Sequence of files (e.g., images logged at different steps)
        - string_series: Text logs converted to .txt files
        - histogram_series: Histogram data rendered as PNG images

        Args:
            run_data: DataFrame containing run data with columns like
                'attribute_type', 'attribute_path', 'file_value', etc.
            run_id: Run ID in LitLogger (used for logging)
            files_base_path: Base directory where exported files are stored
            step_multiplier: Multiplier for converting decimal steps to integers
        """
        if self._active_experiment is None:
            raise RuntimeError("No active experiment")

        # Collect all files to upload as tuples of (local_path, remote_path, is_temp_file)
        # is_temp_file=True means the file should be deleted after upload
        files_to_upload: List[tuple[str, str, bool]] = []

        # -------------------------------------------------------------------------
        # 1. Handle regular files (file, file_set, artifact types)
        # -------------------------------------------------------------------------
        # These are files/directories exported directly from Neptune
        file_data = run_data[
            run_data["attribute_type"].isin(["file", "file_set", "artifact"])
        ]
        for _, row in file_data.iterrows():
            # Skip rows with missing or invalid file_value
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_rel_path = row["file_value"]["path"]
                file_path = files_base_path / file_rel_path

                # Strip Neptune project/run prefix (e.g., "project/run-id/") for cleaner paths
                remote_path = self._strip_neptune_path_prefix(file_rel_path)

                if file_path.exists():
                    if file_path.is_file():
                        files_to_upload.append((str(file_path), remote_path, False))
                    else:
                        # For directories (file_set), recursively collect all files inside
                        for child_file in file_path.rglob("*"):
                            if child_file.is_file():
                                child_rel_path = str(
                                    child_file.relative_to(files_base_path)
                                )
                                child_remote_path = self._strip_neptune_path_prefix(
                                    child_rel_path
                                )
                                files_to_upload.append(
                                    (str(child_file), child_remote_path, False)
                                )
                else:
                    self._logger.warning(f"File not found: {file_path}")

        # -------------------------------------------------------------------------
        # 2. Handle file series (sequence of files logged at different steps)
        # -------------------------------------------------------------------------
        # Similar to regular files but typically images/checkpoints logged over time
        file_series_data = run_data[run_data["attribute_type"] == "file_series"]
        for _, row in file_series_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_rel_path = row["file_value"]["path"]
                file_path = files_base_path / file_rel_path
                remote_path = self._strip_neptune_path_prefix(file_rel_path)

                # Only upload if it's an existing file (not a directory)
                if file_path.exists() and file_path.is_file():
                    files_to_upload.append((str(file_path), remote_path, False))
                elif not file_path.exists():
                    self._logger.warning(f"File not found: {file_path}")

        # -------------------------------------------------------------------------
        # 3. Handle string series (text logs converted to .txt files)
        # -------------------------------------------------------------------------
        # Combine all string values for each attribute into a single text file
        # Format: "step; timestamp; value" per line
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)
            remote_path = f"string_series/{attr_name}.txt"

            # Create a temporary file to hold the combined text content
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"_{attr_name}.txt",
                encoding="utf-8",
                delete=False,  # Keep file until we upload it
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
                        tmp_file.write(
                            f"{series_step}; {timestamp}; {row['string_value']}\n"
                        )

                # Mark as temp file so it gets cleaned up after upload
                files_to_upload.append((tmp_file.name, remote_path, True))

        # -------------------------------------------------------------------------
        # 4. Handle histogram series (render as PNG images)
        # -------------------------------------------------------------------------
        # Convert histogram data (edges + values) into bar chart images
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        if not histogram_series_data.empty:
            try:
                import matplotlib.pyplot as plt

                for attr_path, group in histogram_series_data.groupby("attribute_path"):
                    attr_name = self._sanitize_attribute_name(attr_path)

                    for _, row in group.iterrows():
                        if pd.notna(row["histogram_value"]) and isinstance(
                            row["histogram_value"], dict
                        ):
                            step = (
                                self._convert_step_to_int(row["step"], step_multiplier)
                                if pd.notna(row["step"])
                                else None
                            )
                            hist = row["histogram_value"]
                            edges = hist.get("edges", [])
                            values = hist.get("values", [])

                            # Need at least 2 edges to define bins
                            if edges and values and len(edges) > 1:
                                fig, ax = plt.subplots(figsize=(8, 6))

                                # Calculate bin widths and centers from edges
                                bin_widths = [
                                    edges[i + 1] - edges[i]
                                    for i in range(len(edges) - 1)
                                ]
                                bin_centers = [
                                    (edges[i] + edges[i + 1]) / 2
                                    for i in range(len(edges) - 1)
                                ]

                                ax.bar(
                                    bin_centers,
                                    values,
                                    width=bin_widths,
                                    edgecolor="black",
                                    alpha=0.7,
                                )
                                ax.set_xlabel("Value")
                                ax.set_ylabel("Count")
                                title = f"{attr_name}"
                                if step is not None:
                                    title += f" (step {step})"
                                ax.set_title(title)

                                # Save plot to temp file
                                step_str = step if step is not None else "none"
                                remote_path = (
                                    f"histograms/{attr_name}_step{step_str}.png"
                                )
                                with tempfile.NamedTemporaryFile(
                                    suffix=f"_{attr_name}_step{step_str}.png",
                                    delete=False,
                                ) as tmp_file:
                                    fig.savefig(
                                        tmp_file.name, dpi=100, bbox_inches="tight"
                                    )
                                    files_to_upload.append(
                                        (tmp_file.name, remote_path, True)
                                    )

                                plt.close(fig)

            except ImportError:
                self._logger.warning(
                    "matplotlib not installed, skipping histogram images"
                )

        # -------------------------------------------------------------------------
        # 5. Upload all collected files to LitLogger
        # -------------------------------------------------------------------------
        if files_to_upload:
            file_paths = [local_path for local_path, _, _ in files_to_upload]
            remote_paths = [remote_path for _, remote_path, _ in files_to_upload]
            self._active_experiment.log_files(file_paths, remote_paths=remote_paths)

            # Clean up temporary files (string_series .txt and histogram .png files)
            for local_path, _, is_temp in files_to_upload:
                if is_temp:
                    try:
                        Path(local_path).unlink()
                    except Exception:
                        pass  # Ignore cleanup errors - temp dir will clean eventually

        self._logger.info(f"Uploaded artifacts for run {run_id}")

    def _validate_owner(self, owner_name: Optional[str]) -> Union[User, Organization]:
        from lightning_sdk.utils.resolve import _get_authed_user

        authed_user = _get_authed_user()

        if owner_name is None or authed_user.name == owner_name:
            return authed_user

        for org in authed_user.organizations:
            if org.name == owner_name:
                return org

        raise ValueError(
            f"Owner {owner_name} not found! Either it doesn't exist or the authenticated user is not a member of the organization"
        )
