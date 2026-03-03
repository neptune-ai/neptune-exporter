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

"""Loader for migrating Neptune data to GoodSeed experiment tracker."""

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import pandas as pd
import pyarrow as pa

from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId

import goodseed
import goodseed.config as goodseed_config
from goodseed.sync import upload_run as sync_upload_run


# Neptune sys/ attributes to skip (platform internals + attributes extracted separately).
# Other attributes like sys/creation_time, sys/failed, sys/tags, sys/description pass through.
_SKIP_SYS_ATTRIBUTES = {
    "sys/id",
    "sys/custom_run_id",
    "sys/name",
    "sys/state",
    "sys/owner",
    "sys/size",
    "sys/ping_time",
    "sys/running_time",
    "sys/monitoring_time",
}

# Neptune attribute types that map to GoodSeed configs
_PARAM_TYPES = {"float", "int", "string", "bool", "datetime", "string_set"}

# Neptune attribute types we log warnings for (unsupported)
_FILE_TYPES = {"file", "file_series", "file_set", "artifact"}


class GoodseedLoader(DataLoader):
    """
    Loads Neptune data from parquet files into GoodSeed experiment tracker.

    Supports two storage modes:
    - **local**: Writes directly to GoodSeed's local SQLite storage (default).
    - **remote**: Writes to local SQLite first, then uploads to a GoodSeed
      server via API. Requires an API key.

    Neptune Concept -> GoodSeed Concept
    ------------------------------------
    - Project       -> Project (passed through or remapped for remote)
    - Run           -> Run (SQLite file, optionally synced to server)
    - Parameters    -> Configs (log_configs)
    - Float Series  -> Metrics (log_metrics)
    - String Series -> String Series (log_strings)
    - sys/name      -> name
    - Files         -> Skipped (not supported)
    - Histograms    -> Skipped (not supported)

    Usage:
        loader = GoodseedLoader()
        loader.create_run(project_id, run_name)
        loader.upload_run_data(data_generator, run_id, files_dir, step_multiplier)
    """

    def __init__(
        self,
        storage_mode: str = "auto",
        goodseed_project: Optional[str] = None,
        goodseed_api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ):
        """
        Initialize GoodSeed loader.

        Args:
            storage_mode: Target mode: ``auto``, ``local`` (sqlite files), or
                ``remote`` (API). ``auto`` selects ``remote`` when API key is set.
            goodseed_project: Override project name for all imported runs. If not set,
                uses the Neptune project ID directly.
            goodseed_api_key: API key for remote mode.
            name_prefix: Optional prefix for run names.
            show_client_logs: Enable verbose logging (unused, kept for interface consistency).
        """
        normalized_storage = storage_mode.lower()
        if normalized_storage not in {"auto", "local", "remote"}:
            raise RuntimeError(
                f"Invalid GoodSeed storage mode: {storage_mode!r}. "
                "Use 'auto', 'local', or 'remote'."
            )
        if normalized_storage == "auto":
            normalized_storage = "remote" if goodseed_api_key else "local"
        self._storage_mode = normalized_storage

        if self._storage_mode == "remote" and not goodseed_api_key:
            raise RuntimeError(
                "GoodSeed API key is required in remote mode. "
                "Set GOODSEED_API_KEY or pass --goodseed-api-key."
            )

        self._goodseed_project = goodseed_project
        self._goodseed_api_key = goodseed_api_key
        self._name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)

        # Active run state
        self._active_run: Optional[Any] = None
        self._current_run_id: Optional[TargetRunId] = None
        self._pending_run: Optional[Dict[str, Any]] = None
        self._remote_project_cache: Dict[str, Dict[str, Any]] = {}
        self._me_name: Optional[str] = None
        self._warned_remaps: set[str] = set()

        # Track warnings to avoid spamming
        self._warned_file_skip = False
        self._warned_histogram_skip = False

    def _get_run_name(self, run_name: str) -> str:
        """Build a GoodSeed run name, optionally with prefix."""
        if self._name_prefix:
            return f"{self._name_prefix}_{run_name}"
        return run_name

    def _get_project(self, project_id: str) -> str:
        """Determine the GoodSeed project name."""
        if self._goodseed_project:
            return self._goodseed_project
        return project_id

    def _resolve_remote_target_project(self, source_project_id: str) -> dict[str, Any]:
        cached = self._remote_project_cache.get(source_project_id)
        if cached is not None:
            return cached

        source_workspace = ""
        source_project = source_project_id
        parts = source_project_id.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            source_workspace, source_project = parts

        if self._me_name is None:
            profile = goodseed.me(api_key=self._goodseed_api_key or "")
            me_name = profile.get("name")
            if not me_name:
                raise RuntimeError(
                    "Could not resolve user name from /auth/me response."
                )
            self._me_name = str(me_name)

        requested_workspace = source_workspace
        requested_project = source_project
        if self._goodseed_project:
            parts = self._goodseed_project.split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                requested_workspace, requested_project = parts
            else:
                requested_workspace = ""
                requested_project = self._goodseed_project

        workspace = requested_workspace or self._me_name
        target_project = requested_project
        remapped = False

        # Always write to the authenticated user's workspace.
        # Cross-workspace projects are namespaced as "<workspace>_<project>".
        if requested_workspace and requested_workspace != self._me_name:
            workspace = self._me_name
            target_project = f"{requested_workspace}_{requested_project}"
            remapped = True

        payload = goodseed.ensure_project(
            workspace=workspace,
            project_name=target_project,
            storage="remote",
            api_key=self._goodseed_api_key,
        )
        info = {
            "workspace": workspace,
            "project": payload.get("name", target_project),
            "remapped": remapped,
        }
        self._remote_project_cache[source_project_id] = info

        warning_key = f"{source_project_id}->{info['workspace']}/{info['project']}"
        if info["remapped"] and warning_key not in self._warned_remaps:
            self._warned_remaps.add(warning_key)
            self._logger.warning(
                "Source project '%s' mapped to default workspace target '%s/%s'.",
                source_project_id,
                info["workspace"],
                info["project"],
            )
        return info

    def _convert_step(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to GoodSeed integer step."""
        return int(float(step) * step_multiplier)

    # DataLoader interface

    def create_experiment(
        self, project_id: ProjectId, experiment_name: str
    ) -> TargetExperimentId:
        """
        Return experiment identifier.

        GoodSeed doesn't have a separate experiment creation step - the experiment
        name is set when creating a Run. We just return the name for later use.
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """
        Check if a run already exists in GoodSeed.

        Looks for the SQLite file at the expected path. If found, returns the
        run ID so LoaderManager can skip re-importing.
        """
        gs_run_name = self._get_run_name(run_name)
        if self._storage_mode == "remote":
            return self._find_remote_run(project_id, gs_run_name)

        gs_project = self._get_project(project_id)
        db_path = goodseed_config.get_run_db_path(gs_project, gs_run_name)
        if db_path.exists():
            self._logger.info(
                f"Run '{gs_run_name}' already exists in project '{gs_project}', skipping."
            )
            return TargetRunId(gs_run_name)
        return None

    def _find_remote_run(
        self,
        project_id: ProjectId,
        gs_run_name: str,
    ) -> Optional[TargetRunId]:
        """Check run existence in remote GoodSeed project using list-runs API."""
        target = self._resolve_remote_target_project(project_id)
        runs = goodseed.list_runs(
            workspace=target["workspace"],
            project_name=target["project"],
            storage="remote",
            api_key=self._goodseed_api_key,
        )
        for run in runs:
            if run.get("run_id") == gs_run_name:
                self._logger.info(
                    "Run '%s' already exists in remote project '%s/%s', skipping.",
                    gs_run_name,
                    target["workspace"],
                    target["project"],
                )
                return TargetRunId(gs_run_name)
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
        Prepare a GoodSeed run for deferred creation.

        Actual Run creation happens in upload_run_data so we can extract
        sys/name from the data to use as experiment_name.
        """
        gs_run_name = self._get_run_name(run_name)
        gs_project = self._get_project(project_id)
        remote_workspace = None
        if self._storage_mode == "remote":
            target = self._resolve_remote_target_project(project_id)
            remote_workspace = target["workspace"]
            gs_project = f"{remote_workspace}/{target['project']}"

        self._pending_run = {
            "run_name": gs_run_name,
            "project": gs_project,
            "remote_workspace": remote_workspace,
            "project_id": project_id,
            "original_run_name": run_name,
            "experiment_name": str(experiment_id) if experiment_id else None,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "fork_step": fork_step,
        }

        run_id = TargetRunId(gs_run_name)
        self._current_run_id = run_id

        self._logger.info(
            f"Prepared GoodSeed run '{gs_run_name}' in project '{gs_project}'"
        )
        return run_id

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """
        Upload all data for a single run to GoodSeed.

        Processes data chunks from the parquet generator and dispatches each
        row by attribute_type to the appropriate GoodSeed API.
        """
        if self._pending_run is None or self._current_run_id != run_id:
            raise RuntimeError(f"Run {run_id} is not prepared. Call create_run first.")

        run_closed = False
        try:
            first_chunk = True
            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()

                if first_chunk:
                    self._create_run_from_data(run_df)
                    first_chunk = False

                self._upload_parameters(run_df)
                self._upload_metrics(run_df, step_multiplier)
                self._upload_string_series(run_df, step_multiplier)
                self._warn_skipped_types(run_df)

            # Close the run locally first (sync needs the finalized DB)
            if self._active_run is not None:
                self._active_run.close()
                run_closed = True
                if self._storage_mode == "remote":
                    self._sync_remote_run(
                        project=self._pending_run["project"],
                        run_id=self._pending_run["run_name"],
                    )
                self._logger.info(f"Successfully uploaded run {run_id} to GoodSeed")

        except Exception:
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            if self._active_run is not None and not run_closed:
                try:
                    self._active_run.close(status="failed")
                except Exception:
                    pass
            raise
        finally:
            self._active_run = None
            self._current_run_id = None
            self._pending_run = None
            self._warned_file_skip = False
            self._warned_histogram_skip = False

    # Run creation

    def _create_run_from_data(self, run_df: pd.DataFrame) -> None:
        """Create the GoodSeed Run, extracting experiment_name from data if available."""
        if self._pending_run is None:
            raise RuntimeError("No pending run")

        # Extract sys/name, sys/creation_time, and sys/modification_time from data
        experiment_name = self._pending_run["experiment_name"]
        created_at = None
        modified_at = None

        for attr_path, attr_type, col in [
            ("sys/name", "string", "string_value"),
            ("sys/creation_time", "datetime", "datetime_value"),
            ("sys/modification_time", "datetime", "datetime_value"),
        ]:
            rows = run_df[
                (run_df["attribute_path"] == attr_path)
                & (run_df["attribute_type"] == attr_type)
            ]
            if not rows.empty:
                val = rows.iloc[0][col]
                if pd.notna(val):
                    if attr_path == "sys/name":
                        experiment_name = str(val)
                    elif attr_path == "sys/creation_time":
                        created_at = pd.Timestamp(val).isoformat()
                    elif attr_path == "sys/modification_time":
                        modified_at = pd.Timestamp(val).isoformat()

        self._active_run = goodseed.Run(
            name=experiment_name,
            project=self._pending_run["project"],
            run_id=self._pending_run["run_name"],
            created_at=created_at,
            modified_at=modified_at,
            storage="local",
        )

        # Log Neptune origin metadata as configs
        origin_configs = {
            "neptune/project_id": self._pending_run["project_id"],
            "neptune/run_id": self._pending_run["original_run_name"],
        }
        if self._pending_run.get("parent_run_id"):
            origin_configs["neptune/parent_run_id"] = self._pending_run["parent_run_id"]
        if self._pending_run.get("fork_step") is not None:
            origin_configs["neptune/fork_step"] = self._pending_run["fork_step"]

        self._active_run.log_configs(origin_configs)

    def _sync_remote_run(self, *, project: str, run_id: str) -> None:
        """Delegate remote upload to the GoodSeed package uploader."""
        if not self._goodseed_api_key:
            raise RuntimeError("Missing GoodSeed API key for remote upload.")
        db_path = goodseed_config.get_run_db_path(project, run_id)
        if not db_path.exists():
            raise RuntimeError(f"Run database not found for remote upload: {db_path}")
        sync_upload_run(db_path, self._goodseed_api_key)

    # Parameter upload

    def _upload_parameters(self, run_df: pd.DataFrame) -> None:
        """Upload Neptune parameters as GoodSeed configs."""
        if self._active_run is None:
            return

        param_data = run_df[run_df["attribute_type"].isin(_PARAM_TYPES)]
        if param_data.empty:
            return

        configs: Dict[str, Any] = {}
        for row in param_data.itertuples(index=False):
            attr_path = row.attribute_path

            # Skip most sys/ attributes
            if attr_path in _SKIP_SYS_ATTRIBUTES:
                continue

            value = self._extract_param_value(row)
            if value is not None:
                configs[attr_path] = value

        if configs:
            self._active_run.log_configs(configs)

    def _extract_param_value(self, row: Any) -> Any:
        """Extract a typed value from a parameter row."""
        attr_type = row.attribute_type

        if attr_type == "float" and pd.notna(row.float_value):
            return float(row.float_value)
        elif attr_type == "int" and pd.notna(row.int_value):
            return int(row.int_value)
        elif attr_type == "string" and pd.notna(row.string_value):
            return str(row.string_value)
        elif attr_type == "bool" and pd.notna(row.bool_value):
            return bool(row.bool_value)
        elif attr_type == "datetime" and pd.notna(row.datetime_value):
            return pd.Timestamp(row.datetime_value).isoformat()
        elif attr_type == "string_set" and row.string_set_value is not None:
            return ",".join(row.string_set_value)
        return None

    # Metrics upload

    def _upload_metrics(self, run_df: pd.DataFrame, step_multiplier: int) -> None:
        """Upload Neptune float_series as GoodSeed metrics."""
        if self._active_run is None:
            return

        metrics_data = run_df[run_df["attribute_type"] == "float_series"]
        if metrics_data.empty:
            return

        for row in metrics_data.itertuples(index=False):
            if pd.notna(row.float_value) and pd.notna(row.step):
                step = self._convert_step(row.step, step_multiplier)
                self._active_run.log_metrics(
                    {row.attribute_path: float(row.float_value)}, step=step
                )

    # String series upload

    def _upload_string_series(self, run_df: pd.DataFrame, step_multiplier: int) -> None:
        """Upload Neptune string_series as GoodSeed string series."""
        if self._active_run is None:
            return

        series_data = run_df[run_df["attribute_type"] == "string_series"]
        if series_data.empty:
            return

        for row in series_data.itertuples(index=False):
            if pd.notna(row.string_value):
                step = (
                    self._convert_step(row.step, step_multiplier)
                    if pd.notna(row.step)
                    else 0
                )
                self._active_run.log_strings(
                    {row.attribute_path: str(row.string_value)}, step=step
                )

    # Warnings for unsupported types

    def _warn_skipped_types(self, run_df: pd.DataFrame) -> None:
        """Log warnings for data types that cannot be imported to GoodSeed."""
        if not self._warned_file_skip:
            file_data = run_df[run_df["attribute_type"].isin(_FILE_TYPES)]
            if not file_data.empty:
                count = len(file_data)
                self._logger.warning(
                    f"Skipping {count} file attribute(s) - "
                    f"GoodSeed does not support file storage."
                )
                self._warned_file_skip = True

        if not self._warned_histogram_skip:
            hist_data = run_df[run_df["attribute_type"] == "histogram_series"]
            if not hist_data.empty:
                count = len(hist_data)
                self._logger.warning(
                    f"Skipping {count} histogram attribute(s) - "
                    f"GoodSeed does not support histograms."
                )
                self._warned_histogram_skip = True
