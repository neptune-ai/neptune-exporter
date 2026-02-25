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

"""ZenML loader implementation for uploading Neptune export data.

This module integrates with the ZenML Model Control Plane by mapping:
- Neptune project  -> ZenML Model
- Neptune run      -> ZenML Model Version
- Neptune metadata -> ZenML metadata via log_metadata()

ZenML is treated as an optional dependency. To use this loader, install:

    pip install "neptune-exporter[zenml]"

and ensure you are logged into a ZenML server (e.g., via `zenml login`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Optional,
    Set,
)

import pandas as pd
import pyarrow as pa

from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.storage.types import AnyPath
from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId

if TYPE_CHECKING:
    from zenml.models import ModelVersionResponse

try:
    from zenml.client import Client
    from zenml import log_metadata
    from zenml.artifacts.utils import save_artifact
    from zenml.model.utils import link_artifact_version_to_model_version
    from zenml.materializers import PathMaterializer

    ZENML_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when zenml is missing
    Client = None  # type: ignore[assignment,misc]
    log_metadata = None  # type: ignore[assignment,misc]
    save_artifact = None  # type: ignore[assignment]
    link_artifact_version_to_model_version = None  # type: ignore[assignment]
    PathMaterializer = None  # type: ignore[assignment,misc]
    ZENML_AVAILABLE = False


@dataclass
class SeriesStats:
    """Aggregate statistics for a single float series.

    These stats are used to log summary information for Neptune time-series
    attributes that do not have a direct visualization equivalent in ZenML.
    """

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    final_value: Optional[float] = None
    final_step: Optional[float] = None
    count: int = 0


SeriesStatsByAttr = Dict[str, SeriesStats]


class ZenMLLoader(DataLoader):
    """Loads Neptune data from parquet files into ZenML Model Versions."""

    def __init__(
        self,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ) -> None:
        """Initialize ZenML loader.

        Args:
            name_prefix: Optional prefix for ZenML model names. Useful for
                namespacing imports when multiple exporters share a ZenML server.
            show_client_logs: If True, enable info-level logs from ZenML client;
                otherwise, suppress them to only show exporter logs.
        """
        if not ZENML_AVAILABLE:
            raise RuntimeError(
                "ZenML is not installed. Install with "
                "`pip install 'neptune-exporter[zenml]'` to use the ZenML loader."
            )

        self._logger = logging.getLogger(__name__)
        self._client = Client()
        self.name_prefix = name_prefix

        # Cache mapping Neptune ProjectId -> ZenML Model ID (wrapped as TargetExperimentId)
        self._project_model_cache: Dict[ProjectId, TargetExperimentId] = {}

        # Cache mapping model_version_id -> model_id for artifact linking
        # (ZenML's get_model_version requires both model and version identifiers)
        self._mv_to_model_cache: Dict[str, str] = {}

        # Configure ZenML logging verbosity so CLI flags behave consistently
        zenml_logger = logging.getLogger("zenml")
        if show_client_logs:
            zenml_logger.setLevel(logging.INFO)
        else:
            zenml_logger.setLevel(logging.ERROR)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _sanitize_key_part(self, key_part: str) -> str:
        """Sanitize a single key part (no slashes) for ZenML metadata.

        - Allow alphanumerics, underscores (_), dashes (-), periods (.),
          and spaces.
        - Replace anything else with an underscore.
        - Truncate overly long keys to a reasonable limit.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s]", "_", key_part)

        max_len = 100
        if len(sanitized) > max_len:
            self._logger.warning(
                "Truncated metadata key part '%s' to '%s' to satisfy length limits.",
                key_part,
                sanitized[:max_len],
            )
            sanitized = sanitized[:max_len]
        if not sanitized:
            sanitized = "_key"
        return sanitized

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """Sanitize Neptune attribute path to a ZenML-safe metadata key.

        ZenML metadata keys are ultimately stored in a backing store that
        prefers simple, portable strings. This mirrors the MLflow sanitization:
        - Allow alphanumerics, underscores (_), dashes (-), periods (.),
          spaces, and slashes (/).
        - Replace anything else with an underscore.
        - Truncate overly long keys to a reasonable limit.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s/]", "_", attribute_path)

        max_len = 250
        if len(sanitized) > max_len:
            original = sanitized
            sanitized = sanitized[:max_len]
            self._logger.warning(
                "Truncated metadata key '%s' to '%s' to satisfy length limits.",
                original,
                sanitized,
            )
        if not sanitized:
            sanitized = "_attribute"
        return sanitized

    def _set_nested_value(
        self,
        metadata: Dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        """Set a value in a nested dictionary structure based on a path.

        Given a path like "model/params/dense_units" and value 128, this creates:
        {"model": {"params": {"dense_units": 128}}}

        This allows ZenML to display top-level keys as separate cards in the
        dashboard, making metadata more organized and navigable.
        """
        parts = path.split("/")
        # Sanitize each part
        parts = [self._sanitize_key_part(p) for p in parts if p]

        if not parts:
            return

        # Navigate/create nested structure
        current = metadata
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Conflict: existing value is not a dict, create a special key
                self._logger.warning(
                    "Path conflict at '%s' while setting '%s'; using '_value' subkey",
                    part,
                    path,
                )
                current[part] = {"_value": current[part]}
            current = current[part]

        # Set the final value
        final_key = parts[-1]
        if final_key in current and isinstance(current[final_key], dict):
            # The key exists as a dict (intermediate node), store value under _value
            current[final_key]["_value"] = value
        else:
            current[final_key] = value

    def _sanitize_name(self, name: str, max_len: int = 250) -> str:
        """Sanitize generic names (models, model versions) for ZenML.

        The goal is to keep names human-readable while avoiding characters that
        may cause issues in URLs or backends.
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\s/]", "_", name)
        if len(sanitized) > max_len:
            self._logger.warning(
                "Truncated name '%s' to '%s'", name, sanitized[:max_len]
            )
            sanitized = sanitized[:max_len]
        if not sanitized:
            sanitized = "neptune-import"
        return sanitized

    def _get_model_name(self, project_id: ProjectId) -> str:
        """Derive a ZenML Model name from a Neptune project ID.

        Example:
            project_id = "org/my-project"
            -> "neptune-export-org-my-project"
            -> "prefix-neptune-export-org-my-project" (if name_prefix set)
        """
        project_slug = str(project_id).replace("/", "-")
        base = f"neptune-export-{project_slug}"
        if self.name_prefix:
            base = f"{self.name_prefix}-{base}"
        return self._sanitize_name(base)

    def _get_or_create_model(self, project_id: ProjectId) -> TargetExperimentId:
        """Get or create the ZenML Model corresponding to a Neptune project."""
        if project_id in self._project_model_cache:
            return self._project_model_cache[project_id]

        model_name = self._get_model_name(project_id)

        try:
            models_page = self._client.list_models(name=model_name)
            models = list(models_page)
        except Exception:
            self._logger.error(
                "Error listing ZenML models for name '%s'", model_name, exc_info=True
            )
            models = []

        if models:
            model = models[0]
            self._logger.info(
                "Using existing ZenML model '%s' (id=%s) for Neptune project '%s'",
                getattr(model, "name", model_name),
                getattr(model, "id", "<unknown>"),
                project_id,
            )
        else:
            try:
                model = self._client.create_model(
                    name=model_name,
                    description=f"Imported from Neptune project '{project_id}'",
                    tags=[
                        "neptune-import",
                        f"neptune-project:{project_id}",
                    ],
                )
                self._logger.info(
                    "Created ZenML model '%s' (id=%s) for Neptune project '%s'",
                    getattr(model, "name", model_name),
                    getattr(model, "id", "<unknown>"),
                    project_id,
                )
            except Exception:
                self._logger.error(
                    "Error creating ZenML model '%s' for Neptune project '%s'",
                    model_name,
                    project_id,
                    exc_info=True,
                )
                raise

        model_id = TargetExperimentId(str(getattr(model, "id")))
        self._project_model_cache[project_id] = model_id
        return model_id

    def _accumulate_scalar_metadata(
        self,
        df: pd.DataFrame,
        all_metadata: Dict[str, Any],
        tags: Set[str],
        experiment_tags: Set[str],
        sys_info: Dict[str, Any],
    ) -> None:
        """Accumulate scalar Neptune attributes into ZenML metadata.

        This covers attribute types:
            float, int, string, bool, datetime, string_set

        Special handling is applied for:
        - sys/tags       -> ZenML tags + metadata
        - sys/group_tags -> experiment-level grouping metadata
        - sys/name       -> stored in sys_info for potential later use
        - sys/description-> stored in sys_info for potential later use
        """
        scalar_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        scalar_df = df[df["attribute_type"].isin(scalar_types)]

        if scalar_df.empty:
            return

        for _, row in scalar_df.iterrows():
            attr_path = row["attribute_path"]
            attr_type = row["attribute_type"]

            value: Any = None

            if attr_type == "float":
                v = row["float_value"]
                if pd.notna(v):
                    value = float(v)
            elif attr_type == "int":
                v = row["int_value"]
                if pd.notna(v):
                    value = int(v)
            elif attr_type == "string":
                v = row["string_value"]
                if pd.notna(v):
                    value = str(v)
            elif attr_type == "bool":
                v = row["bool_value"]
                if pd.notna(v):
                    value = bool(v)
            elif attr_type == "datetime":
                v = row["datetime_value"]
                if pd.notna(v):
                    # pandas / arrow timestamps stringify nicely to ISO
                    value = str(v)
            elif attr_type == "string_set":
                v = row["string_set_value"]
                if v is not None:
                    # Ensure we always store plain Python lists
                    value = [str(item) for item in list(v)]

            if value is None:
                continue

            # Track special system attributes separately for later mapping
            if attr_path == "sys/tags" and isinstance(value, list):
                tags.update(str(t) for t in value)
            elif attr_path == "sys/group_tags" and isinstance(value, list):
                experiment_tags.update(str(t) for t in value)
            elif attr_path == "sys/name":
                sys_info["sys/name"] = value
            elif attr_path == "sys/description":
                sys_info["sys/description"] = value

            # Use nested structure for ZenML dashboard cards
            self._set_nested_value(all_metadata, attr_path, value)

    def _accumulate_float_series_stats(
        self,
        df: pd.DataFrame,
        series_stats: SeriesStatsByAttr,
    ) -> None:
        """Accumulate summary statistics for float_series attributes.

        For each attribute_path of type float_series, we track:
        - min_value
        - max_value
        - final_value (value at largest step, or first if no step info)
        - count
        """
        series_df = df[df["attribute_type"] == "float_series"]
        if series_df.empty:
            return

        for _, row in series_df.iterrows():
            value_raw = row["float_value"]
            if pd.isna(value_raw):
                continue

            value = float(value_raw)
            attr_path = row["attribute_path"]

            stats = series_stats.setdefault(attr_path, SeriesStats())

            # Update min and max
            if stats.min_value is None or value < stats.min_value:
                stats.min_value = value
            if stats.max_value is None or value > stats.max_value:
                stats.max_value = value

            # Determine step ordering to pick a "final" value
            step_raw = row["step"]
            step_val: Optional[float]
            if pd.notna(step_raw):
                try:
                    step_val = float(step_raw)
                except Exception:
                    step_val = None
            else:
                step_val = None

            if step_val is not None:
                if stats.final_step is None or step_val >= stats.final_step:
                    stats.final_step = step_val
                    stats.final_value = value
            elif stats.count == 0:
                # No step info: treat the first observation as "final" until
                # we see a later one with explicit step information.
                stats.final_value = value

            stats.count += 1

    def _accumulate_file_references(
        self,
        df: pd.DataFrame,
        file_refs: Dict[str, list[str]],
        files_directory: AnyPath,
    ) -> None:
        """Collect file and artifact references as metadata.

        Files are *not* uploaded to ZenML; instead, we store the relative
        paths produced by the export so users can retrieve them from disk.
        """
        file_types = {"file", "file_series", "file_set", "artifact"}
        file_df = df[df["attribute_type"].isin(file_types)]

        if file_df.empty:
            return

        for _, row in file_df.iterrows():
            file_value = row["file_value"]
            if not isinstance(file_value, dict):
                continue

            rel_path = file_value.get("path")
            if not rel_path:
                continue

            # Optionally warn if referenced file does not exist on disk
            abs_path = files_directory / rel_path
            if not abs_path.exists():
                self._logger.debug(
                    "Referenced file not found on disk for metadata: %s",
                    abs_path,
                )

            attr_path = row["attribute_path"]
            # Store with original path structure for nested grouping
            # Will be placed under "files/{original_path}"
            meta_key = f"files/{attr_path}"

            file_refs.setdefault(meta_key, []).append(str(rel_path))

    def _finalize_series_metadata(
        self,
        series_stats: SeriesStatsByAttr,
        all_metadata: Dict[str, Any],
    ) -> None:
        """Convert accumulated SeriesStats into metadata entries.

        Series stats are nested under a 'series' top-level key, then follow
        the original attribute path structure. For example:
        - attr_path "metrics/loss" -> series/metrics/loss/{min,max,final,count}
        """
        for attr_path, stats in series_stats.items():
            if stats.count == 0:
                continue

            # Prefix with "series" to group all time-series summaries together
            series_path = f"series/{attr_path}"

            if stats.min_value is not None:
                self._set_nested_value(
                    all_metadata, f"{series_path}/min", stats.min_value
                )
            if stats.max_value is not None:
                self._set_nested_value(
                    all_metadata, f"{series_path}/max", stats.max_value
                )
            if stats.final_value is not None:
                self._set_nested_value(
                    all_metadata, f"{series_path}/final", stats.final_value
                )

            self._set_nested_value(all_metadata, f"{series_path}/count", stats.count)

    def _merge_file_refs_into_metadata(
        self,
        file_refs: Dict[str, list[str]],
        all_metadata: Dict[str, Any],
    ) -> None:
        """Merge collected file references into the metadata dictionary.

        File references are nested under a 'files' top-level key, following
        the original attribute path. For example:
        - attr_path "artifacts/model" -> files/artifacts/model
        """
        for key, paths in file_refs.items():
            if not paths:
                continue
            value = paths[0] if len(paths) == 1 else paths
            self._set_nested_value(all_metadata, key, value)

    def _get_model_version_response(
        self, model_version_id: str
    ) -> Optional["ModelVersionResponse"]:
        """Fetch a ZenML ModelVersionResponse by ID, with caching.

        The response object is required by link_artifact_version_to_model_version(),
        which expects full response objects rather than just UUIDs.

        Args:
            model_version_id: The ZenML Model Version ID.

        Returns:
            The ModelVersionResponse object, or None if fetch fails.
        """
        # Check cache first
        cache_key = f"mv_{model_version_id}"
        if hasattr(self, "_mv_cache") and cache_key in self._mv_cache:
            return self._mv_cache[cache_key]

        # Initialize cache if needed
        if not hasattr(self, "_mv_cache"):
            self._mv_cache: Dict[str, "ModelVersionResponse"] = {}

        # ZenML's get_model_version() requires both model_id and model_version_id
        model_id = self._mv_to_model_cache.get(model_version_id)
        if not model_id:
            self._logger.warning(
                "Cannot fetch model version '%s': no model_id mapping found",
                model_version_id,
            )
            return None

        try:
            mv = self._client.get_model_version(model_id, model_version_id)
            self._mv_cache[cache_key] = mv
            return mv
        except Exception:
            self._logger.warning(
                "Failed to fetch model version '%s' for artifact linking",
                model_version_id,
                exc_info=True,
            )
            return None

    def _upload_artifact_to_zenml(
        self,
        local_path: AnyPath,
        artifact_name: str,
        model_version_id: str,
        neptune_attr_path: str,
    ) -> Optional[str]:
        """Upload a local file or directory as a ZenML artifact and link to model version.

        Args:
            local_path: The local path to the file or directory to upload.
            artifact_name: The name to use for the ZenML artifact.
            model_version_id: The ZenML Model Version ID to link the artifact to.
            neptune_attr_path: The original Neptune attribute path (for metadata).

        Returns:
            The artifact version ID if successful, None otherwise.
        """
        if save_artifact is None or link_artifact_version_to_model_version is None:
            self._logger.warning(
                "ZenML artifact save/link functions not available; skipping upload for %s",
                local_path,
            )
            return None

        if not local_path.exists():
            self._logger.warning(
                "Cannot upload artifact: path does not exist: %s",
                local_path,
            )
            return None

        try:
            # Save the artifact to ZenML using PathMaterializer
            # (handles both files and directories automatically)
            artifact_version = save_artifact(
                data=local_path,
                name=artifact_name,
                materializer=PathMaterializer,
                tags=[
                    "neptune-import",
                    f"neptune-attribute:{neptune_attr_path}",
                ],
                user_metadata={
                    "neptune_source": "neptune-exporter",
                    "neptune_attribute_path": neptune_attr_path,
                    "original_path": str(local_path),
                },
            )

            artifact_version_id = str(artifact_version.id)
            self._logger.debug(
                "Saved artifact '%s' (id=%s) for Neptune attribute '%s'",
                artifact_name,
                artifact_version_id,
                neptune_attr_path,
            )

            # Link the artifact to the model version using the public API
            model_version = self._get_model_version_response(model_version_id)
            if model_version is not None:
                try:
                    link_artifact_version_to_model_version(
                        artifact_version=artifact_version,
                        model_version=model_version,
                    )
                    self._logger.debug(
                        "Linked artifact '%s' to model version '%s'",
                        artifact_name,
                        model_version_id,
                    )
                except Exception:
                    self._logger.warning(
                        "Failed to link artifact '%s' to model version '%s'; "
                        "artifact was saved but not linked",
                        artifact_name,
                        model_version_id,
                        exc_info=True,
                    )
            else:
                self._logger.warning(
                    "Could not fetch model version '%s'; artifact '%s' saved but not linked",
                    model_version_id,
                    artifact_name,
                )

            return artifact_version_id

        except Exception:
            self._logger.error(
                "Failed to upload artifact '%s' from '%s'",
                artifact_name,
                local_path,
                exc_info=True,
            )
            return None

    def _upload_files_as_artifacts(
        self,
        file_refs: Dict[str, list[str]],
        files_directory: AnyPath,
        model_version_id: str,
        all_metadata: Dict[str, Any],
    ) -> None:
        """Upload collected file references as ZenML artifacts and link to model version.

        This method iterates through all collected file references, uploads them
        as proper ZenML artifacts using `save_artifact()`, links them to the model
        version, and stores artifact references in metadata.

        Args:
            file_refs: Dictionary mapping Neptune attribute paths to relative file paths.
            files_directory: Base directory where exported files are located.
            model_version_id: The ZenML Model Version ID to link artifacts to.
            all_metadata: Metadata dictionary to update with artifact references.
        """
        if save_artifact is None:
            self._logger.info(
                "ZenML save_artifact not available; files will be stored as metadata only"
            )
            # Fall back to storing paths as metadata
            self._merge_file_refs_into_metadata(file_refs, all_metadata)
            return

        uploaded_artifacts: Dict[str, str] = {}

        for meta_key, rel_paths in file_refs.items():
            if not rel_paths:
                continue

            # meta_key is like "files/artifacts/model"
            # Extract the Neptune attribute path (remove "files/" prefix)
            neptune_attr_path = (
                meta_key[6:] if meta_key.startswith("files/") else meta_key
            )

            for rel_path in rel_paths:
                abs_path = files_directory / rel_path

                if not abs_path.exists():
                    self._logger.debug(
                        "Skipping artifact upload for non-existent path: %s",
                        abs_path,
                    )
                    continue

                # Create a sanitized artifact name from the Neptune attribute path
                artifact_name = self._sanitize_name(
                    f"neptune-{neptune_attr_path.replace('/', '-')}",
                    max_len=100,
                )

                artifact_id = self._upload_artifact_to_zenml(
                    local_path=abs_path,
                    artifact_name=artifact_name,
                    model_version_id=model_version_id,
                    neptune_attr_path=neptune_attr_path,
                )

                if artifact_id:
                    uploaded_artifacts[neptune_attr_path] = artifact_id

        # Store artifact references in metadata
        if uploaded_artifacts:
            for attr_path, artifact_id in uploaded_artifacts.items():
                # Store under "artifacts/{original_path}/zenml_artifact_id"
                self._set_nested_value(
                    all_metadata,
                    f"artifacts/{attr_path}/zenml_artifact_id",
                    artifact_id,
                )
            self._logger.info(
                "Uploaded %d artifacts to ZenML for model version %s",
                len(uploaded_artifacts),
                model_version_id,
            )

        # Also store local paths as fallback metadata (for reference)
        self._merge_file_refs_into_metadata(file_refs, all_metadata)

    def _update_model_version_description_and_tags(
        self,
        run_id: TargetRunId,
        description: Optional[str],
        tags: Set[str],
        experiment_tags: Set[str],
    ) -> None:
        """Best-effort update of ZenML model version description and tags.

        ZenML APIs may evolve over time, so this method is intentionally
        defensive: failures are logged but do not stop the load.
        """
        if not description and not tags and not experiment_tags:
            return

        try:
            get_mv = getattr(self._client, "get_model_version", None)
            update_mv = getattr(self._client, "update_model_version", None)
            if get_mv is None or update_mv is None:
                # Older or incompatible ZenML client; metadata is still logged,
                # so silently skip the description/tag update.
                self._logger.debug(
                    "ZenML client does not expose get_model_version/update_model_version; "
                    "skipping description/tag update for %s",
                    run_id,
                )
                return

            # Fetch current model version to merge with existing tags/description
            mv = get_mv(str(run_id))  # type: ignore[misc]
            existing_tags = set(getattr(mv, "tags", []) or [])
            merged_tags: Set[str] = set(existing_tags)

            # Add plain Neptune tags
            merged_tags.update(tags)

            # Optionally encode experiment/group tags as separate tag namespace
            merged_tags.update({f"neptune-experiment:{t}" for t in experiment_tags})

            new_description = description or getattr(mv, "description", None)

            update_mv(  # type: ignore[misc]
                str(run_id),
                description=new_description,
                tags=sorted(merged_tags) if merged_tags else None,
            )
        except Exception:
            # This update is a convenience; failures should not break the load.
            self._logger.debug(
                "Failed to update ZenML model version %s description/tags; "
                "metadata has still been logged.",
                run_id,
                exc_info=True,
            )

    def _log_metadata_to_zenml(
        self,
        run_id: TargetRunId,
        metadata: Dict[str, Any],
    ) -> None:
        """Log aggregated metadata to the ZenML Model Version."""
        from uuid import UUID as PyUUID

        if not metadata:
            return

        if log_metadata is None:
            # This should not happen because we guard instantiation in __init__,
            # but we keep the check defensive for safety.
            raise RuntimeError(
                "ZenML log_metadata is not available, cannot log metadata."
            )

        try:
            # Convert string ID to UUID as required by ZenML's log_metadata API
            model_version_uuid = PyUUID(str(run_id))
            log_metadata(
                metadata=metadata,
                model_version_id=model_version_uuid,
            )
            self._logger.info("Uploaded metadata for ZenML model version %s", run_id)
        except Exception:
            self._logger.error(
                "Failed to log metadata for ZenML model version %s",
                run_id,
                exc_info=True,
            )
            raise

    # -------------------------------------------------------------------------
    # DataLoader interface
    # -------------------------------------------------------------------------

    def create_experiment(
        self,
        project_id: ProjectId,
        experiment_name: str,  # noqa: ARG002 - kept for interface compatibility
    ) -> TargetExperimentId:
        """Create or get a ZenML Model for a Neptune project.

        Neptune's experiment concept is not modeled directly in ZenML for this
        integration; all runs from a single Neptune project are grouped under
        a single ZenML Model, and experiment-level information is preserved as
        metadata and tags instead.
        """
        return self._get_or_create_model(project_id)

    def find_run(
        self,
        project_id: ProjectId,  # noqa: ARG002 - kept for interface compatibility
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find an existing ZenML Model Version by name under a given Model.

        Args:
            project_id: Neptune project ID (unused, kept for interface symmetry).
            run_name: Neptune run name (used as model version name).
            experiment_id: ZenML Model ID (TargetExperimentId).

        Returns:
            TargetRunId for the existing model version, or None if not found or
            if the client does not support the required query operations.
        """
        if experiment_id is None:
            return None

        try:
            versions_page = self._client.list_model_versions(
                model_name_or_id=str(experiment_id),
                name=run_name,
            )
            versions = list(versions_page)
        except TypeError:
            # Older ZenML versions may not support name filtering; fall back to
            # manual filtering if listing all versions is allowed.
            try:
                versions_page = self._client.list_model_versions(
                    model_name_or_id=str(experiment_id)
                )
            except Exception:
                self._logger.warning(
                    "Error listing ZenML model versions for experiment '%s'",
                    experiment_id,
                    exc_info=True,
                )
                return None

            versions = [
                v for v in versions_page if getattr(v, "name", None) == run_name
            ]
        except Exception:
            self._logger.warning(
                "Error listing ZenML model versions for experiment '%s'",
                experiment_id,
                exc_info=True,
            )
            return None

        if not versions:
            return None

        mv = versions[0]
        mv_id = getattr(mv, "id", None)
        if mv_id is None:
            return None

        return TargetRunId(str(mv_id))

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,  # noqa: ARG002 - not modeled natively
        fork_step: Optional[float] = None,  # noqa: ARG002 - not modeled natively
        step_multiplier: Optional[int] = None,  # noqa: ARG002 - not modeled natively
    ) -> TargetRunId:
        """Create a ZenML Model Version representing a Neptune run.

        Args:
            project_id: Neptune project ID.
            run_name: Neptune run name (used as model version name).
            experiment_id: ZenML Model ID (TargetExperimentId).
            parent_run_id: Not modeled directly in ZenML; preserved as metadata.
            fork_step: Not modeled directly; preserved as metadata.
            step_multiplier: Not used by ZenML for parent relationships.

        Returns:
            TargetRunId corresponding to the ZenML Model Version.
        """
        if experiment_id is None:
            # Ensure the model exists even if caller forgot to pass experiment_id
            experiment_id = self._get_or_create_model(project_id)

        version_name = self._sanitize_name(run_name or "neptune-run")

        try:
            mv = self._client.create_model_version(
                model_name_or_id=str(experiment_id),
                name=version_name,
                description="Imported from Neptune",
                tags=[
                    "neptune-import",
                    f"neptune-project:{project_id}",
                ],
            )
            mv_id = getattr(mv, "id", None)
            if mv_id is None:
                raise RuntimeError(
                    "ZenML returned a model version without an 'id' attribute."
                )

            target_run_id = TargetRunId(str(mv_id))

            # Store mapping for later artifact linking (get_model_version needs both IDs)
            self._mv_to_model_cache[str(mv_id)] = str(experiment_id)

            self._logger.info(
                "Created ZenML model version '%s' (id=%s) for Neptune run '%s'",
                getattr(mv, "name", version_name),
                target_run_id,
                run_name,
            )
            return target_run_id
        except Exception:
            self._logger.error(
                "Error creating ZenML model version for Neptune project '%s', run '%s'",
                project_id,
                run_name,
                exc_info=True,
            )
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: AnyPath,
        step_multiplier: int,  # noqa: ARG002 - not required for summary-only stats
    ) -> None:
        """Upload all data for a single run to a ZenML Model Version.

        The loader:
        - Logs scalar parameters, metrics, and system attributes as metadata.
        - Aggregates float_series into summary statistics (min/max/final/count).
        - Uploads file/artifact references as proper ZenML artifacts using
          save_artifact() and links them to the model version.
        - Best-effort updates model version description and tags based on
          Neptune's sys/ attributes.
        """
        all_metadata: Dict[str, Any] = {}
        series_stats: SeriesStatsByAttr = {}
        file_refs: Dict[str, list[str]] = {}
        tags: Set[str] = set()
        experiment_tags: Set[str] = set()
        sys_info: Dict[str, Any] = {}

        # Basic import provenance metadata (nested under "neptune_import" card)
        self._set_nested_value(all_metadata, "neptune_import/source", "neptune")
        self._set_nested_value(all_metadata, "neptune_import/tool", "neptune-exporter")
        self._set_nested_value(all_metadata, "neptune_import/loader", "zenml")
        self._set_nested_value(
            all_metadata, "neptune_import/target_run_id", str(run_id)
        )

        try:
            for part in run_data:
                df = part.to_pandas()

                self._accumulate_scalar_metadata(
                    df=df,
                    all_metadata=all_metadata,
                    tags=tags,
                    experiment_tags=experiment_tags,
                    sys_info=sys_info,
                )
                self._accumulate_float_series_stats(df=df, series_stats=series_stats)
                self._accumulate_file_references(
                    df=df,
                    file_refs=file_refs,
                    files_directory=files_directory,
                )

            # Finalize aggregated data into metadata dict
            self._finalize_series_metadata(
                series_stats=series_stats, all_metadata=all_metadata
            )

            # Upload files as ZenML artifacts and link to model version
            # This replaces the old _merge_file_refs_into_metadata call
            self._upload_files_as_artifacts(
                file_refs=file_refs,
                files_directory=files_directory,
                model_version_id=str(run_id),
                all_metadata=all_metadata,
            )

            # Preserve Neptune tags and experiment tags as metadata as well
            # These go under the "sys" card alongside other system attributes
            if tags:
                self._set_nested_value(all_metadata, "sys/tags", sorted(tags))
            if experiment_tags:
                self._set_nested_value(
                    all_metadata, "sys/group_tags", sorted(experiment_tags)
                )

            # Attempt to update model version description and tags from sys/ fields
            description = sys_info.get("sys/description")
            self._update_model_version_description_and_tags(
                run_id=run_id,
                description=description,
                tags=tags,
                experiment_tags=experiment_tags,
            )

            # Finally, log all metadata to the ZenML Model Version
            self._log_metadata_to_zenml(run_id=run_id, metadata=all_metadata)
        except Exception:
            self._logger.error(
                "Error uploading data for ZenML model version %s", run_id, exc_info=True
            )
            raise


__all__ = ["ZenMLLoader", "ZENML_AVAILABLE"]
