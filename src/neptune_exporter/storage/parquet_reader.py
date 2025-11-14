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

import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Any, Generator, Optional
import logging
from dataclasses import dataclass
from neptune_exporter import model
from neptune_exporter.types import RunFilePrefix, SourceRunId
from neptune_exporter.utils import sanitize_path_part


@dataclass(frozen=True, order=True, slots=True)
class RunMetadata:
    """Metadata for a run."""

    project_id: str
    run_id: SourceRunId
    custom_run_id: Optional[str]
    experiment_name: Optional[str]
    parent_source_run_id: Optional[SourceRunId]
    fork_step: Optional[float]
    creation_time: Optional[datetime.datetime]


class ParquetReader:
    """Reads exported Neptune data from parquet files."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._logger = logging.getLogger(__name__)

    def check_run_exists(self, project_id: str, run_id: SourceRunId) -> bool:
        """Check if a run exists and is complete (has part_0.parquet).

        Args:
            project_id: The project ID
            run_id: The run ID

        Returns:
            True if the run exists and is complete (has part_0.parquet), False otherwise
        """
        sanitized_project_id = sanitize_path_part(project_id)
        sanitized_run_id = sanitize_path_part(run_id)

        project_directory = self.base_path / sanitized_project_id
        if not project_directory.exists():
            return False

        # Check if part_0 file exists for this run
        part_0_file = project_directory / f"{sanitized_run_id}_part_0.parquet"
        return part_0_file.exists()

    def list_project_directories(
        self, project_ids: Optional[list[str]] = None
    ) -> list[Path]:
        """List all available projects in the exported data."""
        if not self.base_path.exists():
            return []

        if project_ids is not None:
            project_directories = [
                self.base_path / sanitize_path_part(project_id)
                for project_id in project_ids
            ]
            project_directories = [
                item for item in project_directories if item.is_dir()
            ]
        else:
            project_directories = [
                item
                for item in self.base_path.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ]

        return sorted(project_directories)

    def list_run_files(
        self, project_directory: Path, run_ids: Optional[list[SourceRunId]] = None
    ) -> list[RunFilePrefix]:
        """List all complete runs (those with part_0) in a project.

        Reads the file name without the part_0 suffix and extension to get the run IDs.

        Args:
            project_directory: The project directory to scan

        Returns:
            List of actual run IDs that have part_0.parquet (complete runs)
        """
        if not project_directory.exists():
            return []

        if run_ids is not None:
            run_files = [
                project_directory / f"{sanitize_path_part(run_id)}_part_0.parquet"
                for run_id in run_ids
            ]
            run_files = [file_path for file_path in run_files if file_path.exists()]
        else:
            run_files = [
                file_path for file_path in project_directory.glob("*_part_0.parquet")
            ]

        run_file_prefixes = [
            RunFilePrefix(file_path.stem.replace("_part_0", ""))
            for file_path in run_files
        ]

        return sorted(run_file_prefixes)

    def read_project_data(
        self,
        project_directory: Path,
        runs: Optional[list[SourceRunId]] = None,
        attribute_types: Optional[list[str]] = None,
    ) -> Generator[pa.Table, None, None]:
        """Read all data for a project, optionally filtered by project ids, runs and attribute types.

        Reads run-by-run, yielding parts separately (max 1 part in memory at once).
        Only reads complete runs (those with part_0.parquet).
        """
        all_run_file_prefixes = self.list_run_files(project_directory, runs)

        if not all_run_file_prefixes:
            self._logger.warning(f"No runs found in {project_directory}")
            return

        for run_file_prefix in all_run_file_prefixes:
            for part_table in self.read_run_data(
                project_directory, run_file_prefix, attribute_types
            ):
                yield part_table

    def read_run_data(
        self,
        project_directory: Path,
        run_file_prefix: RunFilePrefix,
        attribute_types: Optional[list[str]] = None,
        attribute_paths: Optional[list[str]] = None,
    ) -> Generator[pa.Table, None, None]:
        """Read all parts for a specific run sequentially, yielding one part at a time.

        Args:
            project_id: The project ID
            run_id: The run ID

        Yields:
            PyArrow Table for each part (part_0, part_1, part_2, ...)
        """
        run_part_files = self._get_run_files(project_directory, run_file_prefix)

        if not run_part_files:
            return

        for part_file in run_part_files:
            try:
                table = pq.read_table(part_file, schema=model.SCHEMA)
                table = self._filter_attributes(
                    table,
                    attribute_types=attribute_types,
                    attribute_paths=attribute_paths,
                )
                if len(table) > 0:
                    yield table
            except Exception as e:
                self._logger.error(
                    f"Error reading part file {part_file} for run {run_file_prefix}: {e}"
                )
                continue

    def _get_run_files(
        self, project_directory: Path, run_file_prefix: RunFilePrefix
    ) -> list[Path]:
        """Get sorted list of all part files for a specific run.

        Args:
            project_directory: The project directory
            run_id: The run ID (already sanitized with digest)

        Returns:
            Sorted list of part file paths (part_0, part_1, part_2, ...)
        """
        if not project_directory.exists():
            return []

        pattern = f"{run_file_prefix}_part_*.parquet"

        part_files = []
        for file_path in project_directory.glob(pattern):
            part_files.append(file_path)

        # Sort by part number
        def get_part_number(path: Path) -> int:
            stem = path.stem  # run_id_part_N
            parts = stem.rsplit("_part_", 1)
            if len(parts) == 2:
                return int(parts[1])
            return 0

        return sorted(part_files, key=get_part_number)

    def _filter_attributes(
        self,
        table: pa.Table,
        attribute_types: Optional[list[str]] = None,
        attribute_paths: Optional[list[str]] = None,
    ) -> pa.Table:
        """Filter a table by attribute types."""

        if attribute_types is not None:
            mask = pc.is_in(table["attribute_type"], pa.array(attribute_types))
            table = table.filter(mask)

        if attribute_paths is not None:
            mask = pc.is_in(table["attribute_path"], pa.array(attribute_paths))
            table = table.filter(mask)

        return table

    def read_run_metadata(
        self, project_directory: Path, run_file_prefix: RunFilePrefix
    ) -> Optional[RunMetadata]:
        """Read metadata from a run by reading all parts sequentially.

        Metadata fields: project_id, run_id, sys/custom_run_id, sys/name,
        sys/forking/parent, sys/forking/step.

        Usually metadata is in part_0, but reads all parts if needed for robustness.

        Args:
            project_directory: The project directory
            run_file_prefix: The run file prefix

        Returns:
            RunMetadata object, or None if run doesn't exist
        """
        metadata: dict[str, Any] = {
            "project_id": None,
            "run_id": None,
            "custom_run_id": None,
            "experiment_name": None,
            "parent_source_run_id": None,
            "fork_step": None,
            "creation_time": None,
        }

        # Read all parts to find metadata (usually in part_0, but read all for robustness)
        for part_table in self.read_run_data(
            project_directory,
            run_file_prefix,
            attribute_paths=[
                "sys/custom_run_id",
                "sys/name",
                "sys/forking/parent",
                "sys/forking/step",
                "sys/creation_time",
            ],
        ):
            # Extract metadata fields
            if metadata["project_id"] is None:
                project_ids = pc.unique(part_table["project_id"]).to_pylist()
                if project_ids:
                    metadata["project_id"] = project_ids[0]

            if metadata["run_id"] is None:
                run_ids = pc.unique(part_table["run_id"]).to_pylist()
                if run_ids:
                    metadata["run_id"] = run_ids[0]

            # Extract attribute-based metadata
            if metadata["custom_run_id"] is None:
                custom_run_id = self._get_attribute_value(
                    part_table, "sys/custom_run_id"
                )
                if custom_run_id:
                    metadata["custom_run_id"] = custom_run_id

            if metadata["experiment_name"] is None:
                experiment_name = self._get_attribute_value(part_table, "sys/name")
                if experiment_name:
                    metadata["experiment_name"] = experiment_name

            if metadata["parent_source_run_id"] is None:
                parent = self._get_attribute_value(part_table, "sys/forking/parent")
                if parent:
                    metadata["parent_source_run_id"] = parent

            if metadata["fork_step"] is None:
                fork_step_str = self._get_attribute_value(
                    part_table, "sys/forking/step", attribute_type="float_value"
                )
                if fork_step_str:
                    try:
                        metadata["fork_step"] = float(fork_step_str)
                    except (ValueError, TypeError):
                        pass

            if metadata["creation_time"] is None:
                creation_time_str = self._get_attribute_value(
                    part_table, "sys/creation_time", attribute_type="datetime_value"
                )
                if creation_time_str:
                    try:
                        metadata["creation_time"] = datetime.datetime.fromisoformat(
                            creation_time_str
                        )
                    except (ValueError, TypeError):
                        pass

            # If we have all 'important' metadata, we can stop early (optimization)
            if all(
                v is not None
                for k, v in metadata.items()
                if k in ["run_id", "project_id", "custom_run_id"]
            ):
                return RunMetadata(
                    project_id=metadata["project_id"],
                    run_id=SourceRunId(metadata["run_id"]),
                    custom_run_id=metadata["custom_run_id"],
                    experiment_name=metadata["experiment_name"],
                    parent_source_run_id=SourceRunId(metadata["parent_source_run_id"]),
                    fork_step=metadata["fork_step"],
                    creation_time=metadata["creation_time"],
                )

        if metadata["project_id"] and metadata["run_id"]:
            return RunMetadata(
                project_id=metadata["project_id"],
                run_id=SourceRunId(metadata["run_id"]),
                custom_run_id=metadata["custom_run_id"],
                experiment_name=metadata["experiment_name"],
                parent_source_run_id=SourceRunId(metadata["parent_source_run_id"]),
                fork_step=metadata["fork_step"],
                creation_time=metadata["creation_time"],
            )
        else:
            return None

    @staticmethod
    def _get_attribute_value(
        table: pa.Table, attribute_path: str, attribute_type: str = "string_value"
    ) -> Optional[str]:
        """Extract attribute value from a table.

        Args:
            table: PyArrow table
            attribute_path: The attribute path to look for
            attribute_type: The attribute type column to read from

        Returns:
            The attribute value as string, or None if not found
        """
        mask = pc.equal(table["attribute_path"], attribute_path)
        if pc.sum(mask).as_py() > 0:
            filtered = table.filter(mask)
            if len(filtered) > 0:
                value = filtered[attribute_type].take([0]).to_pylist()[0]
                if value is not None:
                    return str(value)
        return None
