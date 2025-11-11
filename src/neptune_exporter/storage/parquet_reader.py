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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Generator, Optional
import logging

from neptune_exporter import model
from neptune_exporter.utils import sanitize_path_part


class ParquetReader:
    """Reads exported Neptune data from parquet files."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._logger = logging.getLogger(__name__)

    def list_project_directories(self) -> list[Path]:
        """List all available projects in the exported data."""
        if not self.base_path.exists():
            return []

        project_directories = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                project_directories.append(item)
        return sorted(project_directories)

    def read_project_data(
        self,
        project_directory: Path,
        project_ids: Optional[list[str]] = None,
        runs: Optional[list[str]] = None,
        attribute_types: Optional[list[str]] = None,
    ) -> Generator[pa.Table, None, None]:
        """Read all data for a project, optionally filtered by project ids, runs and attribute types.

        Only reads complete parquet files (excludes .tmp files).
        """
        parquet_files = self._list_parquet_files_in_project(project_directory)

        if not parquet_files:
            self._logger.warning(f"No parquet files found in {project_directory}")
            return

        for file_path in parquet_files:
            try:
                # Read parquet file
                table = pq.read_table(file_path, schema=model.SCHEMA)

                # Apply filters using PyArrow compute functions
                if project_ids is not None:
                    mask = pc.is_in(table["project_id"], pa.array(project_ids))
                    table = table.filter(mask)

                if runs is not None:
                    mask = pc.is_in(table["run_id"], pa.array(runs))
                    table = table.filter(mask)

                if attribute_types is not None:
                    mask = pc.is_in(table["attribute_type"], pa.array(attribute_types))
                    table = table.filter(mask)

                if len(table) > 0:
                    yield table

            except Exception as e:
                self._logger.error(f"Error reading parquet file {file_path}: {e}")
                continue

    def check_run_exists(self, project_id: str, run_id: str) -> bool:
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

    def _list_parquet_files_in_project(self, project_directory: Path) -> list[Path]:
        """List all parquet files for a given project.

        Matches pattern run_id_part_N.parquet. The glob pattern automatically
        excludes .tmp files since they end with .parquet.tmp, not .parquet.
        """
        if not project_directory.exists():
            return []

        parquet_files = []
        # Match pattern: *_part_*.parquet (run_id_part_N.parquet)
        # This automatically excludes .tmp files (which end with .parquet.tmp)
        for file_path in project_directory.glob("*_part_*.parquet"):
            parquet_files.append(file_path)

        return sorted(parquet_files)
