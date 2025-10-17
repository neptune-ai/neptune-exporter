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
from typing import Generator, Optional, Set
import logging

from neptune_exporter import model


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
        runs: Optional[Set[str]] = None,
        attribute_types: Optional[Set[str]] = None,
    ) -> Generator[pa.Table, None, None]:
        """Read all data for a project, optionally filtered by runs and attribute types."""
        parquet_files = self._list_parquet_files_in_project(project_directory)

        if not parquet_files:
            self._logger.warning(f"No parquet files found in {project_directory}")
            return

        for file_path in parquet_files:
            try:
                # Read parquet file
                table = pq.read_table(file_path, schema=model.SCHEMA)

                # Apply filters using PyArrow compute functions
                if runs is not None:
                    mask = pc.is_in(table["run_id"], pa.array(list(runs)))
                    table = table.filter(mask)

                if attribute_types is not None:
                    mask = pc.is_in(
                        table["attribute_type"], pa.array(list(attribute_types))
                    )
                    table = table.filter(mask)

                if len(table) > 0:
                    yield table

            except Exception as e:
                self._logger.error(f"Error reading parquet file {file_path}: {e}")
                continue

    def _list_parquet_files_in_project(self, project_directory: Path) -> list[Path]:
        """List all parquet files for a given project."""
        if not project_directory.exists():
            return []

        parquet_files = []
        for file_path in project_directory.glob("part_*.parquet"):
            parquet_files.append(file_path)

        return sorted(parquet_files)
