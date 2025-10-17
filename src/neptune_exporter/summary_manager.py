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

from pathlib import Path
from typing import Any
import logging
import pyarrow.compute as pc
from neptune_exporter.storage.parquet_reader import ParquetReader


class SummaryManager:
    """Manages analysis and reporting of exported Neptune data."""

    def __init__(self, parquet_reader: ParquetReader):
        self._parquet_reader = parquet_reader
        self._logger = logging.getLogger(__name__)

    def get_data_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of available data.

        Returns:
            Dictionary with detailed data summary including project counts, run counts, and attribute types.
        """
        project_directories = self._parquet_reader.list_project_directories()

        summary: dict[str, Any] = {
            "total_projects": len(project_directories),
            "projects": {},
        }

        for project_directory in project_directories:
            project_summary = self.get_project_summary(project_directory)
            summary["projects"][project_directory] = project_summary

        return summary

    def get_project_summary(self, project_directory: Path) -> dict[str, Any] | None:
        """
        Get detailed summary for a specific project.

        Args:
            project_directory: The project directory to analyze.

        Returns:
            Dictionary with detailed project information.
        """
        try:
            project_data_generator = self._parquet_reader.read_project_data(
                project_directory
            )
            all_tables = list(project_data_generator)

            if not all_tables:
                return {
                    "project_id": None,
                    "total_runs": 0,
                    "attribute_types": [],
                    "runs": [],
                }

            # Get project_id from the first table
            project_id = all_tables[0]["project_id"][0].as_py()

            # Combine all tables for analysis
            import pyarrow as pa

            combined_table = pa.concat_tables(all_tables)

            # Get unique runs and attribute types
            unique_runs = pc.unique(combined_table["run_id"]).to_pylist()
            unique_attribute_types = pc.unique(
                combined_table["attribute_type"]
            ).to_pylist()

            return {
                "project_id": project_id,
                "total_runs": len(unique_runs),
                "attribute_types": sorted(unique_attribute_types),
                "runs": sorted(unique_runs),
            }
        except Exception as e:
            self._logger.error(f"Error analyzing project {project_directory}: {e}")
            return None
