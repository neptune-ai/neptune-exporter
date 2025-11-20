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
import pyarrow as pa
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
            Dictionary with detailed project information including statistics.
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
                    "total_records": 0,
                    "attribute_breakdown": {},
                    "run_breakdown": {},
                    "file_info": {},
                }

            # Get project_id from the first table
            project_id = all_tables[0]["project_id"][0].as_py()

            # Combine all tables for analysis
            combined_table = pa.concat_tables(all_tables)
            total_records = len(combined_table)

            # Get unique runs and attribute types
            unique_runs = pc.unique(combined_table["run_id"]).to_pylist()
            unique_attribute_types = pc.unique(
                combined_table["attribute_type"]
            ).to_pylist()

            # Calculate attribute breakdown (count of unique attribute paths per type)
            attribute_breakdown = {}
            for attr_type in unique_attribute_types:
                type_mask = pc.equal(combined_table["attribute_type"], attr_type)
                filtered_table = combined_table.filter(type_mask)
                unique_paths = pc.unique(filtered_table["attribute_path"])
                attribute_breakdown[attr_type] = len(unique_paths)

            # Calculate run breakdown (record count per run)
            run_breakdown = {}
            for run_id in unique_runs:
                run_mask = pc.equal(combined_table["run_id"], run_id)
                run_table = combined_table.filter(run_mask)
                run_breakdown[run_id] = len(run_table)

            # Calculate file information
            file_info = {
                "total_files": len(all_tables),
                "total_size_bytes": sum(table.nbytes for table in all_tables),
                "records_per_file": [len(table) for table in all_tables],
            }

            # Calculate step statistics for numeric steps
            step_stats = self._calculate_step_statistics(combined_table)

            return {
                "project_id": project_id,
                "total_runs": len(unique_runs),
                "attribute_types": sorted(unique_attribute_types),
                "runs": sorted(unique_runs),
                "total_records": total_records,
                "attribute_breakdown": attribute_breakdown,
                "run_breakdown": run_breakdown,
                "file_info": file_info,
                "step_statistics": step_stats,
            }
        except Exception:
            self._logger.error(
                f"Error analyzing project {project_directory}", exc_info=True
            )
            return None

    def _calculate_step_statistics(self, table: pa.Table) -> dict[str, Any]:
        """Calculate statistics for step values in the table."""
        try:
            # Filter out null steps
            non_null_mask = pc.true_unless_null(table["step"])
            non_null_table = table.filter(non_null_mask)

            if len(non_null_table) == 0:
                return {
                    "total_steps": 0,
                    "min_step": None,
                    "max_step": None,
                    "unique_steps": 0,
                }

            # Get step values as Python floats
            step_values = non_null_table["step"].to_pylist()

            return {
                "total_steps": len(step_values),
                "min_step": min(step_values),
                "max_step": max(step_values),
                "unique_steps": len(set(step_values)),
            }
        except Exception:
            self._logger.error("Could not calculate step statistics", exc_info=True)
            return {
                "total_steps": 0,
                "min_step": None,
                "max_step": None,
                "unique_steps": 0,
            }
