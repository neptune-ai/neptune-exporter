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
from typing import Any
import logging
import pyarrow.compute as pc

from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.storage.types import AnyPath


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

    def get_project_summary(self, project_directory: AnyPath) -> dict[str, Any] | None:
        """
        Get detailed summary for a specific project.

        Args:
            project_directory: The project directory to analyze.

        Returns:
            Dictionary with detailed project information including statistics.
        """
        try:
            project_id: str | None = None
            total_files = 0
            total_records = 0
            total_size_bytes = 0
            records_per_file: list[int] = []

            unique_runs: set[Any] = set()
            unique_attribute_types: set[Any] = set()
            attribute_paths_by_type: dict[Any, set[Any]] = {}
            run_breakdown: dict[Any, int] = {}

            step_stats: dict[str, Any] = {
                "total_steps": 0,
                "min_step": None,
                "max_step": None,
                "unique_steps": 0,
            }
            unique_steps: set[Any] = set()

            for table in self._parquet_reader.read_project_data(project_directory):
                record_count = len(table)
                if record_count == 0:
                    continue

                total_files += 1
                total_records += record_count
                total_size_bytes += table.nbytes
                records_per_file.append(record_count)

                if project_id is None:
                    project_id = table["project_id"][0].as_py()

                run_counts = pc.value_counts(table["run_id"]).to_pylist()
                for run_count in run_counts:
                    run_id = run_count["values"]
                    count = run_count["counts"]
                    unique_runs.add(run_id)
                    run_breakdown[run_id] = run_breakdown.get(run_id, 0) + count

                table_attribute_types = pc.unique(table["attribute_type"]).to_pylist()
                for attr_type in table_attribute_types:
                    unique_attribute_types.add(attr_type)
                    type_mask = pc.equal(table["attribute_type"], attr_type)
                    unique_paths = pc.unique(
                        pc.filter(table["attribute_path"], type_mask)
                    ).to_pylist()
                    if attr_type not in attribute_paths_by_type:
                        attribute_paths_by_type[attr_type] = set()
                    attribute_paths_by_type[attr_type].update(unique_paths)

                # Update step statistics incrementally to avoid loading all tables.
                if "step" in table.column_names:
                    non_null_mask = pc.true_unless_null(table["step"])
                    non_null_steps = pc.filter(table["step"], non_null_mask)
                    if len(non_null_steps) > 0:
                        step_stats["total_steps"] += len(non_null_steps)
                        unique_steps.update(pc.unique(non_null_steps).to_pylist())
                        table_min = pc.min(non_null_steps).as_py()
                        table_max = pc.max(non_null_steps).as_py()
                        if (
                            step_stats["min_step"] is None
                            or table_min < step_stats["min_step"]
                        ):
                            step_stats["min_step"] = table_min
                        if (
                            step_stats["max_step"] is None
                            or table_max > step_stats["max_step"]
                        ):
                            step_stats["max_step"] = table_max

            if total_files == 0:
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

            step_stats["unique_steps"] = len(unique_steps)
            attribute_breakdown = {
                attr_type: len(paths)
                for attr_type, paths in attribute_paths_by_type.items()
            }
            file_info = {
                "total_files": total_files,
                "total_size_bytes": total_size_bytes,
                "records_per_file": records_per_file,
            }

            return {
                "project_id": project_id,
                "total_runs": len(unique_runs),
                "attribute_types": sorted(unique_attribute_types, key=lambda x: str(x)),
                "runs": sorted(unique_runs, key=lambda x: str(x)),
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
