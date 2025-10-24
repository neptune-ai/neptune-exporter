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
import pyarrow.compute as pc
from pathlib import Path
from typing import Generator, Optional
from tqdm import tqdm
import logging

from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.loaders.mlflow import MLflowLoader
from neptune_exporter.utils import sanitize_path_part


class LoaderManager:
    """Manages the loading of Neptune data from parquet files to target platforms."""

    def __init__(
        self,
        parquet_reader: ParquetReader,
        data_loader: MLflowLoader,
        files_directory: Path,
    ):
        self._parquet_reader = parquet_reader
        self._data_loader = data_loader
        self._files_directory = files_directory
        self._logger = logging.getLogger(__name__)

    def load(
        self,
        project_ids: Optional[list[str]] = None,
        runs: Optional[list[str]] = None,
    ) -> None:
        """
        Load Neptune data from files to target platforms (e.g. MLflow).

        Args:
            project_ids: List of project IDs to load. If None, loads all available projects.
            runs: Set of run IDs to filter by. If None, loads all runs.
        """
        # Get projects to process
        project_directories = self._parquet_reader.list_project_directories()

        if not project_directories:
            self._logger.warning("No projects found to load in the input path")
            return

        self._logger.info(
            f"Starting data loading for {len(project_directories)} project(s)"
        )

        # Process each project
        for project_directory in tqdm(
            project_directories, desc="Loading projects", unit="project"
        ):
            try:
                self._load_project(project_directory, project_ids, runs)
            except Exception:
                self._logger.error(
                    f"Error loading project {project_directory}", exc_info=True
                )
                continue

        self._logger.info("MLflow loading completed")

    def _load_project(
        self,
        project_directory: Path,
        project_ids: Optional[list[str]] = None,
        runs: Optional[list[str]] = None,
    ) -> None:
        """Load a single project to target platform."""
        self._logger.info(f"Loading data from {project_directory} to target platform")

        project_data_generator: Generator[pa.Table, None, None] = (
            self._parquet_reader.read_project_data(project_directory, project_ids, runs)
        )

        run_id_to_target_run_id: dict[str, str] = {}

        for project_data in project_data_generator:
            project_id = project_data["project_id"].to_pylist()[0]
            run_ids = pc.unique(project_data["run_id"]).to_pylist()

            for source_run_id in run_ids:
                try:
                    run_mask = pc.equal(project_data["run_id"], source_run_id)
                    run_data = project_data.filter(run_mask)

                    if source_run_id in run_id_to_target_run_id:
                        target_run_id = run_id_to_target_run_id[source_run_id]
                    else:
                        custom_run_id = (
                            self._get_attribute_value(run_data, "sys/custom_run_id")
                            or source_run_id
                        )
                        experiment_name = self._get_attribute_value(
                            run_data, "sys/experiment/name"
                        )
                        # parent_run_id = self._get_attribute_value(run_table, "sys/forking/parent")

                        if experiment_name is not None:
                            target_experiment_id = self._data_loader.create_experiment(
                                project_id=project_id, experiment_name=experiment_name
                            )
                        else:
                            target_experiment_id = None

                        target_run_id = self._data_loader.create_run(
                            project_id=project_id,
                            run_name=custom_run_id,
                            experiment_id=target_experiment_id,
                        )
                        run_id_to_target_run_id[source_run_id] = target_run_id

                    self._data_loader.upload_run_data(
                        run_data=run_data,
                        run_id=target_run_id,
                        files_directory=self._files_directory
                        / sanitize_path_part(project_id),
                    )
                except Exception:
                    self._logger.error(
                        f"Error loading project data for run {source_run_id}",
                        exc_info=True,
                    )
                    continue

    @staticmethod
    def _get_attribute_value(
        table: pa.Table, attribute_path: str, attribute_type: str = "string_value"
    ) -> Optional[str]:
        mask = pc.equal(table["attribute_path"], attribute_path)
        if pc.sum(mask).as_py() > 0:
            return table.filter(mask)[attribute_type].take([0]).to_pylist()[0]
        return None
