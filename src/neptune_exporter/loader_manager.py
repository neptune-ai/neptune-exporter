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
from typing import Optional, Set
from tqdm import tqdm
import logging

from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.loaders.mlflow import MLflowLoader


class LoaderManager:
    """Manages the loading of Neptune data from parquet files to target platforms."""

    def __init__(
        self,
        parquet_reader: ParquetReader,
        mlflow_loader: MLflowLoader,
        files_base_path: Path,
    ):
        self._parquet_reader = parquet_reader
        self._mlflow_loader = mlflow_loader
        self._files_base_path = files_base_path
        self._logger = logging.getLogger(__name__)

    def load_to_mlflow(
        self,
        project_ids: Optional[list[str]] = None,
        runs: Optional[Set[str]] = None,
        attribute_types: Optional[Set[str]] = None,
    ) -> None:
        """
        Load Neptune data from parquet files to MLflow.

        Args:
            project_ids: List of project IDs to load. If None, loads all available projects.
            runs: Set of run IDs to filter by. If None, loads all runs.
            attribute_types: Set of attribute types to load. If None, loads all types.
        """
        # Get projects to process
        project_directories = self._parquet_reader.list_project_directories()

        if not project_directories:
            self._logger.warning("No projects found to load in the input path")
            return

        self._logger.info(
            f"Starting MLflow loading for {len(project_directories)} project(s)"
        )

        # Process each project
        for project_directory in tqdm(
            project_directories, desc="Loading projects to MLflow", unit="project"
        ):
            try:
                self._load_project(project_directory, runs, attribute_types)
            except Exception as e:
                self._logger.error(f"Error loading project {project_directory}: {e}")
                continue

        self._logger.info("MLflow loading completed")

    def _load_project(
        self,
        project_directory: Path,
        runs: Optional[Set[str]] = None,
        attribute_types: Optional[Set[str]] = None,
    ) -> None:
        """Load a single project to MLflow."""
        self._logger.info(f"Loading project {project_directory} to MLflow")

        # Create MLflow experiment
        project_id = project_directory.name  # TODO: wrong
        experiment_id = self._mlflow_loader.create_experiment(project_id)

        # Get runs to process
        # if runs is None:
        #     runs = self._parquet_reader.get_unique_runs(project_id)

        if not runs:
            self._logger.warning(f"No runs found for project {project_id}")
            return

        # Process each run
        for run_id in tqdm(
            runs, desc=f"Loading runs from {project_id}", unit="run", leave=False
        ):
            try:
                self._load_run(project_id, run_id, experiment_id, attribute_types)
            except Exception as e:
                self._logger.error(
                    f"Error loading run {run_id} from project {project_id}: {e}"
                )
                continue

    def _load_run(
        self,
        project_id: str,
        run_id: str,
        experiment_id: str,
        attribute_types: Optional[Set[str]] = None,
    ) -> None:
        """Load a single run to MLflow."""
        # Read run data from parquet
        # run_data = self._parquet_reader.read_run_data(project_id, run_id, attribute_types)

        # if len(run_data) == 0:
        #     self._logger.warning(f"No data found for run {run_id} in project {project_id}")
        #     return

        # # Upload to MLflow
        # files_base_path = self._files_base_path / project_id
        # self._mlflow_loader.upload_run_data(
        #     run_data=run_data,
        #     run_id=run_id,
        #     experiment_id=experiment_id,
        #     files_base_path=files_base_path,
        # )
