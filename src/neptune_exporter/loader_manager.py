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
from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.utils import sanitize_path_part


class LoaderManager:
    """Manages the loading of Neptune data from parquet files to target platforms."""

    def __init__(
        self,
        parquet_reader: ParquetReader,
        data_loader: DataLoader,
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
        Load Neptune data from files to target platforms.

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

        self._logger.info("Data loading completed")

    def _load_project(
        self,
        project_directory: Path,
        project_ids: Optional[list[str]] = None,
        runs: Optional[list[str]] = None,
    ) -> None:
        """Load a single project to target platform using streaming buffering approach.

        Runs are processed in topological order: parents before children.
        Runs waiting for their parent are buffered until parent becomes available.
        """
        self._logger.info(f"Loading data from {project_directory} to target platform")

        project_data_generator: Generator[pa.Table, None, None] = (
            self._parquet_reader.read_project_data(project_directory, project_ids, runs)
        )

        # Track processed runs
        processed_runs: set[str] = set()

        # Track parent-child relationships for efficient lookup
        parent_to_children: dict[str, list[str]] = {}

        # Track target run IDs
        run_id_to_target_run_id: dict[str, str] = {}

        # Buffer runs waiting for their parent (assumes each run's data is in a single table)
        # Structure: {run_id: {"data": pa.Table, "metadata": dict}}
        buffered_runs: dict[str, dict] = {}

        # Stream through parquet data
        for project_data in project_data_generator:
            project_id = project_data["project_id"].to_pylist()[0]
            run_ids = pc.unique(project_data["run_id"]).to_pylist()

            for source_run_id in run_ids:
                run_mask = pc.equal(project_data["run_id"], source_run_id)
                run_data = project_data.filter(run_mask)

                # Extract metadata
                custom_run_id = (
                    self._get_attribute_value(run_data, "sys/custom_run_id")
                    or source_run_id
                )
                experiment_name = self._get_attribute_value(
                    run_data, "sys/experiment/name"
                )
                parent_source_run_id = self._get_attribute_value(
                    run_data, "sys/forking/parent"
                )
                fork_step_str = self._get_attribute_value(run_data, "sys/forking/step")
                fork_step = float(fork_step_str) if fork_step_str is not None else None

                metadata = {
                    "project_id": project_id,
                    "custom_run_id": custom_run_id,
                    "experiment_name": experiment_name,
                    "parent_source_run_id": parent_source_run_id,
                    "fork_step": fork_step,
                }

                # Check if run can be processed immediately
                if (
                    parent_source_run_id is None
                    or parent_source_run_id in processed_runs
                ):
                    # Process immediately
                    self._process_run(
                        source_run_id=source_run_id,
                        run_data=run_data,
                        metadata=metadata,
                        processed_runs=processed_runs,
                        parent_to_children=parent_to_children,
                        run_id_to_target_run_id=run_id_to_target_run_id,
                        buffered_runs=buffered_runs,
                    )
                else:
                    # Buffer for later
                    buffered_runs[source_run_id] = {
                        "data": run_data,
                        "metadata": metadata,
                    }
                    # Track parent-child relationship
                    if parent_source_run_id not in parent_to_children:
                        parent_to_children[parent_source_run_id] = []
                    parent_to_children[parent_source_run_id].append(source_run_id)

        # Process any remaining buffered runs (orphaned - parent not in dataset)
        for source_run_id in list(buffered_runs.keys()):
            try:
                self._logger.warning(
                    f"Processing orphaned run {source_run_id} (parent not found in dataset)"
                )
                run_info = buffered_runs[source_run_id]
                self._process_run(
                    source_run_id=source_run_id,
                    run_data=run_info["data"],
                    metadata=run_info["metadata"],
                    processed_runs=processed_runs,
                    parent_to_children=parent_to_children,
                    run_id_to_target_run_id=run_id_to_target_run_id,
                    buffered_runs=buffered_runs,
                )
            except Exception:
                self._logger.error(
                    f"Error processing orphaned run {source_run_id}",
                    exc_info=True,
                )
                continue

    def _process_run(
        self,
        source_run_id: str,
        run_data: pa.Table,
        metadata: dict,
        processed_runs: set[str],
        parent_to_children: dict[str, list[str]],
        run_id_to_target_run_id: dict[str, str],
        buffered_runs: dict[str, dict],
    ) -> None:
        """Process a single run and recursively process its buffered children.

        Args:
            source_run_id: Source run ID from Neptune
            run_data: PyArrow table with run data
            metadata: Dictionary with run metadata (project_id, custom_run_id, etc.)
            processed_runs: Set of processed run IDs
            parent_to_children: Dictionary mapping parent to list of child IDs
            run_id_to_target_run_id: Dictionary mapping source run IDs to target run IDs
            buffered_runs: Dictionary of buffered runs waiting for parents
        """
        project_id = metadata["project_id"]
        custom_run_id = metadata["custom_run_id"]
        experiment_name = metadata["experiment_name"]
        parent_source_run_id = metadata["parent_source_run_id"]
        fork_step = metadata["fork_step"]

        # Get or create experiment
        if experiment_name is not None:
            target_experiment_id = self._data_loader.create_experiment(
                project_id=project_id, experiment_name=experiment_name
            )
        else:
            target_experiment_id = None

        # Get parent target run ID if parent exists
        parent_target_run_id = None
        if parent_source_run_id and parent_source_run_id in run_id_to_target_run_id:
            parent_target_run_id = run_id_to_target_run_id[parent_source_run_id]

        # Calculate step multiplier for W&B (needed for fork_step conversion)
        # MLflow returns None and calculates per-series, W&B returns global multiplier
        step_multiplier = None
        if fork_step is not None:
            step_multiplier = self._data_loader.calculate_global_step_multiplier(
                run_data, fork_step
            )

        # Create run in target platform
        target_run_id = self._data_loader.create_run(
            project_id=project_id,
            run_name=custom_run_id,
            experiment_id=target_experiment_id,
            parent_run_id=parent_target_run_id,
            fork_step=fork_step,
            step_multiplier=step_multiplier,
        )
        run_id_to_target_run_id[source_run_id] = target_run_id

        # Upload run data
        self._data_loader.upload_run_data(
            run_data=run_data,
            run_id=target_run_id,
            files_directory=self._files_directory / sanitize_path_part(project_id),
            fork_step=fork_step,
            step_multiplier=step_multiplier,
        )

        # Mark as processed
        processed_runs.add(source_run_id)

        # Recursively process all buffered children of this run
        if source_run_id in parent_to_children:
            for child_source_run_id in parent_to_children[source_run_id]:
                if child_source_run_id in buffered_runs:
                    child_info = buffered_runs.pop(child_source_run_id)
                    self._process_run(
                        source_run_id=child_source_run_id,
                        run_data=child_info["data"],
                        metadata=child_info["metadata"],
                        processed_runs=processed_runs,
                        parent_to_children=parent_to_children,
                        run_id_to_target_run_id=run_id_to_target_run_id,
                        buffered_runs=buffered_runs,
                    )

    @staticmethod
    def _get_attribute_value(
        table: pa.Table, attribute_path: str, attribute_type: str = "string_value"
    ) -> Optional[str]:
        mask = pc.equal(table["attribute_path"], attribute_path)
        if pc.sum(mask).as_py() > 0:
            return table.filter(mask)[attribute_type].take([0]).to_pylist()[0]
        return None
