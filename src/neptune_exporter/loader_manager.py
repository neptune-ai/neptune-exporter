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
import heapq
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

from neptune_exporter.storage.parquet_reader import ParquetReader, RunMetadata
from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.types import SourceRunId, RunFilePrefix, TargetRunId
from neptune_exporter.utils import sanitize_path_part


class LoaderManager:
    """Manages the loading of Neptune data from parquet files to target platforms."""

    def __init__(
        self,
        parquet_reader: ParquetReader,
        data_loader: DataLoader,
        files_directory: Path,
        step_multiplier: int,
    ):
        self._parquet_reader = parquet_reader
        self._data_loader = data_loader
        self._files_directory = files_directory
        self._step_multiplier = step_multiplier
        self._logger = logging.getLogger(__name__)

    def load(
        self,
        project_ids: Optional[list[str]] = None,
        runs: Optional[list[SourceRunId]] = None,
    ) -> None:
        """
        Load Neptune data from files to target platforms.

        Args:
            project_ids: List of project IDs to load. If None, loads all available projects.
            runs: Set of run IDs to filter by. If None, loads all runs.
        """
        # Get projects to process
        project_directories = self._parquet_reader.list_project_directories(project_ids)

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
                self._load_project(project_directory, runs=runs)
            except Exception:
                self._logger.error(
                    f"Error loading project {project_directory}", exc_info=True
                )
                continue

        self._logger.info("Data loading completed")

    def _topological_sort_runs(
        self, run_metadata: list[RunMetadata]
    ) -> list[RunMetadata]:
        """Topologically sort runs so parents are processed before children.

        Uses Kahn's algorithm. Orphaned runs (parent not in dataset) are treated
        as root nodes and processed first.

        Args:
            run_metadata: List of run metadata

        Returns:
            List of run metadata in topological order (parents before children)
        """
        # Build dependency graph: parent_run_id -> list[child_run_id]
        parent_to_children: dict[SourceRunId, list[RunMetadata]] = {}

        # Track in-degree for each run (if it has a parent that is in the dataset, it has 1 in-degree)
        in_degree: dict[SourceRunId, int] = {}

        # Initialize all runs
        for metadata in run_metadata:
            in_degree[metadata.run_id] = 0

        # Build graph and calculate in-degrees
        for metadata in run_metadata:
            parent_source_run_id = (
                metadata.parent_source_run_id
                if metadata.parent_source_run_id is not None
                else None
            )

            # If parent exists in dataset, add edge and increment in-degree
            if parent_source_run_id is not None and parent_source_run_id in in_degree:
                if parent_source_run_id not in parent_to_children:
                    parent_to_children[parent_source_run_id] = []
                parent_to_children[parent_source_run_id].append(metadata)
                in_degree[metadata.run_id] = 1
            # Otherwise, run is a root (orphaned or no parent)

        # Kahn's algorithm: start with nodes with in-degree 0
        # Results are sorted by metadata.creation_time, those without creation time are last
        queue = [
            (metadata.creation_time or datetime.datetime.max, metadata)
            for metadata in run_metadata
            if in_degree[metadata.run_id] == 0
        ]
        heapq.heapify(queue)
        result: list[RunMetadata] = []

        while queue:
            _, metadata = heapq.heappop(queue)
            result.append(metadata)

            # Process children of this run
            source_run_id = metadata.run_id
            if source_run_id in parent_to_children:
                for child_metadata in parent_to_children[source_run_id]:
                    heapq.heappush(
                        queue,
                        (
                            child_metadata.creation_time or datetime.datetime.max,
                            child_metadata,
                        ),
                    )

        # Check for cycles (shouldn't happen in Neptune, but defensive)
        if len(result) != len(run_metadata):
            remaining_len = len(run_metadata) - len(result)
            self._logger.warning(
                f"Circular dependency detected or missing runs. "
                f"Remaining runs: {remaining_len}"
            )
            result_ids = set(metadata.run_id for metadata in result)
            remaining_metadata = [
                metadata
                for metadata in run_metadata
                if metadata.run_id not in result_ids
            ]
            remaining_metadata.sort(
                key=lambda x: (x.creation_time or datetime.datetime.max, x.run_id)
            )
            result.extend(remaining_metadata)

        return result

    def _load_project(
        self,
        project_directory: Path,
        runs: Optional[list[SourceRunId]],
    ) -> None:
        """Load a single project to target platform using topological sorting.

        Reads metadata for all runs upfront, builds a dependency graph,
        topologically sorts runs (parents before children), and processes them in order.
        """
        self._logger.info(f"Loading data from {project_directory} to target platform")

        # List all complete runs in the project
        all_run_file_prefixes = self._parquet_reader.list_run_files(
            project_directory, runs
        )

        if not all_run_file_prefixes:
            self._logger.warning(f"No complete runs found in {project_directory}")
            return

        # Read metadata for all runs upfront
        run_metadata: list[RunMetadata] = []
        source_run_id_to_file_prefix: dict[SourceRunId, RunFilePrefix] = {}

        for source_run_file_prefix in tqdm(
            all_run_file_prefixes,
            desc="Reading run metadata",
            unit="run",
            leave=False,
        ):
            metadata = self._parquet_reader.read_run_metadata(
                project_directory, source_run_file_prefix
            )

            if metadata is None:
                self._logger.warning(
                    f"Could not read metadata for run {source_run_file_prefix}, skipping"
                )
                continue

            source_run_id_to_file_prefix[metadata.run_id] = source_run_file_prefix
            run_metadata.append(metadata)

        if not run_metadata:
            self._logger.warning(f"No valid run metadata found in {project_directory}")
            return

        # Topologically sort runs (parents before children)
        sorted_run_metadata = self._topological_sort_runs(run_metadata)

        # Track target run IDs for parent lookups
        run_id_to_target_run_id: dict[SourceRunId, TargetRunId] = {}

        # Process runs in topological order
        for metadata in tqdm(
            sorted_run_metadata,
            desc=f"Loading runs from {project_directory}",
            unit="run",
            leave=False,
        ):
            try:
                self._process_run(
                    project_directory=project_directory,
                    source_run_file_prefix=source_run_id_to_file_prefix[
                        metadata.run_id
                    ],
                    metadata=metadata,
                    run_id_to_target_run_id=run_id_to_target_run_id,
                )
            except Exception:
                self._logger.error(
                    f"Error processing run {source_run_file_prefix}",
                    exc_info=True,
                )
                continue

    def _process_run(
        self,
        project_directory: Path,
        source_run_file_prefix: RunFilePrefix,
        metadata: RunMetadata,
        run_id_to_target_run_id: dict[SourceRunId, TargetRunId],
    ) -> None:
        """Process a single run.

        Reads run data from disk using read_run_data() and uploads it to the target platform part by part.

        Args:
            project_directory: Project directory
            source_run_file_prefix: Source run file prefix from Neptune
            metadata: Run metadata (project_id, custom_run_id, etc.)
            run_id_to_target_run_id: Dictionary mapping source run IDs to target run IDs
        """
        project_id = metadata.project_id
        custom_run_id = metadata.custom_run_id or metadata.run_id
        experiment_name = metadata.experiment_name
        parent_source_run_id = (
            metadata.parent_source_run_id
            if metadata.parent_source_run_id is not None
            else None
        )
        fork_step = metadata.fork_step

        # Read run data from disk (all parts), yielding one part at a time
        run_data_parts_generator = self._parquet_reader.read_run_data(
            project_directory, source_run_file_prefix
        )

        # Get or create experiment
        if experiment_name is not None:
            target_experiment_id = self._data_loader.create_experiment(
                project_id=project_id, experiment_name=experiment_name
            )
        else:
            target_experiment_id = None

        # Check if run already exists (for resumable loading)
        target_run_id = self._data_loader.find_run(
            project_id=project_id,
            run_name=custom_run_id,
            experiment_id=target_experiment_id,
        )

        # Create run if it doesn't exist
        if target_run_id is None:
            # Get parent target run ID if parent exists
            parent_target_run_id = None
            if parent_source_run_id and parent_source_run_id in run_id_to_target_run_id:
                parent_target_run_id = run_id_to_target_run_id[parent_source_run_id]

            target_run_id = self._data_loader.create_run(
                project_id=project_id,
                run_name=custom_run_id,
                experiment_id=target_experiment_id,
                parent_run_id=parent_target_run_id,
                fork_step=fork_step,
                step_multiplier=self._step_multiplier,
            )
            # Upload run data only for newly created runs
            self._data_loader.upload_run_data(
                run_data=run_data_parts_generator,
                run_id=target_run_id,
                files_directory=self._files_directory / sanitize_path_part(project_id),
                step_multiplier=self._step_multiplier,
            )
            self._logger.info(f"Created and uploaded run '{custom_run_id}'")
        else:
            self._logger.info(
                f"Found existing run '{custom_run_id}' with ID {target_run_id}, skipping upload"
            )

        # Store mapping for parent/child relationships
        run_id_to_target_run_id[metadata.run_id] = TargetRunId(target_run_id)
