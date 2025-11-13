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
from typing import NewType, Optional
from collections import deque
from tqdm import tqdm
import logging

from neptune_exporter.storage.parquet_reader import ParquetReader, RunMetadata
from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.utils import sanitize_path_part


SourceRunId = NewType("SourceRunId", str)
TargetRunId = NewType("TargetRunId", str)
RunFilePrefix = NewType("RunFilePrefix", str)


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
        runs: Optional[list[str]] = None,
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
        self,
        run_metadata: dict[RunFilePrefix, RunMetadata],
        all_run_ids: set[SourceRunId],
    ) -> list[RunFilePrefix]:
        """Topologically sort runs so parents are processed before children.

        Uses Kahn's algorithm. Orphaned runs (parent not in dataset) are treated
        as root nodes and processed first.

        Args:
            run_metadata: Dictionary mapping run file prefix to metadata
            all_run_ids: Set of all run IDs in the dataset

        Returns:
            List of run file prefixes in topological order (parents before children)
        """
        # Build dependency graph: parent_run_id -> list[child_run_file_prefix]
        parent_to_children: dict[SourceRunId, list[RunFilePrefix]] = {}

        # Track in-degree for each run (how many parents it has that are in the dataset)
        in_degree: dict[RunFilePrefix, int] = {}

        # Initialize all runs
        for run_file_prefix in run_metadata.keys():
            in_degree[run_file_prefix] = 0

        # Build graph and calculate in-degrees
        for run_file_prefix, metadata in run_metadata.items():
            parent_source_run_id = (
                SourceRunId(metadata.parent_source_run_id)
                if metadata.parent_source_run_id is not None
                else None
            )

            # If parent exists in dataset, add edge and increment in-degree
            if parent_source_run_id is not None and parent_source_run_id in all_run_ids:
                if parent_source_run_id not in parent_to_children:
                    parent_to_children[parent_source_run_id] = []
                parent_to_children[parent_source_run_id].append(run_file_prefix)
                in_degree[run_file_prefix] = 1
            # Otherwise, run is a root (orphaned or no parent)

        # Kahn's algorithm: start with nodes with in-degree 0
        queue = deque(
            run_file_prefix
            for run_file_prefix, degree in in_degree.items()
            if degree == 0
        )
        result: list[RunFilePrefix] = []

        while queue:
            run_file_prefix = queue.popleft()
            result.append(run_file_prefix)

            # Get the run ID for this run to find its children
            metadata = run_metadata[run_file_prefix]
            run_id = SourceRunId(metadata.run_id)

            # Process children of this run
            if run_id in parent_to_children:
                for child_run_file_prefix in parent_to_children[run_id]:
                    in_degree[child_run_file_prefix] -= 1
                    if in_degree[child_run_file_prefix] == 0:
                        queue.append(child_run_file_prefix)

        # Check for cycles (shouldn't happen in Neptune, but defensive)
        if len(result) != len(run_metadata):
            remaining = set(run_metadata.keys()) - set(result)
            self._logger.warning(
                f"Circular dependency detected or missing runs. "
                f"Remaining runs: {remaining}"
            )
            # Add remaining runs at the end (they'll be processed but may have issues)
            result.extend(remaining)

        return result

    def _load_project(
        self,
        project_directory: Path,
        runs: Optional[list[str]],
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
        run_metadata: dict[RunFilePrefix, RunMetadata] = {}
        all_run_ids: set[SourceRunId] = set()

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

            run_file_prefix = RunFilePrefix(source_run_file_prefix)
            run_metadata[run_file_prefix] = metadata
            all_run_ids.add(SourceRunId(metadata.run_id))

        if not run_metadata:
            self._logger.warning(f"No valid run metadata found in {project_directory}")
            return

        # Topologically sort runs (parents before children)
        sorted_run_file_prefixes = self._topological_sort_runs(
            run_metadata, all_run_ids
        )

        # Get project_id for progress bar description (from first valid metadata)
        project_id = next(iter(run_metadata.values())).project_id

        # Track target run IDs for parent lookups
        run_id_to_target_run_id: dict[SourceRunId, TargetRunId] = {}

        # Process runs in topological order
        for source_run_file_prefix in tqdm(
            sorted_run_file_prefixes,
            desc=f"Loading runs from {project_id}",
            unit="run",
            leave=False,
        ):
            if source_run_file_prefix not in run_metadata:
                continue

            try:
                metadata = run_metadata[source_run_file_prefix]
                self._process_run(
                    project_directory=project_directory,
                    source_run_file_prefix=source_run_file_prefix,
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
        original_run_id = SourceRunId(metadata.run_id)
        custom_run_id = metadata.custom_run_id or original_run_id
        experiment_name = metadata.experiment_name
        parent_source_run_id = (
            SourceRunId(metadata.parent_source_run_id)
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

        # Get parent target run ID if parent exists
        parent_target_run_id = None
        if parent_source_run_id and parent_source_run_id in run_id_to_target_run_id:
            parent_target_run_id = run_id_to_target_run_id[parent_source_run_id]

        # Create run in target platform
        target_run_id = self._data_loader.create_run(
            project_id=project_id,
            run_name=custom_run_id,
            experiment_id=target_experiment_id,
            parent_run_id=parent_target_run_id,
            fork_step=fork_step,
            step_multiplier=self._step_multiplier,
        )
        run_id_to_target_run_id[SourceRunId(original_run_id)] = TargetRunId(
            target_run_id
        )

        # Upload run data
        self._data_loader.upload_run_data(
            run_data=run_data_parts_generator,
            run_id=target_run_id,
            files_directory=self._files_directory / sanitize_path_part(project_id),
            step_multiplier=self._step_multiplier,
        )
