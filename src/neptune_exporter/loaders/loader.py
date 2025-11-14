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

"""Core loader abstract base class for data loading to target platforms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional
import pyarrow as pa

from neptune_exporter.types import TargetExperimentId, TargetRunId


class DataLoader(ABC):
    """Abstract base class for data loaders that upload Neptune data to target platforms."""

    @abstractmethod
    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """
        Create or get an experiment/project in the target platform.

        Args:
            project_id: Neptune project ID
            experiment_name: Name of the experiment

        Returns:
            Experiment/project ID in the target platform
        """
        pass

    @abstractmethod
    def find_run(
        self,
        project_id: str,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find a run by name in an experiment."""
        pass

    @abstractmethod
    def create_run(
        self,
        project_id: str,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """
        Create a run in the target platform.

        Args:
            project_id: Neptune project ID
            run_name: Name of the run
            experiment_id: Optional experiment/project ID in target platform
            parent_run_id: Optional parent run ID for nested runs
            fork_step: Optional fork step if this is a forked run
            step_multiplier: Optional step multiplier for converting decimal steps to integers
                (used by W&B for fork_step conversion)

        Returns:
            Run ID in the target platform
        """
        pass

    @abstractmethod
    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """
        Upload all data for a single run to the target platform.

        Args:
            run_data: PyArrow table containing run data
            run_id: Run ID in the target platform
            files_directory: Base directory for file artifacts
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        pass
