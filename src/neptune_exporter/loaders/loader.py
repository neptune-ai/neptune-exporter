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

"""Core loader protocol for data loading to target platforms."""

from pathlib import Path
from typing import Optional, Protocol
import pyarrow as pa


class DataLoader(Protocol):
    """Protocol for data loaders that upload Neptune data to target platforms."""

    def create_experiment(self, project_id: str, experiment_name: str) -> str:
        """
        Create or get an experiment/project in the target platform.

        Args:
            project_id: Neptune project ID
            experiment_name: Name of the experiment

        Returns:
            Experiment/project ID in the target platform
        """
        ...

    def create_run(
        self,
        project_id: str,
        run_name: str,
        experiment_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> str:
        """
        Create a run in the target platform.

        Args:
            project_id: Neptune project ID
            run_name: Name of the run
            experiment_id: Optional experiment/project ID in target platform
            parent_run_id: Optional parent run ID for nested runs

        Returns:
            Run ID in the target platform
        """
        ...

    def update_run_parent(self, child_run_id: str, parent_run_id: str) -> None:
        """
        Update the parent relationship of an existing run.

        Args:
            child_run_id: ID of the child run
            parent_run_id: ID of the parent run
        """
        ...

    def upload_run_data(
        self,
        run_data: pa.Table,
        run_id: str,
        files_directory: Path,
    ) -> None:
        """
        Upload all data for a single run to the target platform.

        Args:
            run_data: PyArrow table containing run data
            run_id: Run ID in the target platform
            files_directory: Base directory for file artifacts
        """
        ...
