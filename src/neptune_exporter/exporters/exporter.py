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

"""Core exporter abstract base class and types."""

from abc import ABC, abstractmethod
from typing import Generator, Optional, Sequence
from pathlib import Path
import pyarrow as pa

from neptune_exporter.types import ProjectId, SourceRunId


class NeptuneExporter(ABC):
    """Abstract base class for Neptune data exporters."""

    @abstractmethod
    def list_projects(self) -> list[ProjectId]:
        """List Neptune projects."""
        pass

    @abstractmethod
    def list_runs(
        self, project_id: ProjectId, runs: Optional[str] = None
    ) -> list[SourceRunId]:
        """List Neptune runs."""
        pass

    @abstractmethod
    def download_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download parameters from Neptune runs."""
        pass

    @abstractmethod
    def download_metrics(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download metrics from Neptune runs."""
        pass

    @abstractmethod
    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download series data from Neptune runs."""
        pass

    @abstractmethod
    def download_files(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        destination: Path,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download files from Neptune runs."""
        pass

    def close(self) -> None:
        """Close the exporter."""
        pass
