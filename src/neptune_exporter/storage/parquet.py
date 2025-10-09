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
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ProjectWriter:
    """Tracks the state of a parquet writer for a specific project."""

    project_id: str
    writer: pq.ParquetWriter
    current_file_path: Path
    current_part: int = 0

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()

    def get_compressed_size(self) -> int:
        """Get the current compressed file size."""
        if self.current_file_path.exists():
            return self.current_file_path.stat().st_size
        return 0


class ProjectWriterContext:
    """Context manager for writing to a specific project."""

    def __init__(self, storage: "ParquetStorage", project_id: str):
        self.storage = storage
        self.project_id = project_id

    def __enter__(self) -> "ProjectWriterContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.storage.close_project(self.project_id)

    def save(self, data: pa.RecordBatch) -> None:
        """Save a RecordBatch to this project."""
        self.storage.save(self.project_id, data)


class ParquetStorage:
    def __init__(
        self, base_path: Path, target_part_size_bytes: int = 200 * 1024 * 1024
    ):
        self.base_path = base_path
        self._target_part_size_bytes = target_part_size_bytes
        self._initialize_directory()

        # Track current part state per project
        self._project_writers: dict[str, ProjectWriter] = {}

    def _initialize_directory(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, project_id: str, data: pa.RecordBatch) -> None:
        """Stream data to current part, creating new part when compressed size limit reached."""
        if project_id not in self._project_writers:
            # Create first part for new project
            self._create_new_part(project_id, data)
            return

        writer_state = self._project_writers[project_id]

        # If current part is too large, start new part
        if writer_state.get_compressed_size() > self._target_part_size_bytes:
            # Close current writer
            writer_state.close()
            # Create new part
            self._create_new_part(project_id, data)
        else:
            # Write batch to current part
            writer_state.writer.write_batch(data)

    def project_writer(self, project_id: str) -> ProjectWriterContext:
        """Get a context manager for writing to a specific project."""
        return ProjectWriterContext(self, project_id)

    def close_project(self, project_id: str) -> None:
        """Close the writer for a specific project."""
        if project_id in self._project_writers:
            self._project_writers[project_id].close()

    def _create_new_part(self, project_id: str, data: pa.RecordBatch) -> None:
        """Create a new part for the given project."""
        if project_id not in self._project_writers:
            # New project - start with part 0
            current_part = 0
        else:
            # Existing project - increment part number
            current_part = self._project_writers[project_id].current_part + 1

        table_path = self.base_path / f"{project_id}/part_{current_part}.parquet"
        table_path.parent.mkdir(parents=True, exist_ok=True)

        writer = pq.ParquetWriter(table_path, data.schema, compression="snappy")

        # Create or update ProjectWriter
        self._project_writers[project_id] = ProjectWriter(
            project_id=project_id,
            writer=writer,
            current_file_path=table_path,
            current_part=current_part,
        )

        # Write the data to the new part
        writer.write_batch(data)

    def close_all(self) -> None:
        """Close all open writers."""
        for writer_state in self._project_writers.values():
            writer_state.close()
