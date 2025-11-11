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
from dataclasses import dataclass, field
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from neptune_exporter.utils import sanitize_path_part


@dataclass
class RunWriter:
    """Tracks the state of a parquet writer for a specific run."""

    project_id: str
    run_id: str
    writer: pq.ParquetWriter
    current_file_path: Path
    current_part: int = 0
    part_paths: list[Path] = field(default_factory=list)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()

    def get_compressed_size(self) -> int:
        """Get the current compressed file size."""
        if self.current_file_path.exists():
            return self.current_file_path.stat().st_size
        return 0


class RunWriterContext:
    """Context manager for writing to a specific run."""

    def __init__(self, storage: "ParquetWriter", project_id: str, run_id: str):
        self.storage = storage
        self.project_id = project_id
        self.run_id = run_id

    def __enter__(self) -> "RunWriterContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.storage.finish_run(self.project_id, self.run_id)

    def save(self, data: pa.RecordBatch) -> None:
        """Save a RecordBatch to this run."""
        self.storage.save(self.project_id, self.run_id, data)

    def finish_run(self) -> None:
        """Signal that the current run is complete."""
        self.storage.finish_run(self.project_id, self.run_id)


class ParquetWriter:
    def __init__(
        self, base_path: Path, target_part_size_bytes: int = 200 * 1024 * 1024
    ):
        self.base_path = base_path
        self._target_part_size_bytes = target_part_size_bytes
        self._initialize_directory()
        self._logger = logging.getLogger(__name__)

        # Track current part state per run (keyed by (project_id, run_id))
        self._run_writers: dict[tuple[str, str], RunWriter] = {}

    def _initialize_directory(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    def run_writer(self, project_id: str, run_id: str) -> RunWriterContext:
        """Get a context manager for writing to a specific run.

        Cleans up any leftover .tmp files from previous interrupted writes for this run.
        """
        # Clean up any leftover .tmp files from previous interrupted writes
        self._cleanup_existing_parts(project_id, run_id)
        return RunWriterContext(self, project_id, run_id)

    def save(self, project_id: str, run_id: str, data: pa.RecordBatch) -> None:
        """Stream data to current part, creating new part when needed.

        Enforces one run per file - validates that run_id in data matches current run.
        Creates a new part if the current part exceeds the size limit.

        Args:
            project_id: The project ID
            run_id: The run ID
            data: The RecordBatch to save
        """
        run_key = (project_id, run_id)

        if run_key not in self._run_writers:
            # Create first part for new run
            self._create_new_part(project_id, run_id, data)
            return

        writer_state = self._run_writers[run_key]

        # Check if current part exceeds size limit and create new part if needed
        if writer_state.get_compressed_size() > self._target_part_size_bytes:
            # Close current part
            writer_state.close()
            # Create new part for this run
            self._create_new_part(project_id, run_id, data)
            return

        # Write batch to current part
        writer_state.writer.write_batch(data)

    def _cleanup_existing_parts(self, project_id: str, run_id: str) -> None:
        """Clean up any leftover files from a previous interrupted write for this run.

        Removes both .tmp files and existing .parquet files for this run, since
        we're starting a fresh write and want to overwrite any incomplete previous attempt.

        Args:
            project_id: The project ID
            run_id: The run ID
        """
        sanitized_project_id = sanitize_path_part(project_id)
        sanitized_run_id = sanitize_path_part(run_id)

        project_dir = self.base_path / sanitized_project_id
        if not project_dir.exists():
            return

        # Find all .tmp files for this run
        tmp_pattern = f"{sanitized_run_id}_part_*.parquet.tmp"
        tmp_files = list(project_dir.glob(tmp_pattern))

        # Find all .parquet files for this run (incomplete previous write)
        parquet_pattern = f"{sanitized_run_id}_part_*.parquet"
        parquet_files = list(project_dir.glob(parquet_pattern))

        # Delete any leftover files (both .tmp and .parquet)
        all_files = tmp_files + parquet_files
        for file_path in all_files:
            try:
                file_path.unlink()
            except Exception as e:
                self._logger.warning(f"Failed to delete leftover file {file_path}: {e}")

    def _create_new_part(
        self, project_id: str, run_id: str, data: pa.RecordBatch
    ) -> None:
        """Create a new part for the given run."""
        run_key = (project_id, run_id)

        # Determine next part number from existing parts
        if run_key in self._run_writers:
            # Increment from last part
            current_part = self._run_writers[run_key].current_part + 1
        else:
            # Brand new run - start with part 0
            current_part = 0

        # Sanitize project_id and run_id for safe file path usage
        sanitized_project_id = sanitize_path_part(project_id)
        sanitized_run_id = sanitize_path_part(
            run_id
        )  # !! can sanitization result in collisions?

        # Create temporary file path: run_id_part_N.parquet.tmp
        table_path = (
            self.base_path
            / sanitized_project_id
            / f"{sanitized_run_id}_part_{current_part}.parquet.tmp"
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)

        writer = pq.ParquetWriter(table_path, data.schema, compression="snappy")

        # Create or update RunWriter
        if run_key in self._run_writers:
            # Add to existing run writer
            writer_state = self._run_writers[run_key]
            writer_state.writer = writer
            writer_state.current_file_path = table_path
            writer_state.current_part = current_part
            writer_state.part_paths.append(table_path)
        else:
            # Create new run writer
            self._run_writers[run_key] = RunWriter(
                project_id=project_id,
                run_id=run_id,
                writer=writer,
                current_file_path=table_path,
                current_part=current_part,
                part_paths=[table_path],
            )

        # Write the data to the new part
        writer.write_batch(data)

    def finish_run(self, project_id: str, run_id: str) -> None:
        """Signal that a run is complete.

        Closes the current part and renames all .tmp files to final location,
        moving part_0 last to indicate completion.

        Args:
            project_id: The project ID
            run_id: The run ID
        """
        run_key = (project_id, run_id)
        if run_key not in self._run_writers:
            return  # No active writer

        writer_state = self._run_writers[run_key]

        # Close current part
        writer_state.close()

        # Get all part paths before removing from tracking
        part_paths = writer_state.part_paths.copy()

        # Remove from tracking
        del self._run_writers[run_key]

        # Rename all parts for this run
        self._rename_run_parts(part_paths)

    def _rename_run_parts(self, part_paths: list[Path]) -> None:
        """Rename all .tmp files to final location, moving part_0 last.

        Args:
            part_paths: List of temporary file paths to rename
        """
        if not part_paths:
            return

        # Sort parts by part number (extract from filename)
        def get_part_number(path: Path) -> int:
            # Extract part number from filename like "run_id_part_N.parquet.tmp"
            stem = path.stem  # "run_id_part_N.parquet"
            if stem.endswith(".parquet"):
                stem = stem[:-8]  # Remove ".parquet"
            # Split on "_part_" and get the last part
            parts = stem.rsplit("_part_", 1)
            if len(parts) == 2:
                return int(parts[1])
            raise ValueError(f"Could not extract part number from {path}")

        sorted_parts = sorted(part_paths, key=get_part_number, reverse=True)

        # Rename parts in reverse order, moving part_0 last to indicate completion
        for part_path in sorted_parts:
            # Remove .tmp extension
            final_path = part_path.with_suffix("")  # Remove .tmp
            part_path.rename(final_path)

    def close_all(self) -> None:
        """Close all open writers."""
        for writer_state in self._run_writers.values():
            writer_state.close()
        self._run_writers.clear()
