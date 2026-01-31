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


import logging
from dataclasses import dataclass
from typing import Iterable, Literal
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.progress.listeners import ProgressListenerFactory
from neptune_exporter.storage.parquet_writer import ParquetWriter, RunWriterContext
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.types import ProjectId, SourceRunId
from neptune_exporter.utils import sanitize_path_part


@dataclass(frozen=True)
class ExportResult:
    total_runs: int
    skipped_runs: int


class ExportManager:
    def __init__(
        self,
        exporter: NeptuneExporter,
        reader: ParquetReader,
        writer: ParquetWriter,
        error_reporter: ErrorReporter,
        files_destination: Path,
        progress_listener_factory: ProgressListenerFactory,
        batch_size: int = 16,
    ):
        self._exporter = exporter
        self._reader = reader
        self._writer = writer
        self._error_reporter = error_reporter
        self._files_destination = files_destination
        self._progress_listener_factory = progress_listener_factory
        self._batch_size = batch_size
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        project_ids: list[ProjectId],
        runs: None | str = None,
        runs_query: None | str = None,
        attributes: None | str | list[str] = None,
        export_classes: Iterable[
            Literal["parameters", "metrics", "series", "files"]
        ] = {"parameters", "metrics", "series", "files"},
    ) -> ExportResult:
        # Step 1: List all runs for all projects
        project_runs = {
            project_id: self._exporter.list_runs(
                project_id=project_id, runs=runs, query=runs_query
            )
            for project_id in project_ids
        }
        skipped_runs = 0

        # Check if any runs were found
        total_runs = sum(len(run_ids) for run_ids in project_runs.values())
        if total_runs == 0:
            return ExportResult(total_runs=0, skipped_runs=0)

        live = self._progress_listener_factory.create_live()

        with live:
            listener = self._progress_listener_factory.create_listener(live)
            # Step 2: Process each project's runs
            for project_id, run_ids in project_runs.items():
                # Filter out already-exported runs
                original_count = len(run_ids)
                run_ids = [
                    rid
                    for rid in run_ids
                    if not self._reader.check_run_exists(project_id, rid)
                ]
                skipped = original_count - len(run_ids)
                if skipped > 0:
                    skipped_runs += skipped
                    self._logger.info(
                        f"Skipping {skipped} already exported run(s) in {project_id}"
                    )
                listener.on_project_total(project_id, len(run_ids))

                if not run_ids:
                    continue  # All runs already exported or deleted, skip to next project

                # Process runs in batches for concurrent downloading
                for batch_start in range(0, len(run_ids), self._batch_size):
                    batch_run_ids = run_ids[
                        batch_start : batch_start + self._batch_size
                    ]

                    for run_id in batch_run_ids:
                        listener.on_run_started(run_id)

                    # Create writers for all runs in this batch
                    writers = {
                        run_id: self._writer.run_writer(project_id, run_id)
                        for run_id in batch_run_ids
                    }

                    try:
                        if "parameters" in export_classes:
                            for batch in self._exporter.download_parameters(
                                project_id=project_id,
                                run_ids=batch_run_ids,
                                attributes=attributes,
                                progress=listener,
                            ):
                                self._route_batch_to_writers(batch, writers)

                        if "metrics" in export_classes:
                            for batch in self._exporter.download_metrics(
                                project_id=project_id,
                                run_ids=batch_run_ids,
                                attributes=attributes,
                                progress=listener,
                            ):
                                self._route_batch_to_writers(batch, writers)

                        if "series" in export_classes:
                            for batch in self._exporter.download_series(
                                project_id=project_id,
                                run_ids=batch_run_ids,
                                attributes=attributes,
                                progress=listener,
                            ):
                                self._route_batch_to_writers(batch, writers)

                        if "files" in export_classes:
                            for batch in self._exporter.download_files(
                                project_id=project_id,
                                run_ids=batch_run_ids,
                                attributes=attributes,
                                destination=self._files_destination
                                / sanitize_path_part(project_id),
                                progress=listener,
                            ):
                                self._route_batch_to_writers(batch, writers)
                    finally:
                        # Exit all writer contexts
                        for writer in writers.values():
                            writer.finish_run()
                        for run_id in batch_run_ids:
                            listener.on_run_finished(run_id)

                    listener.on_project_advance(project_id, len(batch_run_ids))

                # Keep per-project task visible after completion

        exception_summary = self._error_reporter.get_summary()
        if exception_summary.exception_count > 0:
            self._logger.error(
                f"{exception_summary.exception_count} exceptions occurred during export. See the logs for details."
            )

        return ExportResult(total_runs=total_runs, skipped_runs=skipped_runs)

    def _route_batch_to_writers(
        self,
        batch: pa.RecordBatch,
        writers: dict[SourceRunId, RunWriterContext],
    ) -> None:
        """Route a batch to the appropriate writer(s) based on run_id.

        Batches may contain data for multiple runs, so we split by run_id
        and route each split to the correct writer.
        """
        if batch.num_rows == 0:
            return

        # Get run_id column
        run_id_array = batch.column("run_id")

        # Get unique run_ids in this batch
        unique_run_ids = set(run_id_array.unique().to_pylist())

        # Split batch by run_id and route to appropriate writers
        for run_id in unique_run_ids:
            if run_id not in writers:
                # Skip if this run_id isn't in our batch (shouldn't happen! but be safe)
                continue

            # Filter batch to only rows for this run_id
            filtered_batch = batch.filter(pc.equal(run_id_array, run_id))

            if filtered_batch.num_rows > 0:
                writers[run_id].save(filtered_batch)
