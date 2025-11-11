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


from typing import Iterable, Literal
from pathlib import Path
from tqdm import tqdm
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.utils import sanitize_path_part


class ExportManager:
    def __init__(
        self,
        exporter: NeptuneExporter,
        reader: ParquetReader,
        writer: ParquetWriter,
        files_destination: Path,
    ):
        self._exporter = exporter
        self._reader = reader
        self._writer = writer
        self._files_destination = files_destination

    def run(
        self,
        project_ids: list[str],
        runs: None | str = None,
        attributes: None | str | list[str] = None,
        export_classes: Iterable[
            Literal["parameters", "metrics", "series", "files"]
        ] = {"parameters", "metrics", "series", "files"},
    ) -> int:
        # Step 1: List all runs for all projects
        project_runs = {}
        for project_id in tqdm(
            project_ids, desc="Listing runs in projects", unit="project"
        ):
            run_ids = self._exporter.list_runs(project_id, runs)
            project_runs[project_id] = run_ids

        # Check if any runs were found
        total_runs = sum(len(run_ids) for run_ids in project_runs.values())
        if total_runs == 0:
            return 0

        # Step 2: Process each project's runs
        for project_id, run_ids in tqdm(
            project_runs.items(), desc="Exporting projects", unit="project"
        ):
            # Filter out already-exported runs
            original_count = len(run_ids)
            run_ids = [
                rid
                for rid in run_ids
                if not self._reader.check_run_exists(project_id, rid)
            ]
            skipped = original_count - len(run_ids)
            if skipped > 0:
                tqdm.write(
                    f"Skipping {skipped} already exported run(s) in {project_id}"
                )

            if not run_ids:
                continue  # All runs already exported or deleted, skip to next project

            for run_id in tqdm(
                run_ids,
                desc=f"Exporting runs from {project_id}",
                unit="run",
                leave=False,
            ):
                with self._writer.run_writer(project_id, run_id) as writer:
                    if "parameters" in export_classes:
                        with tqdm(
                            desc=f"  Parameters for {run_id}",
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        ) as pbar:
                            for batch in self._exporter.download_parameters(
                                project_id=project_id,
                                run_ids=[run_id],
                                attributes=attributes,
                            ):
                                writer.save(batch)
                                pbar.update(batch.nbytes)

                    if "metrics" in export_classes:
                        with tqdm(
                            desc=f"  Metrics for {run_id}",
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        ) as pbar:
                            for batch in self._exporter.download_metrics(
                                project_id=project_id,
                                run_ids=[run_id],
                                attributes=attributes,
                            ):
                                writer.save(batch)
                                pbar.update(batch.nbytes)

                    if "series" in export_classes:
                        with tqdm(
                            desc=f"  Series for {run_id}",
                            unit="B",
                            unit_scale=True,
                            leave=False,
                        ) as pbar:
                            for batch in self._exporter.download_series(
                                project_id=project_id,
                                run_ids=[run_id],
                                attributes=attributes,
                            ):
                                writer.save(batch)
                                pbar.update(batch.nbytes)

                    if "files" in export_classes:
                        with tqdm(
                            desc=f"  Files for {run_id}",
                            unit=" files",
                            leave=False,
                        ) as pbar:
                            for batch in self._exporter.download_files(
                                project_id=project_id,
                                run_ids=[run_id],
                                attributes=attributes,
                                destination=self._files_destination
                                / sanitize_path_part(project_id),
                            ):
                                writer.save(batch)
                                pbar.update(batch.num_rows)

        return total_runs
