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
from tqdm import tqdm
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.storage import ParquetStorage


class ExportManager:
    def __init__(self, exporter: NeptuneExporter, storage: ParquetStorage):
        self.exporter = exporter
        self.storage = storage

    def run(
        self,
        project_ids: list[str],
        runs: None | str = None,
        attributes: None | str | list[str] = None,
    ) -> None:
        project_run_ids = []
        for project_id in tqdm(
            project_ids, desc="Listing runs in projects", unit="project"
        ):
            run_ids = self.exporter.list_runs(project_id, runs)
            project_run_ids.extend([(project_id, run_id) for run_id in run_ids])

        # Accumulate RecordBatches for each project
        project_batches: dict[str, list[pa.RecordBatch]] = {}
        for project_id, run_id in tqdm(
            project_run_ids, desc="Exporting runs", unit="run"
        ):
            if project_id not in project_batches:
                project_batches[project_id] = []

            # Collect all RecordBatches for this run
            for batch in self.exporter.download_parameters(
                project_id=project_id, run_ids=[run_id], attributes=attributes
            ):
                project_batches[project_id].append(batch)
            for batch in self.exporter.download_metrics(
                project_id=project_id, run_ids=[run_id], attributes=attributes
            ):
                project_batches[project_id].append(batch)
            for batch in self.exporter.download_series(
                project_id=project_id, run_ids=[run_id], attributes=attributes
            ):
                project_batches[project_id].append(batch)

            # for batch in self.exporter.download_files(
            #     project_id=project_id, run_ids=[run_id], attributes=attributes
            # ):
            #     project_batches[project_id].append(batch)

        # Convert accumulated batches to tables and save
        for project_id, batches in project_batches.items():
            if batches:
                # Combine all RecordBatches into a single table
                table = pa.Table.from_batches(batches)
                self.storage.save(project_id, table)
