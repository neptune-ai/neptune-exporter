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
from pathlib import Path
from typing import Iterable, Literal

import pyarrow as pa
import pyarrow.compute as pc

from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter
from neptune_exporter.progress.listeners import ProgressListenerFactory
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.storage.parquet_writer import ParquetWriter, RunWriterContext
from neptune_exporter.types import ProjectId, SourceRunId
from neptune_exporter.utils import sanitize_path_part


_EXPORT_CLASS = Literal["parameters", "metrics", "series", "files"]
_ENTITY_SCOPE = Literal["models", "model_versions"]


@dataclass(frozen=True)
class ModelRegistryExportResult:
    total_models: int
    skipped_models: int
    total_model_versions: int
    skipped_model_versions: int


class ModelRegistryExportManager:
    def __init__(
        self,
        exporter: Neptune2Exporter,
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
        models_query: None | str = None,
        attributes: None | str | list[str] = None,
        include_archived_models: bool = False,
        export_classes: Iterable[_EXPORT_CLASS] = {
            "parameters",
            "metrics",
            "series",
            "files",
        },
    ) -> ModelRegistryExportResult:
        project_models = {
            project_id: self._exporter.list_models(
                project_id=project_id,
                query=models_query,
                include_trashed=include_archived_models,
            )
            for project_id in project_ids
        }

        total_models = sum(len(model_ids) for model_ids in project_models.values())
        skipped_models = 0
        total_model_versions = 0
        skipped_model_versions = 0

        live = self._progress_listener_factory.create_live()

        with live:
            listener = self._progress_listener_factory.create_listener(live)
            for project_id, all_model_ids in project_models.items():
                # Export models
                model_ids = [
                    model_id
                    for model_id in all_model_ids
                    if not self._reader.check_run_exists(
                        project_id, model_id, entity_scope="models"
                    )
                ]
                skipped_models += len(all_model_ids) - len(model_ids)
                listener.on_project_total(project_id, len(model_ids))

                self._export_entities(
                    project_id=project_id,
                    entity_ids=model_ids,
                    entity_scope="models",
                    attributes=attributes,
                    export_classes=export_classes,
                    listener=listener,
                )

                # Export model versions for all selected models
                all_model_version_ids: list[SourceRunId] = []
                for model_id in all_model_ids:
                    try:
                        all_model_version_ids.extend(
                            self._exporter.list_model_versions(
                                project_id=project_id, model_id=model_id
                            )
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"Failed to list model versions for {project_id}/{model_id}",
                            exc_info=True,
                        )
                        self._error_reporter.record_exception(
                            project_id=project_id,
                            run_id=model_id,
                            attribute_path=None,
                            attribute_type=None,
                            exception=e,
                        )

                total_model_versions += len(all_model_version_ids)
                model_version_ids = [
                    model_version_id
                    for model_version_id in all_model_version_ids
                    if not self._reader.check_run_exists(
                        project_id,
                        model_version_id,
                        entity_scope="model_versions",
                    )
                ]
                skipped_model_versions += len(all_model_version_ids) - len(
                    model_version_ids
                )
                listener.on_project_total(project_id, len(model_version_ids))
                self._export_entities(
                    project_id=project_id,
                    entity_ids=model_version_ids,
                    entity_scope="model_versions",
                    attributes=attributes,
                    export_classes=export_classes,
                    listener=listener,
                )

        return ModelRegistryExportResult(
            total_models=total_models,
            skipped_models=skipped_models,
            total_model_versions=total_model_versions,
            skipped_model_versions=skipped_model_versions,
        )

    def _export_entities(
        self,
        project_id: ProjectId,
        entity_ids: list[SourceRunId],
        entity_scope: _ENTITY_SCOPE,
        attributes: None | str | list[str],
        export_classes: Iterable[_EXPORT_CLASS],
        listener,
    ) -> None:
        if not entity_ids:
            return

        for batch_start in range(0, len(entity_ids), self._batch_size):
            batch_entity_ids = entity_ids[batch_start : batch_start + self._batch_size]
            for entity_id in batch_entity_ids:
                listener.on_run_started(entity_id)

            writers = {
                entity_id: self._writer.run_writer(
                    project_id, entity_id, entity_scope=entity_scope
                )
                for entity_id in batch_entity_ids
            }

            try:
                if "parameters" in export_classes:
                    for batch in self._download_parameters(
                        project_id, batch_entity_ids, attributes, entity_scope, listener
                    ):
                        self._route_batch_to_writers(batch, writers)

                if "metrics" in export_classes:
                    for batch in self._download_metrics(
                        project_id, batch_entity_ids, attributes, entity_scope, listener
                    ):
                        self._route_batch_to_writers(batch, writers)

                if "series" in export_classes:
                    for batch in self._download_series(
                        project_id, batch_entity_ids, attributes, entity_scope, listener
                    ):
                        self._route_batch_to_writers(batch, writers)

                if "files" in export_classes:
                    for batch in self._download_files(
                        project_id, batch_entity_ids, attributes, entity_scope, listener
                    ):
                        self._route_batch_to_writers(batch, writers)
            finally:
                for writer in writers.values():
                    writer.finish_run()
                for entity_id in batch_entity_ids:
                    listener.on_run_finished(entity_id)

            listener.on_project_advance(project_id, len(batch_entity_ids))

    def _download_parameters(
        self,
        project_id: ProjectId,
        entity_ids: list[SourceRunId],
        attributes: None | str | list[str],
        entity_scope: _ENTITY_SCOPE,
        listener,
    ):
        if entity_scope == "models":
            return self._exporter.download_model_parameters(
                project_id=project_id,
                model_ids=entity_ids,
                attributes=attributes,
                progress=listener,
            )
        return self._exporter.download_model_version_parameters(
            project_id=project_id,
            model_version_ids=entity_ids,
            attributes=attributes,
            progress=listener,
        )

    def _download_metrics(
        self,
        project_id: ProjectId,
        entity_ids: list[SourceRunId],
        attributes: None | str | list[str],
        entity_scope: _ENTITY_SCOPE,
        listener,
    ):
        if entity_scope == "models":
            return self._exporter.download_model_metrics(
                project_id=project_id,
                model_ids=entity_ids,
                attributes=attributes,
                progress=listener,
            )
        return self._exporter.download_model_version_metrics(
            project_id=project_id,
            model_version_ids=entity_ids,
            attributes=attributes,
            progress=listener,
        )

    def _download_series(
        self,
        project_id: ProjectId,
        entity_ids: list[SourceRunId],
        attributes: None | str | list[str],
        entity_scope: _ENTITY_SCOPE,
        listener,
    ):
        if entity_scope == "models":
            return self._exporter.download_model_series(
                project_id=project_id,
                model_ids=entity_ids,
                attributes=attributes,
                progress=listener,
            )
        return self._exporter.download_model_version_series(
            project_id=project_id,
            model_version_ids=entity_ids,
            attributes=attributes,
            progress=listener,
        )

    def _download_files(
        self,
        project_id: ProjectId,
        entity_ids: list[SourceRunId],
        attributes: None | str | list[str],
        entity_scope: _ENTITY_SCOPE,
        listener,
    ):
        destination = (
            self._files_destination / sanitize_path_part(project_id) / entity_scope
        )
        if entity_scope == "models":
            return self._exporter.download_model_files(
                project_id=project_id,
                model_ids=entity_ids,
                attributes=attributes,
                destination=destination,
                progress=listener,
            )
        return self._exporter.download_model_version_files(
            project_id=project_id,
            model_version_ids=entity_ids,
            attributes=attributes,
            destination=destination,
            progress=listener,
        )

    def _route_batch_to_writers(
        self,
        batch: pa.RecordBatch,
        writers: dict[SourceRunId, RunWriterContext],
    ) -> None:
        if batch.num_rows == 0:
            return

        run_id_array = batch.column("run_id")
        unique_run_ids = set(run_id_array.unique().to_pylist())

        for run_id in unique_run_ids:
            if run_id not in writers:
                continue

            filtered_batch = batch.filter(pc.equal(run_id_array, run_id))
            if filtered_batch.num_rows > 0:
                writers[run_id].save(filtered_batch)
