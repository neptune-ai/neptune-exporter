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
from pathlib import Path
from typing import Any

import pyarrow.compute as pc

from neptune_exporter.storage.parquet_reader import ParquetReader


class ModelRegistrySummaryManager:
    """Manages analysis and reporting of exported model registry data."""

    def __init__(self, parquet_reader: ParquetReader):
        self._parquet_reader = parquet_reader
        self._logger = logging.getLogger(__name__)

    def get_data_summary(self) -> dict[str, Any]:
        model_projects = set(
            self._parquet_reader.list_project_directories(entity_scope="models")
        )
        model_version_projects = set(
            self._parquet_reader.list_project_directories(entity_scope="model_versions")
        )
        project_directories = sorted(model_projects | model_version_projects)

        summary: dict[str, Any] = {
            "total_projects": len(project_directories),
            "total_models": 0,
            "total_model_versions": 0,
            "projects": {},
        }

        for project_directory in project_directories:
            project_summary = self.get_project_summary(project_directory)
            summary["projects"][project_directory] = project_summary
            if project_summary is None:
                continue
            summary["total_models"] += project_summary["models"]["total_entities"]
            summary["total_model_versions"] += project_summary["model_versions"][
                "total_entities"
            ]

        return summary

    def get_project_summary(self, project_directory: Path) -> dict[str, Any] | None:
        try:
            models_summary = self._get_entity_summary(project_directory, "models")
            model_versions_summary = self._get_entity_summary(
                project_directory, "model_versions"
            )
            project_id = (
                models_summary["project_id"] or model_versions_summary["project_id"]
            )

            return {
                "project_id": project_id,
                "models": models_summary,
                "model_versions": model_versions_summary,
            }
        except Exception:
            self._logger.error(
                f"Error analyzing model registry project {project_directory}",
                exc_info=True,
            )
            return None

    def _get_entity_summary(
        self, project_directory: Path, entity_scope: str
    ) -> dict[str, Any]:
        project_id: str | None = None
        total_records = 0
        unique_entities: set[Any] = set()
        unique_attribute_types: set[Any] = set()
        attribute_paths_by_type: dict[Any, set[Any]] = {}

        for table in self._parquet_reader.read_project_data(
            project_directory, entity_scope=entity_scope
        ):
            if len(table) == 0:
                continue

            total_records += len(table)
            if project_id is None:
                project_id = table["project_id"][0].as_py()

            unique_entities.update(pc.unique(table["run_id"]).to_pylist())
            table_attribute_types = pc.unique(table["attribute_type"]).to_pylist()
            unique_attribute_types.update(table_attribute_types)

            for attr_type in table_attribute_types:
                type_mask = pc.equal(table["attribute_type"], attr_type)
                unique_paths = pc.unique(
                    pc.filter(table["attribute_path"], type_mask)
                ).to_pylist()
                if attr_type not in attribute_paths_by_type:
                    attribute_paths_by_type[attr_type] = set()
                attribute_paths_by_type[attr_type].update(unique_paths)

        if total_records == 0:
            return {
                "project_id": None,
                "total_entities": 0,
                "entities": [],
                "total_records": 0,
                "attribute_types": [],
                "attribute_breakdown": {},
            }

        attribute_breakdown: dict[Any, int] = {
            attr_type: len(paths)
            for attr_type, paths in attribute_paths_by_type.items()
        }

        return {
            "project_id": project_id,
            "total_entities": len(unique_entities),
            "entities": sorted(unique_entities, key=lambda x: str(x)),
            "total_records": total_records,
            "attribute_types": sorted(unique_attribute_types, key=lambda x: str(x)),
            "attribute_breakdown": attribute_breakdown,
        }
