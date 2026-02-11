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
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from neptune_exporter import model
from neptune_exporter.model_registry_summary_manager import ModelRegistrySummaryManager
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.utils import sanitize_path_part


def test_model_registry_summary_manager_empty():
    """Test model registry summary manager with no model registry data."""
    with TemporaryDirectory() as temp_dir:
        reader = ParquetReader(Path(temp_dir))
        manager = ModelRegistrySummaryManager(parquet_reader=reader)

        summary = manager.get_data_summary()

        assert summary["total_projects"] == 0
        assert summary["total_models"] == 0
        assert summary["total_model_versions"] == 0
        assert summary["projects"] == {}


def test_model_registry_summary_manager_with_models_and_versions():
    """Test model registry summary manager with model and model version data."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_id = "workspace/project"
        project_dir = temp_path / sanitize_path_part(project_id)
        models_dir = project_dir / "models"
        versions_dir = project_dir / "model_versions"
        models_dir.mkdir(parents=True)
        versions_dir.mkdir(parents=True)

        model_table = pa.Table.from_pydict(
            {
                "project_id": [project_id],
                "run_id": ["MODEL-1"],
                "attribute_path": ["sys/name"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["baseline-model"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            },
            schema=model.SCHEMA,
        )
        model_version_table = pa.Table.from_pydict(
            {
                "project_id": [project_id],
                "run_id": ["MODEL-1-1"],
                "attribute_path": ["sys/stage"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["production"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            },
            schema=model.SCHEMA,
        )

        pq.write_table(
            model_table,
            models_dir / f"{sanitize_path_part('MODEL-1')}_part_0.parquet",
        )
        pq.write_table(
            model_version_table,
            versions_dir / f"{sanitize_path_part('MODEL-1-1')}_part_0.parquet",
        )

        reader = ParquetReader(temp_path)
        manager = ModelRegistrySummaryManager(parquet_reader=reader)

        summary = manager.get_data_summary()

        assert summary["total_projects"] == 1
        assert summary["total_models"] == 1
        assert summary["total_model_versions"] == 1
        project_summary = summary["projects"][project_dir]
        assert project_summary["project_id"] == project_id
        assert project_summary["models"]["total_entities"] == 1
        assert project_summary["model_versions"]["total_entities"] == 1
