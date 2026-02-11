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
from unittest.mock import Mock

import pyarrow as pa

from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.model_registry_export_manager import ModelRegistryExportManager
from neptune_exporter.progress.listeners import NoopProgressListenerFactory
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.storage.parquet_writer import ParquetWriter


def test_model_registry_export_manager_returns_zero_for_empty_projects():
    """Test model export manager returns empty result when no models are found."""
    mock_exporter = Mock()
    mock_exporter.list_models.return_value = []
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)
    mock_error_reporter = Mock(spec=ErrorReporter)

    manager = ModelRegistryExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/tmp/files"),
        progress_listener_factory=NoopProgressListenerFactory(),
    )

    result = manager.run(project_ids=["workspace/project"])

    assert result.total_models == 0
    assert result.skipped_models == 0
    assert result.total_model_versions == 0
    assert result.skipped_model_versions == 0
    mock_writer.run_writer.assert_not_called()


def test_model_registry_export_manager_skips_existing_entities():
    """Test model export manager skip logic for models and model versions."""
    mock_exporter = Mock()
    mock_exporter.list_models.return_value = ["MODEL-1", "MODEL-2"]

    def list_model_versions_side_effect(project_id, model_id):
        if model_id == "MODEL-1":
            return ["MODEL-1-1"]
        return ["MODEL-2-1"]

    mock_exporter.list_model_versions.side_effect = list_model_versions_side_effect
    mock_exporter.download_model_parameters.return_value = []
    mock_exporter.download_model_metrics.return_value = []
    mock_exporter.download_model_series.return_value = []
    mock_exporter.download_model_files.return_value = []
    mock_exporter.download_model_version_parameters.return_value = []
    mock_exporter.download_model_version_metrics.return_value = []
    mock_exporter.download_model_version_series.return_value = []
    mock_exporter.download_model_version_files.return_value = []

    mock_reader = Mock(spec=ParquetReader)

    def check_run_exists_side_effect(project_id, run_id, entity_scope=None):
        if entity_scope == "models":
            return run_id == "MODEL-1"
        if entity_scope == "model_versions":
            return run_id == "MODEL-2-1"
        return False

    mock_reader.check_run_exists.side_effect = check_run_exists_side_effect

    mock_writer = Mock(spec=ParquetWriter)
    mock_writer.run_writer.side_effect = lambda *args, **kwargs: Mock()
    mock_error_reporter = Mock(spec=ErrorReporter)

    manager = ModelRegistryExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/tmp/files"),
        progress_listener_factory=NoopProgressListenerFactory(),
    )

    result = manager.run(project_ids=["workspace/project"])

    assert result.total_models == 2
    assert result.skipped_models == 1
    assert result.total_model_versions == 2
    assert result.skipped_model_versions == 1


def test_model_registry_export_manager_routes_parameter_batches():
    """Test model export manager routes batches to scoped writers."""
    mock_exporter = Mock()
    mock_exporter.list_models.return_value = ["MODEL-1"]
    mock_exporter.list_model_versions.return_value = []
    mock_exporter.download_model_parameters.return_value = [
        pa.record_batch(
            {
                "project_id": ["workspace/project"],
                "run_id": ["MODEL-1"],
                "attribute_path": ["sys/name"],
                "attribute_type": ["string"],
            }
        )
    ]
    mock_exporter.download_model_metrics.return_value = []
    mock_exporter.download_model_series.return_value = []
    mock_exporter.download_model_files.return_value = []
    mock_exporter.download_model_version_parameters.return_value = []
    mock_exporter.download_model_version_metrics.return_value = []
    mock_exporter.download_model_version_series.return_value = []
    mock_exporter.download_model_version_files.return_value = []

    mock_reader = Mock(spec=ParquetReader)
    mock_reader.check_run_exists.return_value = False
    mock_writer = Mock(spec=ParquetWriter)
    writer_context = Mock()
    mock_writer.run_writer.return_value = writer_context
    mock_error_reporter = Mock(spec=ErrorReporter)

    manager = ModelRegistryExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/tmp/files"),
        progress_listener_factory=NoopProgressListenerFactory(),
    )

    manager.run(
        project_ids=["workspace/project"],
        export_classes={"parameters"},
    )

    mock_writer.run_writer.assert_called_once_with(
        "workspace/project",
        "MODEL-1",
        entity_scope="models",
    )
    writer_context.save.assert_called_once()
    writer_context.finish_run.assert_called_once()
