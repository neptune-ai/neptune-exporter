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
import queue
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa

from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter


def test_list_models(monkeypatch):
    """Test listing models from Neptune2 project."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    mock_project = Mock()
    mock_project.fetch_models_table.return_value.to_pandas.return_value = pd.DataFrame(
        {"sys/id": ["MODEL-1", "MODEL-2"]}
    )
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_project)
    mock_context.__exit__ = Mock(return_value=None)

    monkeypatch.setattr(
        "neptune_exporter.exporters.neptune2.neptune.init_project",
        Mock(return_value=mock_context),
    )

    model_ids = exporter.list_models(
        "workspace/project",
        query='`sys/id`:string = "MODEL-1"',
        include_trashed=True,
    )

    exporter.close()

    assert model_ids == ["MODEL-1", "MODEL-2"]
    mock_project.fetch_models_table.assert_called_once()


def test_list_model_versions(monkeypatch):
    """Test listing model versions from Neptune2 model."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    mock_model = Mock()
    mock_model.fetch_model_versions_table.return_value.to_pandas.return_value = (
        pd.DataFrame({"sys/id": ["MODEL-1-1", "MODEL-1-2"]})
    )
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_model)
    mock_context.__exit__ = Mock(return_value=None)

    monkeypatch.setattr(
        exporter, "_init_model_container", Mock(return_value=mock_context)
    )

    model_version_ids = exporter.list_model_versions(
        "workspace/project",
        "MODEL-1",
        query=None,
    )

    exporter.close()

    assert model_version_ids == ["MODEL-1-1", "MODEL-1-2"]
    mock_model.fetch_model_versions_table.assert_called_once()


def test_download_model_parameters_delegates_to_container_helper(monkeypatch):
    """Test model parameter download delegates to generic container helper."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    expected_batch = pa.record_batch(
        {
            "project_id": ["workspace/project"],
            "run_id": ["MODEL-1"],
            "attribute_path": ["sys/name"],
            "attribute_type": ["string"],
        }
    )
    mock_delegate = Mock(return_value=iter([expected_batch]))
    monkeypatch.setattr(exporter, "_download_container_parameters", mock_delegate)

    actual_batches = list(
        exporter.download_model_parameters(
            project_id="workspace/project",
            model_ids=["MODEL-1"],
            attributes=None,
        )
    )

    exporter.close()

    assert actual_batches == [expected_batch]
    assert mock_delegate.call_count == 1


def test_download_model_version_files_delegates_to_container_helper(monkeypatch):
    """Test model version file download delegates to generic container helper."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    expected_batch = pa.record_batch(
        {
            "project_id": ["workspace/project"],
            "run_id": ["MODEL-1-1"],
            "attribute_path": ["artifacts/weights"],
            "attribute_type": ["file"],
        }
    )
    mock_delegate = Mock(return_value=iter([expected_batch]))
    monkeypatch.setattr(exporter, "_download_container_files", mock_delegate)

    actual_batches = list(
        exporter.download_model_version_files(
            project_id="workspace/project",
            model_version_ids=["MODEL-1-1"],
            attributes=None,
            destination=Path("/tmp/model_files"),
        )
    )

    exporter.close()

    assert actual_batches == [expected_batch]
    assert mock_delegate.call_count == 1


def test_container_attribute_workers_process_all_containers(monkeypatch):
    """Test generic container scheduler dispatches all listed containers."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter), max_workers=4)

    monkeypatch.setattr(
        exporter,
        "_list_container_attributes_by_id",
        lambda **_: [("metrics/loss", "float_series")],
    )

    totals: list[tuple[str, int]] = []
    completed: list[str] = []

    def worker(_project_id, container_id, attribute_queue, on_attribute_done):
        processed = 0
        while True:
            try:
                attribute_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if on_attribute_done is not None:
                on_attribute_done(container_id)
        return [f"{container_id}:{processed}"]

    results = list(
        exporter._container_attribute_workers(
            project_id="workspace/project",
            container_ids=["MODEL-1", "MODEL-2"],
            attributes=None,
            allowed_types=("float_series",),
            init_container=Mock(),
            worker_fn=worker,
            on_attribute_total=lambda rid, total: totals.append((rid, total)),
            on_attribute_done=lambda rid: completed.append(rid),
        )
    )

    exporter.close()

    assert sorted(results) == [
        ("MODEL-1", ["MODEL-1:1"]),
        ("MODEL-2", ["MODEL-2:1"]),
    ]
    assert sorted(totals) == [("MODEL-1", 1), ("MODEL-2", 1)]
    assert sorted(completed) == ["MODEL-1", "MODEL-2"]


def test_container_file_worker_paths_are_relative_to_project_files_root(tmp_path):
    """Model file paths should include entity scope relative to project files root."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    attribute_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
    attribute_queue.put(("artifacts/model.bin", "file"))

    mock_file_attribute = Mock()

    def write_downloaded_file(path, progress_bar=False):  # noqa: ARG001
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("artifact")

    mock_file_attribute.download.side_effect = write_downloaded_file

    mock_container = Mock()
    mock_container.get_attribute.return_value = mock_file_attribute
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_container)
    mock_context.__exit__ = Mock(return_value=None)

    destination = tmp_path / "workspace_project" / "models"
    destination.mkdir(parents=True, exist_ok=True)

    result = exporter._container_file_worker(
        project_id="workspace/project",
        container_id="MODEL-1",
        attribute_queue=attribute_queue,
        destination=destination,
        init_container=Mock(return_value=mock_context),
    )

    exporter.close()

    assert len(result) == 1
    assert result[0]["file_value"]["path"] == "models/MODEL-1/artifacts/model.bin"


def test_container_file_worker_reports_invalid_file_series_step(tmp_path):
    """Invalid file-series step names should be treated as attribute errors."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))
    exporter._handle_attribute_exception = Mock()

    attribute_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
    attribute_queue.put(("artifacts/series", "file_series"))

    mock_file_series_attribute = Mock()

    def write_invalid_series_file(path, progress_bar=False):  # noqa: ARG001
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "not-a-number").write_text("artifact")

    mock_file_series_attribute.download.side_effect = write_invalid_series_file

    mock_container = Mock()
    mock_container.get_attribute.return_value = mock_file_series_attribute
    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_container)
    mock_context.__exit__ = Mock(return_value=None)

    destination = tmp_path / "workspace_project" / "models"
    destination.mkdir(parents=True, exist_ok=True)

    result = exporter._container_file_worker(
        project_id="workspace/project",
        container_id="MODEL-1",
        attribute_queue=attribute_queue,
        destination=destination,
        init_container=Mock(return_value=mock_context),
    )

    exporter.close()

    assert result == []
    exporter._handle_attribute_exception.assert_called_once()
