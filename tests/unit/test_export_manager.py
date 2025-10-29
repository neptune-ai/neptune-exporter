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

from unittest.mock import Mock
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

from neptune_exporter.export_manager import ExportManager
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter import model


def test_export_manager_resume_skips_existing_runs():
    """Test that resume=True skips already-exported runs."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs initially
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]

    # Mock reader to return existing runs (some overlap)
    mock_reader.get_unique_run_ids.return_value = {"RUN-123", "RUN-456"}

    # Create export manager with resume=True
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        files_destination=Path("/fake/files"),
        resume=True,
    )

    # Mock the project writer context
    mock_writer_context = Mock()
    mock_writer.project_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.project_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []

    # Run export with project that has existing runs
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Verify reader was called to check existing runs
    mock_reader.get_unique_run_ids.assert_called_once()

    # Verify writer.delete_project_data was NOT called (resume mode)
    mock_writer.delete_project_data.assert_not_called()

    # Should return 3 since it returns total runs found, not processed
    assert result == 3


def test_export_manager_overwrite_deletes_existing_data():
    """Test that resume=False deletes existing project data."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456"]

    # Mock reader to return no existing runs (after deletion)
    mock_reader.get_unique_run_ids.return_value = set()

    # Create export manager with resume=False (overwrite mode)
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        files_destination=Path("/fake/files"),
        resume=False,
    )

    # Mock the project writer context
    mock_writer_context = Mock()
    mock_writer.project_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.project_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []

    # Run export
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Verify delete_project_data was called (overwrite mode)
    mock_writer.delete_project_data.assert_called_once_with("test-project")

    # Should return 2 since both runs were processed
    assert result == 2


def test_export_manager_with_no_existing_data():
    """Test export manager behavior when no existing data is present."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456"]

    # Mock reader to return no existing runs
    mock_reader.get_unique_run_ids.return_value = set()

    # Create export manager
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        files_destination=Path("/fake/files"),
        resume=True,
    )

    # Mock the project writer context
    mock_writer_context = Mock()
    mock_writer.project_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.project_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []

    # Run export
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Verify delete_project_data was NOT called (resume mode with no existing data)
    mock_writer.delete_project_data.assert_not_called()

    # Should return 2 since both runs were processed
    assert result == 2


def test_export_manager_with_partial_existing_data():
    """Test export manager with some existing runs and some new runs."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]

    # Mock reader to return some existing runs
    mock_reader.get_unique_run_ids.return_value = {"RUN-123"}

    # Create export manager with resume=True
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        files_destination=Path("/fake/files"),
        resume=True,
    )

    # Mock the project writer context
    mock_writer_context = Mock()
    mock_writer.project_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.project_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []

    # Run export with multiple runs (one existing, two new)
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Verify delete_project_data was NOT called (resume mode)
    mock_writer.delete_project_data.assert_not_called()

    # Should return 3 since it returns total runs found, not processed
    assert result == 3


def test_export_manager_integration_with_real_files():
    """Test export manager with real parquet files to verify resume behavior."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create real storage and reader
        storage = ParquetWriter(base_path=temp_path)
        reader = ParquetReader(base_path=temp_path)

        # Create some existing data
        existing_data = pa.Table.from_pydict(
            {
                "project_id": ["test-project", "test-project"],
                "run_id": ["RUN-123", "RUN-456"],
                "attribute_path": ["param1", "param2"],
                "attribute_type": ["string", "string"],
                "step": [None, None],
                "timestamp": [None, None],
                "int_value": [None, None],
                "float_value": [None, None],
                "string_value": ["value1", "value2"],
                "bool_value": [None, None],
                "datetime_value": [None, None],
                "string_set_value": [None, None],
                "file_value": [None, None],
                "histogram_value": [None, None],
            },
            schema=model.SCHEMA,
        )

        # Write existing data
        project_dir = temp_path / "test-project"
        project_dir.mkdir()
        pq.write_table(existing_data, project_dir / "part_0.parquet")

        # Create mock exporter
        mock_exporter = Mock()
        mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]
        mock_exporter.download_parameters.return_value = []
        mock_exporter.download_metrics.return_value = []
        mock_exporter.download_series.return_value = []
        mock_exporter.download_files.return_value = []

        # Test resume mode
        export_manager = ExportManager(
            exporter=mock_exporter,
            reader=reader,
            writer=storage,
            files_destination=Path("/fake/files"),
            resume=True,
        )

        # Run export
        result = export_manager.run(
            project_ids=["test-project"],
            runs=None,
            attributes=None,
            export_classes={"parameters", "metrics", "series", "files"},
        )

        # Verify existing data is still there
        assert (project_dir / "part_0.parquet").exists()

        # Should return 3 since it returns total runs found, not processed
        assert result == 3
