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
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tempfile

from neptune_exporter.export_manager import ExportManager
from neptune_exporter.exporters.error_reporter import ErrorReporter, ErrorSummary
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter import model
from neptune_exporter.utils import sanitize_path_part


def test_export_manager_skips_existing_runs():
    """Test that export manager skips already-exported runs."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)
    mock_error_reporter = Mock(spec=ErrorReporter)
    mock_error_reporter.get_summary.return_value = ErrorSummary(exception_count=0)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs initially
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]

    # Mock reader to return True for existing runs
    def mock_check_run_exists(project_id, run_id):
        return run_id in {"RUN-123", "RUN-456"}

    mock_reader.check_run_exists.side_effect = mock_check_run_exists

    # Create export manager
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/fake/files"),
    )

    # Mock the run writer context
    mock_writer_context = Mock()
    mock_writer.run_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.run_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []
    mock_exporter.get_exception_infos.return_value = []

    # Run export with project that has existing runs
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Verify reader was called to check existing runs (3 times, once per run)
    assert mock_reader.check_run_exists.call_count == 3

    # Should return 3 since it returns total runs found, not processed
    assert result.total_runs == 3
    assert result.skipped_runs == 2


def test_export_manager_with_no_existing_data():
    """Test export manager behavior when no existing data is present."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)
    mock_error_reporter = Mock(spec=ErrorReporter)
    mock_error_reporter.get_summary.return_value = ErrorSummary(exception_count=0)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456"]

    # Mock reader to return False for all runs (no existing runs)
    mock_reader.check_run_exists.return_value = False

    # Create export manager
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/fake/files"),
    )

    # Mock the run writer context
    mock_writer_context = Mock()
    mock_writer.run_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.run_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []
    mock_exporter.get_exception_infos.return_value = []
    # Run export
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Should return 2 since both runs were processed
    assert result.total_runs == 2
    assert result.skipped_runs == 0


def test_export_manager_with_partial_existing_data():
    """Test export manager with some existing runs and some new runs."""
    # Create mocks
    mock_exporter = Mock()
    mock_reader = Mock(spec=ParquetReader)
    mock_writer = Mock(spec=ParquetWriter)
    mock_error_reporter = Mock(spec=ErrorReporter)
    mock_error_reporter.get_summary.return_value = ErrorSummary(exception_count=0)

    # Mock writer base_path
    mock_writer.base_path = Path("/fake/path")

    # Mock exporter to return some runs
    mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]

    # Mock reader to return True only for RUN-123
    def mock_check_run_exists(project_id, run_id):
        return run_id == "RUN-123"

    mock_reader.check_run_exists.side_effect = mock_check_run_exists

    # Create export manager
    export_manager = ExportManager(
        exporter=mock_exporter,
        reader=mock_reader,
        writer=mock_writer,
        error_reporter=mock_error_reporter,
        files_destination=Path("/fake/files"),
    )

    # Mock the run writer context
    mock_writer_context = Mock()
    mock_writer.run_writer.return_value.__enter__ = Mock(
        return_value=mock_writer_context
    )
    mock_writer.run_writer.return_value.__exit__ = Mock(return_value=None)

    # Mock exporter methods to return empty generators
    mock_exporter.download_parameters.return_value = []
    mock_exporter.download_metrics.return_value = []
    mock_exporter.download_series.return_value = []
    mock_exporter.download_files.return_value = []
    mock_exporter.get_exception_infos.return_value = []
    # Run export with multiple runs (one existing, two new)
    result = export_manager.run(
        project_ids=["test-project"],
        runs=None,
        attributes=None,
        export_classes={"parameters", "metrics", "series", "files"},
    )

    # Should return 3 since it returns total runs found, not processed
    assert result.total_runs == 3
    assert result.skipped_runs == 1


def test_export_manager_integration_with_real_files():
    """Test export manager with real parquet files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create real storage and reader
        storage = ParquetWriter(base_path=temp_path)
        reader = ParquetReader(base_path=temp_path)

        # Create error reporter
        error_reporter = ErrorReporter(path=temp_path / "errors.jsonl")

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

        # Write existing data (complete runs with part_0)
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()
        # Create complete runs (with part_0)
        run_123_mask = pc.equal(existing_data["run_id"], "RUN-123")
        run_456_mask = pc.equal(existing_data["run_id"], "RUN-456")
        pq.write_table(
            existing_data.filter(run_123_mask),
            project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet",
        )
        pq.write_table(
            existing_data.filter(run_456_mask),
            project_dir / f"{sanitize_path_part('RUN-456')}_part_0.parquet",
        )

        # Create mock exporter
        mock_exporter = Mock()
        mock_exporter.list_runs.return_value = ["RUN-123", "RUN-456", "RUN-789"]
        mock_exporter.download_parameters.return_value = []
        mock_exporter.download_metrics.return_value = []
        mock_exporter.download_series.return_value = []
        mock_exporter.download_files.return_value = []
        mock_exporter.get_exception_infos.return_value = []
        # Create export manager
        export_manager = ExportManager(
            exporter=mock_exporter,
            reader=reader,
            writer=storage,
            error_reporter=error_reporter,
            files_destination=Path("/fake/files"),
        )

        # Run export
        result = export_manager.run(
            project_ids=["test-project"],
            runs=None,
            attributes=None,
            export_classes={"parameters", "metrics", "series", "files"},
        )

        # Verify existing data is still there
        assert (
            project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet"
        ).exists()
        assert (
            project_dir / f"{sanitize_path_part('RUN-456')}_part_0.parquet"
        ).exists()

        # Should return 3 since it returns total runs found, not processed
        assert result.total_runs == 3
        assert result.skipped_runs == 2
