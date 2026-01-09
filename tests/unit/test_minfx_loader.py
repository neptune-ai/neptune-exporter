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

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_neptune():
    """Create a mock neptune module."""
    # Need to mock both the module import and where it's used
    mock_neptune_module = MagicMock()
    mock_neptune_module.init_run = MagicMock()
    mock_neptune_module.init_project = MagicMock()
    mock_neptune_module.ANONYMOUS_API_TOKEN = "test-anonymous-token"

    with patch.dict("sys.modules", {"minfx.neptune_v2": mock_neptune_module}):
        with patch(
            "neptune_exporter.loaders.minfx_loader.neptune", mock_neptune_module
        ):
            yield mock_neptune_module


@pytest.fixture
def mock_file():
    """Create a mock File class (neptune.types.File)."""
    with patch("neptune_exporter.loaders.minfx_loader.File") as mock_file_class:
        yield mock_file_class


@pytest.fixture
def loader(mock_neptune, mock_file):
    """Create a loader instance with mocked minfx."""
    from neptune_exporter.loaders.minfx_loader import MinfxLoader

    return MinfxLoader(
        project="test-workspace/test-project",
        api_token="test-api-token",
        name_prefix="test-prefix",
    )


def test_init(mock_neptune, mock_file):
    """Test MinfxLoader initialization."""
    from neptune_exporter.loaders.minfx_loader import MinfxLoader

    loader = MinfxLoader(
        project="test-workspace/test-project",
        api_token="test-api-token",
        name_prefix="test-prefix",
    )

    assert loader._project == "test-workspace/test-project"
    assert loader._api_token == "test-api-token"
    assert loader._name_prefix == "test-prefix"
    assert loader._active_run is None


def test_init_without_api_token(mock_neptune, mock_file):
    """Test MinfxLoader initialization without API token."""
    from neptune_exporter.loaders.minfx_loader import MinfxLoader

    loader = MinfxLoader(
        project="test-workspace/test-project",
        name_prefix="test-prefix",
    )

    assert loader._project == "test-workspace/test-project"
    assert loader._api_token is None
    assert loader._name_prefix == "test-prefix"


def test_get_run_name(loader):
    """Test run name generation with prefix."""
    # Arrange
    expected_name = "test-prefix_run-123"

    # Act
    actual_name = loader._get_run_name("run-123")

    # Assert
    assert actual_name == expected_name


def test_get_run_name_without_prefix(mock_neptune, mock_file):
    """Test run name generation without prefix."""
    from neptune_exporter.loaders.minfx_loader import MinfxLoader

    # Arrange
    loader_no_prefix = MinfxLoader(
        project="test-workspace/test-project",
    )
    expected_name = "run-456"

    # Act
    actual_name = loader_no_prefix._get_run_name("run-456")

    # Assert
    assert actual_name == expected_name


def test_convert_step(loader):
    """Test step conversion from Decimal to float."""
    # Test normal conversion
    assert loader._convert_step(Decimal("1.5")) == 1.5
    assert loader._convert_step(Decimal("0.0")) == 0.0
    assert loader._convert_step(Decimal("100.123456")) == 100.123456

    # Test None step - returns 0.0 as default
    assert loader._convert_step(None) == 0.0

    # Test NaN-like value - returns 0.0 as default
    assert loader._convert_step(float("nan")) == 0.0


def test_get_value_for_tuple_int(loader):
    """Test value extraction for int type from tuple."""
    # Arrange - create DataFrame and get tuple via itertuples
    df = pd.DataFrame(
        {
            "int_value": [42],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "int")

    # Assert
    assert actual_value == 42
    assert isinstance(actual_value, int)


def test_get_value_for_tuple_float(loader):
    """Test value extraction for float type from tuple."""
    # Arrange
    df = pd.DataFrame(
        {
            "int_value": [None],
            "float_value": [3.14],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "float")

    # Assert
    assert actual_value == 3.14
    assert isinstance(actual_value, float)


def test_get_value_for_tuple_string(loader):
    """Test value extraction for string type from tuple."""
    # Arrange
    df = pd.DataFrame(
        {
            "int_value": [None],
            "float_value": [None],
            "string_value": ["test_value"],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "string")

    # Assert
    assert actual_value == "test_value"


def test_get_value_for_tuple_bool(loader):
    """Test value extraction for bool type from tuple."""
    # Arrange
    df = pd.DataFrame(
        {
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [True],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "bool")

    # Assert
    assert actual_value is True


def test_get_value_for_tuple_datetime(loader):
    """Test value extraction for datetime type from tuple."""
    # Arrange
    test_datetime = pd.Timestamp("2023-01-01 12:00:00")
    df = pd.DataFrame(
        {
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [test_datetime],
            "string_set_value": [None],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "datetime")

    # Assert
    assert actual_value == test_datetime


def test_get_value_for_tuple_string_set(loader):
    """Test value extraction for string_set type (list to StringSet conversion) from tuple."""
    from neptune.types import StringSet

    # Arrange
    df = pd.DataFrame(
        {
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [["value1", "value2", "value3"]],
            "file_value": [None],
        }
    )
    row = next(df.itertuples(index=False))

    # Act
    actual_value = loader._get_value_for_tuple(row, "string_set")

    # Assert - returns Neptune's StringSet type for API compatibility
    assert isinstance(actual_value, StringSet)
    assert actual_value.values == {"value1", "value2", "value3"}


def test_create_experiment(loader):
    """Test creating an experiment (returns experiment name as ID)."""
    # Act
    actual_id = loader.create_experiment("test-project-id", "test-experiment")

    # Assert
    assert actual_id == "test-experiment"


def test_find_run_not_found(loader):
    """Test finding a run returns None (not implemented)."""
    # Act
    actual_result = loader.find_run("test-project", "test-run", "test-experiment")

    # Assert
    assert actual_result is None


def test_create_run(mock_neptune, loader):
    """Test creating a Minfx run."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(fetch=MagicMock(return_value="neptune-run-123"))
    )
    mock_run.__setitem__ = MagicMock()
    mock_neptune.init_run.return_value = mock_run

    # Act
    actual_run_id = loader.create_run("test-project", "run-name", "experiment-id")

    # Assert
    assert actual_run_id == "neptune-run-123"
    mock_neptune.init_run.assert_called_once_with(
        project="test-workspace/test-project",
        api_token="test-api-token",
        name="experiment-id",
        source_files=[],  # Disable automatic source code tracking
        git_ref=False,  # Disable git tracking completely
        capture_hardware_metrics=False,  # Disable hardware monitoring
        capture_stdout=False,  # Disable stdout capture
        capture_stderr=False,  # Disable stderr capture
        capture_traceback=False,  # Disable traceback
    )
    # Verify import/original_run_id and import/original_project_id were set
    setitem_calls = mock_run.__setitem__.call_args_list
    paths_set = {call[0][0] for call in setitem_calls}
    assert "import/original_run_id" in paths_set
    assert "import/original_project_id" in paths_set


def test_create_run_with_parent(mock_neptune, loader):
    """Test creating a Minfx run with fork parent."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(fetch=MagicMock(return_value="neptune-run-child"))
    )
    mock_run.__setitem__ = MagicMock()
    mock_neptune.init_run.return_value = mock_run

    # Act
    actual_run_id = loader.create_run(
        "test-project", "child-run", "experiment-id", "parent-run-id", fork_step=100.0
    )

    # Assert
    assert actual_run_id == "neptune-run-child"

    # Verify fork metadata was set (using import/ namespace)
    setitem_calls = mock_run.__setitem__.call_args_list
    paths_set = [call[0][0] for call in setitem_calls]
    assert "import/forking/parent" in paths_set
    assert "import/forking/step" in paths_set


def test_upload_parameters(mock_neptune, loader):
    """Test parameter upload to Minfx."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__setitem__ = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": [
                "test/param1",
                "test/param2",
                "test/param3",
                "test/param4",
            ],
            "attribute_type": ["string", "float", "int", "bool"],
            "string_value": ["test_value", None, None, None],
            "float_value": [None, 3.14, None, None],
            "int_value": [None, None, 42, None],
            "bool_value": [None, None, None, True],
            "datetime_value": [None, None, None, None],
            "string_set_value": [None, None, None, None],
            "file_value": [None, None, None, None],
        }
    )

    # Act
    loader.upload_parameters(test_data, "RUN-123")

    # Assert
    setitem_calls = mock_run.__setitem__.call_args_list
    paths_set = {call[0][0] for call in setitem_calls}

    assert "test/param1" in paths_set
    assert "test/param2" in paths_set
    assert "test/param3" in paths_set
    assert "test/param4" in paths_set


def test_upload_parameters_skips_sys_attributes(mock_neptune, loader):
    """Test that sys/ attributes are skipped during parameter upload."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__setitem__ = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["sys/name", "sys/id", "test/param"],
            "attribute_type": ["string", "string", "string"],
            "string_value": ["run-name", "run-id", "test_value"],
            "float_value": [None, None, None],
            "int_value": [None, None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
            "file_value": [None, None, None],
        }
    )

    # Act
    loader.upload_parameters(test_data, "RUN-123")

    # Assert
    setitem_calls = mock_run.__setitem__.call_args_list
    paths_set = {call[0][0] for call in setitem_calls}

    # sys/ attributes should be skipped
    assert "sys/name" not in paths_set
    assert "sys/id" not in paths_set
    # Regular attributes should be uploaded
    assert "test/param" in paths_set


def test_upload_parameters_string_set(mock_neptune, loader):
    """Test parameter upload with string_set type."""
    from neptune.types import StringSet

    # Arrange
    mock_run = MagicMock()
    mock_run.__setitem__ = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/tags"],
            "attribute_type": ["string_set"],
            "string_value": [None],
            "float_value": [None],
            "int_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [["tag1", "tag2", "tag3"]],
            "file_value": [None],
        }
    )

    # Act
    loader.upload_parameters(test_data, "RUN-123")

    # Assert
    setitem_calls = mock_run.__setitem__.call_args_list
    # Find the call for test/tags
    for call in setitem_calls:
        if call[0][0] == "test/tags":
            # Value should be Neptune's StringSet type for API compatibility
            assert isinstance(call[0][1], StringSet)
            assert call[0][1].values == {"tag1", "tag2", "tag3"}
            break
    else:
        pytest.fail("test/tags attribute not uploaded")


def test_upload_metrics(mock_neptune, loader):
    """Test metrics upload to Minfx using batch FloatSeries."""
    # Arrange
    mock_run = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/metric1", "test/metric1", "test/metric2"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [Decimal("1.0"), Decimal("2.0"), Decimal("1.0")],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-01"),
            ],
            "float_value": [0.5, 0.7, 0.3],
            "string_value": [None, None, None],
        }
    )

    # Act
    loader.upload_metrics(test_data, "RUN-123")

    # Assert - verify __setitem__ was called for each unique metric (batch upload)
    # 2 unique metrics: test/metric1 (2 points) and test/metric2 (1 point)
    assert mock_run.__setitem__.call_count == 2


def test_upload_string_series(mock_neptune, loader):
    """Test string series upload to Minfx using batch StringSeries."""
    # Arrange
    mock_run = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/log", "test/log"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
            ],
            "float_value": [None, None],
            "string_value": ["log entry 1", "log entry 2"],
        }
    )

    # Act
    loader.upload_series(test_data, "RUN-123")

    # Assert - verify __setitem__ was called for each unique series (batch upload)
    # 1 unique series: test/log (2 points)
    assert mock_run.__setitem__.call_count == 1


def test_upload_files(mock_neptune, mock_file, loader, temp_dir):
    """Test file upload to Minfx."""
    # Arrange
    mock_run = MagicMock()
    mock_upload = MagicMock()
    mock_run.__getitem__ = MagicMock(return_value=MagicMock(upload=mock_upload))
    loader._active_run = mock_run

    # Create a test file
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("test content")

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file1"],
            "attribute_type": ["file"],
            "file_value": [{"path": "test_file.txt"}],
        }
    )

    # Act
    loader.upload_files(test_data, "RUN-123", temp_dir)

    # Assert - verify File was created and upload was called with wait=False
    mock_upload.assert_called_once()
    # Verify File class was used to create the file object
    mock_file.assert_called_once()
    # Verify wait=False was passed (uploads are batched, sync() ensures completion)
    assert mock_upload.call_args[1].get("wait") is False


def test_upload_files_missing_file(mock_neptune, loader, temp_dir, caplog):
    """Test file upload with missing file logs warning."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__setitem__ = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/missing_file"],
            "attribute_type": ["file"],
            "file_value": [{"path": "nonexistent.txt"}],
        }
    )

    # Act
    loader.upload_files(test_data, "RUN-123", temp_dir)

    # Assert - no file should be uploaded
    setitem_calls = mock_run.__setitem__.call_args_list
    paths_set = {call[0][0] for call in setitem_calls}
    assert "test/missing_file" not in paths_set


def test_upload_run_data(mock_neptune, loader, temp_dir):
    """Test uploading complete run data."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(
            fetch=MagicMock(return_value="neptune-run-id"),
            append=MagicMock(),
        )
    )
    mock_run.__setitem__ = MagicMock()
    mock_run.stop = MagicMock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "project_id": ["test-project"] * 3,
            "run_id": ["RUN-123"] * 3,
            "attribute_path": ["test/param", "test/metric", "test/log"],
            "attribute_type": ["string", "float_series", "string_series"],
            "step": [None, Decimal("1.0"), Decimal("1.0")],
            "timestamp": [None, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "int_value": [None, None, None],
            "float_value": [None, 0.5, None],
            "string_value": ["test_value", None, "log entry"],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
            "file_value": [None, None, None],
            "histogram_value": [None, None, None],
        }
    )

    # Convert to PyArrow table with proper schema
    from neptune_exporter import model

    table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)

    # upload_run_data expects a generator of tables
    def table_generator():
        yield table

    # Act
    loader.upload_run_data(
        table_generator(), "neptune-run-id", temp_dir, step_multiplier=1
    )

    # Assert - run should be stopped at the end
    mock_run.stop.assert_called_once()


def test_upload_run_data_histogram_series_warning(mock_neptune, loader, temp_dir):
    """Test that histogram_series logs a warning."""
    # Arrange
    mock_run = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(
            fetch=MagicMock(return_value="neptune-run-id"),
            append=MagicMock(),
            upload=MagicMock(),
        )
    )
    mock_run.__setitem__ = MagicMock()
    mock_run.stop = MagicMock()
    loader._active_run = mock_run

    # Include both string_series (so the method doesn't return early) and histogram_series
    test_data = pd.DataFrame(
        {
            "project_id": ["test-project", "test-project"],
            "run_id": ["RUN-123", "RUN-123"],
            "attribute_path": ["test/log", "test/histogram"],
            "attribute_type": ["string_series", "histogram_series"],
            "step": [Decimal("1.0"), Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "int_value": [None, None],
            "float_value": [None, None],
            "string_value": ["log entry", None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
            "file_value": [None, None],
            "histogram_value": [
                None,
                {"type": "histogram", "edges": [0.0, 1.0], "values": [10]},
            ],
        }
    )

    from neptune_exporter import model

    table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)

    def table_generator():
        yield table

    # Mock the logger to capture warning calls
    with patch.object(loader, "_logger") as mock_logger:
        # Act
        loader.upload_run_data(
            table_generator(), "neptune-run-id", temp_dir, step_multiplier=1
        )

        # Assert - warning about histogram_series should be logged
        warning_calls = mock_logger.warning.call_args_list
        histogram_warning_found = any(
            "histogram_series" in str(call) for call in warning_calls
        )
        assert histogram_warning_found, (
            f"Expected histogram_series warning, got: {warning_calls}"
        )


def test_upload_file_set_with_files_zip(mock_neptune, loader, temp_dir):
    """Test uploading file_set that contains files.zip (Neptune export format)."""
    import zipfile

    # Arrange
    mock_run = MagicMock()
    mock_upload_files = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(upload_files=mock_upload_files)
    )
    loader._active_run = mock_run

    # Create a file_set directory with files.zip (Neptune export format)
    file_set_dir = temp_dir / "source_code" / "files"
    file_set_dir.mkdir(parents=True)

    # Create a zip file with some content
    files_zip = file_set_dir / "files.zip"
    with zipfile.ZipFile(files_zip, "w") as zf:
        zf.writestr("test_file.py", "print('hello')")
        zf.writestr("another_file.txt", "test content")

    # Act
    loader._upload_file_set("source_code/files", file_set_dir)

    # Assert - upload_files should be called with temp directory (not the original)
    mock_upload_files.assert_called_once()
    call_path = mock_upload_files.call_args[0][0]
    # The path should NOT be the original file_set_dir (which contains files.zip)
    assert str(file_set_dir) not in call_path


def test_upload_file_set_without_files_zip(mock_neptune, loader, temp_dir):
    """Test uploading file_set that doesn't contain files.zip (direct files)."""
    # Arrange
    mock_run = MagicMock()
    mock_upload_files = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(upload_files=mock_upload_files)
    )
    loader._active_run = mock_run

    # Create a file_set directory with regular files (not Neptune export format)
    file_set_dir = temp_dir / "custom_files"
    file_set_dir.mkdir(parents=True)
    (file_set_dir / "file1.txt").write_text("content1")
    (file_set_dir / "file2.txt").write_text("content2")

    # Act
    loader._upload_file_set("custom_files", file_set_dir)

    # Assert - upload_files should be called with the original directory and wait=True
    mock_upload_files.assert_called_once_with(str(file_set_dir), wait=True)


def test_upload_files_file_set_type(mock_neptune, loader, temp_dir):
    """Test that file_set types use _upload_file_set method."""
    import zipfile

    # Arrange
    mock_run = MagicMock()
    mock_upload_files = MagicMock()
    mock_run.__getitem__ = MagicMock(
        return_value=MagicMock(upload_files=mock_upload_files)
    )
    loader._active_run = mock_run

    # Create a file_set directory with files.zip
    file_set_dir = temp_dir / "source_code" / "files"
    file_set_dir.mkdir(parents=True)
    files_zip = file_set_dir / "files.zip"
    with zipfile.ZipFile(files_zip, "w") as zf:
        zf.writestr("script.py", "print('test')")

    test_data = pd.DataFrame(
        {
            "attribute_path": ["source_code/files"],
            "attribute_type": ["file_set"],
            "file_value": [{"path": "source_code/files"}],
        }
    )

    # Act
    loader.upload_files(test_data, "RUN-123", temp_dir)

    # Assert - upload_files should be called (via _upload_file_set)
    mock_upload_files.assert_called_once()
    # The path should be a temp directory (extracted contents), not the original
    call_path = mock_upload_files.call_args[0][0]
    assert "files.zip" not in call_path
