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

import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from neptune_exporter.loaders.mlflow_loader import MLflowLoader


def test_init():
    """Test MLflowLoader initialization."""
    loader = MLflowLoader(
        tracking_uri="http://localhost:5000",
        name_prefix="test-prefix",
    )

    assert loader.tracking_uri == "http://localhost:5000"
    assert loader.name_prefix == "test-prefix"


def test_sanitize_attribute_name():
    """Test attribute name sanitization."""
    loader = MLflowLoader()

    # Test normal name
    assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    # Test name with invalid characters
    assert (
        loader._sanitize_attribute_name("invalid@name#with$chars")
        == "invalid_name_with_chars"
    )

    # Test long name
    long_name = "a" * 300
    sanitized = loader._sanitize_attribute_name(long_name)
    assert len(sanitized) == 250
    assert sanitized == "a" * 250


def test_convert_step_to_int():
    """Test step conversion from decimal to int."""
    loader = MLflowLoader()

    # Test normal conversion
    assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    # Test None step
    assert loader._convert_step_to_int(None, 1000) == 0

    # Test zero step
    assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


def test_get_experiment_name():
    """Test experiment name generation."""
    loader = MLflowLoader(name_prefix="test-prefix")

    # Test with prefix
    assert (
        loader._get_experiment_name("my-project", "experiment")
        == "test-prefix/my-project/experiment"
    )

    # Test without prefix
    loader_no_prefix = MLflowLoader()
    assert (
        loader_no_prefix._get_experiment_name("my-project", "experiment")
        == "my-project/experiment"
    )


@patch("mlflow.get_experiment_by_name", spec=mlflow.get_experiment_by_name)
@patch("mlflow.create_experiment", spec=mlflow.create_experiment)
def test_create_experiment_new(mock_create, mock_get):
    """Test creating a new experiment."""
    mock_get.return_value = None
    mock_create.return_value = "exp-123"

    loader = MLflowLoader()
    experiment_id = loader.create_experiment("test-project", "experiment")

    assert experiment_id == "exp-123"
    mock_get.assert_called_once()
    mock_create.assert_called_once()


@patch("mlflow.get_experiment_by_name", spec=mlflow.get_experiment_by_name)
@patch("mlflow.create_experiment", spec=mlflow.create_experiment)
def test_create_experiment_existing(mock_create, mock_get):
    """Test using an existing experiment."""
    mock_experiment = Mock()
    mock_experiment.experiment_id = "exp-456"
    mock_get.return_value = mock_experiment

    loader = MLflowLoader()
    experiment_id = loader.create_experiment("test-project", "experiment")

    assert experiment_id == "exp-456"
    mock_get.assert_called_once()
    mock_create.assert_not_called()


@patch("mlflow.start_run", spec=mlflow.start_run)
def test_create_run(mock_start_run):
    """Test creating a run."""
    mock_run = Mock()
    mock_run.info.run_id = "run-123"
    mock_start_run.return_value.__enter__.return_value = mock_run

    loader = MLflowLoader()
    run_id = loader.create_run("test-project", "run-name", "exp-123")

    assert run_id == "run-123"
    mock_start_run.assert_called_once()


@patch("mlflow.start_run", spec=mlflow.start_run)
def test_create_run_with_parent(mock_start_run):
    """Test creating a run with parent."""
    mock_run = Mock()
    mock_run.info.run_id = "run-123"
    mock_start_run.return_value.__enter__.return_value = mock_run

    loader = MLflowLoader()
    run_id = loader.create_run("test-project", "run-name", "exp-123", "parent-run-123")

    assert run_id == "run-123"
    mock_start_run.assert_called_once()


@patch("mlflow.search_runs", spec=mlflow.search_runs)
def test_find_run_exists(mock_search_runs):
    """Test finding an existing run."""
    mock_run = Mock()
    mock_run.info.run_id = "found-run-123"
    mock_search_runs.return_value = [mock_run]

    loader = MLflowLoader()
    run_id = loader.find_run("test-project", "run-name", "exp-123")

    assert run_id == "found-run-123"
    mock_search_runs.assert_called_once_with(
        experiment_ids=["exp-123"],
        filter_string="attributes.run_name = 'run-name'",
        output_format="list",
        max_results=1,
    )


@patch("mlflow.search_runs", spec=mlflow.search_runs)
def test_find_run_not_exists(mock_search_runs):
    """Test finding a run that doesn't exist."""
    mock_search_runs.return_value = []

    loader = MLflowLoader()
    run_id = loader.find_run("test-project", "run-name", "exp-123")

    assert run_id is None
    mock_search_runs.assert_called_once()


@patch("mlflow.search_runs", spec=mlflow.search_runs)
def test_find_run_no_experiment_id(mock_search_runs):
    """Test finding a run without experiment_id."""
    mock_run = Mock()
    mock_run.info.run_id = "found-run-123"
    mock_search_runs.return_value = [mock_run]

    loader = MLflowLoader()
    run_id = loader.find_run("test-project", "run-name", None)

    assert run_id == "found-run-123"
    mock_search_runs.assert_called_once_with(
        experiment_ids=None,
        filter_string="attributes.run_name = 'run-name'",
        output_format="list",
        max_results=1,
    )


@patch("mlflow.search_runs", spec=mlflow.search_runs)
def test_find_run_error_handling(mock_search_runs):
    """Test that find_run handles errors gracefully."""
    mock_search_runs.side_effect = Exception("Search failed")

    loader = MLflowLoader()
    run_id = loader.find_run("test-project", "run-name", "exp-123")

    assert run_id is None
    mock_search_runs.assert_called_once()


def test_upload_parameters():
    """Test parameter upload."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/param1", "test/param2", "test/param3"],
            "attribute_type": ["string", "float", "int"],
            "string_value": ["test_value", None, None],
            "float_value": [None, 3.14, None],
            "int_value": [None, None, 42],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
        }
    )

    with patch("mlflow.log_params", spec=mlflow.log_params) as mock_log_params:
        loader.upload_parameters(test_data, "RUN-123")

        # Verify parameters were logged
        mock_log_params.assert_called_once()
        logged_params = mock_log_params.call_args[0][0]

        assert "test/param1" in logged_params
        assert "test/param2" in logged_params
        assert "test/param3" in logged_params
        assert logged_params["test/param1"] == "test_value"
        assert logged_params["test/param2"] == "3.14"
        assert logged_params["test/param3"] == "42.0"


def test_upload_parameters_string_set():
    """Test parameter upload with string_set type."""
    loader = MLflowLoader()

    # Create test data with string_set
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/string_set"],
            "attribute_type": ["string_set"],
            "string_value": [None],
            "float_value": [None],
            "int_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [["value1", "value2", "value3"]],
        }
    )

    with patch("mlflow.log_params", spec=mlflow.log_params) as mock_log_params:
        loader.upload_parameters(test_data, "RUN-123")

        # Verify parameters were logged
        mock_log_params.assert_called_once()
        logged_params = mock_log_params.call_args[0][0]

        assert "test/string_set" in logged_params
        assert logged_params["test/string_set"] == "value1,value2,value3"


def test_upload_metrics():
    """Test metrics upload."""
    loader = MLflowLoader()

    # Create test data
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
        }
    )

    with patch(
        "neptune_exporter.loaders.mlflow_loader.MlflowClient", spec=MlflowClient
    ) as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        loader.upload_metrics(test_data, "RUN-123", step_multiplier=100)

        # Verify log_batch was called twice (once for each metric group)
        assert mock_client.log_batch.call_count == 2

        # Check the calls
        calls = mock_client.log_batch.call_args_list

        # First call should be for test/metric1 (2 metrics)
        first_call = calls[0]
        assert first_call[1]["run_id"] == "RUN-123"
        metrics = first_call[1]["metrics"]
        assert len(metrics) == 2
        assert all(metric.key == "test/metric1" for metric in metrics)
        assert all(isinstance(metric.step, int) for metric in metrics)

        # Second call should be for test/metric2 (1 metric)
        second_call = calls[1]
        assert second_call[1]["run_id"] == "RUN-123"
        metrics = second_call[1]["metrics"]
        assert len(metrics) == 1
        assert metrics[0].key == "test/metric2"
        assert isinstance(metrics[0].step, int)


def test_upload_run_data():
    """Test uploading complete run data."""
    loader = MLflowLoader()

    # Create test data with all required schema columns
    test_data = pd.DataFrame(
        {
            "project_id": ["test-project"] * 3,
            "run_id": ["RUN-123"] * 3,
            "attribute_path": ["test/param", "test/metric", "test/file"],
            "attribute_type": ["string", "float_series", "file"],
            "step": [None, Decimal("1.0"), None],
            "timestamp": [None, pd.Timestamp("2023-01-01"), None],
            "int_value": [None, None, None],
            "float_value": [None, 0.5, None],
            "string_value": ["test_value", None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
            "file_value": [None, None, {"path": "file.txt"}],
            "histogram_value": [None, None, None],
        }
    )

    with (
        patch("mlflow.start_run", spec=mlflow.start_run) as mock_start_run,
        patch("mlflow.log_params", spec=mlflow.log_params) as mock_log_params,
        patch(
            "neptune_exporter.loaders.mlflow_loader.MlflowClient", spec=MlflowClient
        ) as mock_client_class,
        patch("mlflow.log_artifact", spec=mlflow.log_artifact) as mock_log_artifact,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_start_run.return_value.__enter__.return_value = None
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Convert to PyArrow table with proper schema
        import pyarrow as pa
        from neptune_exporter import model

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)

        # upload_run_data now expects a generator of tables
        def table_generator():
            yield table

        loader.upload_run_data(
            table_generator(), "RUN-123", Path("/test/files"), step_multiplier=100
        )

        # Verify methods were called
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()
        mock_client.log_batch.assert_called_once()  # For metrics
        mock_log_artifact.assert_called_once()


def test_upload_artifacts_files():
    """Test file artifact upload."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file1", "test/file2"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("mlflow.log_artifact", spec=mlflow.log_artifact) as mock_log_artifact,
        patch("pathlib.Path.exists", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify artifacts were logged
        assert mock_log_artifact.call_count == 2

        calls = mock_log_artifact.call_args_list
        file_paths = [call[1]["local_path"] for call in calls]
        artifact_paths = [call[1]["artifact_path"] for call in calls]

        assert "/test/files/file1.txt" in file_paths
        assert "/test/files/file2.txt" in file_paths
        assert "test/file1" in artifact_paths
        assert "test/file2" in artifact_paths


def test_upload_artifacts_file_series():
    """Test file series artifact upload."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_series", "test/file_series"],
            "attribute_type": ["file_series", "file_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("mlflow.log_artifact", spec=mlflow.log_artifact) as mock_log_artifact,
        patch("pathlib.Path.exists", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify artifacts were logged with step information
        assert mock_log_artifact.call_count == 2

        calls = mock_log_artifact.call_args_list
        artifact_paths = [call[1]["artifact_path"] for call in calls]

        # Steps should be converted using determined multiplier
        assert any("test/file_series/step_" in path for path in artifact_paths)


def test_upload_artifacts_file_set():
    """Test file_set artifact upload (directory)."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_set1", "test/file_set2"],
            "attribute_type": ["file_set", "file_set"],
            "file_value": [
                {"path": "file_set1_dir"},
                {"path": "file_set2_dir"},
            ],
        }
    )

    with (
        patch("mlflow.log_artifact", spec=mlflow.log_artifact) as mock_log_artifact,
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify artifacts were logged
        assert mock_log_artifact.call_count == 2

        calls = mock_log_artifact.call_args_list
        # file_sets use positional argument for local_path, not keyword
        file_paths = [call[0][0] for call in calls]
        artifact_paths = [call[1]["artifact_path"] for call in calls]

        assert "/test/files/file_set1_dir" in file_paths
        assert "/test/files/file_set2_dir" in file_paths
        assert "test/file_set1" in artifact_paths
        assert "test/file_set2" in artifact_paths


def test_upload_artifacts_string_series():
    """Test string series artifact upload."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/string_series", "test/string_series"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "string_value": ["value1", "value2"],
        }
    )

    with (
        patch("mlflow.log_text", spec=mlflow.log_text) as mock_log_text,
        patch("tempfile.NamedTemporaryFile") as mock_temp_file,
    ):
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.txt"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_temp_file.return_value.__enter__.return_value = mock_file

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify text was logged
        mock_log_text.assert_called_once()
        call_args = mock_log_text.call_args
        assert call_args[1]["artifact_file"] == "test/string_series/series.txt"

        # Verify text content
        text_content = call_args[1]["text"]
        # Steps are converted using step_multiplier (100), so 1.0 becomes 100, 2.0 becomes 200
        assert "100; 2023-01-01T00:00:00; value1" in text_content
        assert "200; 2023-01-02T00:00:00; value2" in text_content


def test_upload_artifacts_histogram_series():
    """Test histogram series artifact upload."""
    loader = MLflowLoader()

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/hist_series"],
            "attribute_type": ["histogram_series"],
            "step": [Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01")],
            "histogram_value": [
                {"type": "histogram", "edges": [0.0, 1.0, 2.0], "values": [10, 20]}
            ],
        }
    )

    with patch("mlflow.log_dict", spec=mlflow.log_dict) as mock_log_dict:
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify dict was logged
        mock_log_dict.assert_called_once()
        call_args = mock_log_dict.call_args

        # Check keyword arguments
        assert call_args[1]["artifact_file"] == "test/hist_series/histograms.json"
        assert call_args[1]["run_id"] == "RUN-123"

        # Verify dictionary parameter contains histogram data wrapped in "histograms" key
        dictionary = call_args[1]["dictionary"]
        assert "histograms" in dictionary
        histogram_list = dictionary["histograms"]
        assert len(histogram_list) == 1
        # Step is converted using step_multiplier (100), so 1.0 becomes 100
        assert histogram_list[0]["step"] == 100
        assert histogram_list[0]["type"] == "histogram"
        assert histogram_list[0]["edges"] == [0.0, 1.0, 2.0]
        assert histogram_list[0]["values"] == [10, 20]


def test_upload_artifacts_artifact_type():
    """Test artifact type upload (JSON file containing artifact metadata)."""
    loader = MLflowLoader()

    # Create test data with artifact type
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/artifact1", "test/artifact2"],
            "attribute_type": ["artifact", "artifact"],
            "file_value": [
                {"path": "project/run/test/artifact1/files_list.json"},
                {"path": "project/run/test/artifact2/files_list.json"},
            ],
        }
    )

    with (
        patch("mlflow.log_artifact", spec=mlflow.log_artifact) as mock_log_artifact,
        patch("pathlib.Path.exists", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=100
        )

        # Verify artifacts were logged
        assert mock_log_artifact.call_count == 2

        calls = mock_log_artifact.call_args_list
        file_paths = [call[1]["local_path"] for call in calls]
        artifact_paths = [call[1]["artifact_path"] for call in calls]

        assert "/test/files/project/run/test/artifact1/files_list.json" in file_paths
        assert "/test/files/project/run/test/artifact2/files_list.json" in file_paths
        assert "test/artifact1" in artifact_paths
        assert "test/artifact2" in artifact_paths
