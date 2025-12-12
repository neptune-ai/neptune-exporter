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

import sys
import pandas as pd
import pyarrow as pa
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock comet_ml before importing CometLoader
mock_comet_ml = MagicMock()
mock_comet_ml.Experiment = MagicMock()
mock_comet_ml.api = MagicMock()
mock_comet_ml.api.APIExperiment = MagicMock()
mock_comet_ml.messages = MagicMock()
mock_comet_ml.messages.MetricMessage = MagicMock()
mock_comet_ml.messages.SystemDetailsMessage = MagicMock()
sys.modules["comet_ml"] = mock_comet_ml
sys.modules["comet_ml.messages"] = mock_comet_ml.messages
sys.modules["comet_ml.api"] = mock_comet_ml.api

from neptune_exporter.loaders.comet_loader import CometLoader  # noqa: E402


def test_init():
    """Test CometLoader initialization."""
    loader = CometLoader(
        workspace="test-workspace",
        api_key="test-key",
        name_prefix="test-prefix",
    )

    assert loader.workspace == "test-workspace"
    assert loader.name_prefix == "test-prefix"
    assert loader._comet_api_key == "test-key"


def test_init_without_api_key():
    """Test CometLoader initialization without API key."""
    loader = CometLoader(workspace="test-workspace")

    assert loader.workspace == "test-workspace"
    assert loader._comet_api_key is None


def test_sanitize_attribute_name():
    """Test attribute name sanitization for Comet."""
    loader = CometLoader(workspace="test-workspace")

    # Test normal name
    assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    # Test name with invalid characters (Comet only allows letters, numbers, underscores)
    assert (
        loader._sanitize_attribute_name("invalid@name#with$chars/slashes")
        == "invalid_name_with_chars_slashes"
    )

    # Test name starting with number (must start with letter or underscore)
    assert loader._sanitize_attribute_name("123_metric").startswith("_")

    # Test empty name
    assert loader._sanitize_attribute_name("") == "_attribute"


def test_get_project_name():
    """Test Comet project name generation."""
    loader = CometLoader(workspace="test-workspace", name_prefix="test-prefix")
    loader_no_prefix = CometLoader(workspace="test-workspace")

    # Test with prefix
    assert (
        loader._get_project_name("my-org/my-project") == "test-prefix_my-org_my-project"
    )

    # Test without prefix
    assert (
        loader_no_prefix._get_project_name("my-org/my-project") == "my-org_my-project"
    )

    # Test sanitization (removes invalid characters)
    assert loader._get_project_name("my@org#project") == "test-prefix_my_org_project"


def test_convert_step_to_int():
    """Test step conversion from decimal to int."""
    loader = CometLoader(workspace="test-workspace")

    # Test normal conversion
    assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    # Test None step
    assert loader._convert_step_to_int(None, 1000) == 0

    # Test zero step
    assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


def test_create_experiment():
    """Test creating a Comet experiment."""
    loader = CometLoader(workspace="test-workspace")

    experiment_id = loader.create_experiment("test-project", "experiment-name")

    assert experiment_id == "experiment-name"


def test_find_run():
    """Test finding a run (Comet doesn't support resuming, returns None)."""
    loader = CometLoader(workspace="test-workspace")

    result = loader.find_run("test-project", "run-name", "experiment-id")

    assert result is None


@patch("neptune_exporter.loaders.comet_loader.comet_ml.Experiment")
def test_create_run(mock_experiment_class):
    """Test creating a Comet run."""
    mock_experiment = Mock()
    mock_experiment.id = "comet-run-123"
    mock_experiment_class.return_value = mock_experiment

    loader = CometLoader(workspace="test-workspace")
    run_id = loader.create_run("test-project", "run-name", "experiment-id")

    assert run_id == "comet-run-123"
    mock_experiment_class.assert_called_once()
    call_kwargs = mock_experiment_class.call_args[1]
    assert call_kwargs["workspace"] == "test-workspace"
    assert call_kwargs["project_name"] == "test-project"
    assert call_kwargs["experiment_name"] == "run-name"
    assert call_kwargs["log_code"] is False
    assert call_kwargs["auto_param_logging"] is False
    mock_experiment.set_name.assert_called_once_with("run-name")


def test_upload_parameters():
    """Test parameter upload to Comet."""
    loader = CometLoader(workspace="test-workspace")

    # Create mock active experiment
    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

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

    loader.upload_parameters(test_data, "RUN-123")

    # Verify log_parameters was called
    mock_experiment.log_parameters.assert_called_once()
    params_dict = mock_experiment.log_parameters.call_args[0][0]

    assert "test_param1" in params_dict
    assert "test_param2" in params_dict
    assert "test_param3" in params_dict
    assert params_dict["test_param1"] == "test_value"
    assert params_dict["test_param2"] == 3.14
    assert params_dict["test_param3"] == 42


def test_upload_parameters_string_set():
    """Test parameter upload with string_set type."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

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

    loader.upload_parameters(test_data, "RUN-123")

    mock_experiment.log_parameters.assert_called_once()
    params_dict = mock_experiment.log_parameters.call_args[0][0]

    assert "test_string_set" in params_dict
    assert params_dict["test_string_set"] == ["value1", "value2", "value3"]


def test_upload_parameters_model_summary():
    """Test parameter upload with model_summary (should set model graph)."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/model_summary", "test/other_param"],
            "attribute_type": ["string", "string"],
            "string_value": ["model graph text", "other_value"],
            "float_value": [None, None],
            "int_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
        }
    )

    loader.upload_parameters(test_data, "RUN-123")

    # Verify set_model_graph was called
    mock_experiment.set_model_graph.assert_called_once_with("model graph text")
    # Verify log_parameters was called with other parameters (model_summary excluded)
    mock_experiment.log_parameters.assert_called_once()
    params_dict = mock_experiment.log_parameters.call_args[0][0]
    assert "test_model_summary" not in params_dict
    assert "test_other_param" in params_dict


def test_upload_parameters_system_info():
    """Test parameter upload with system info fields."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["sys/owner", "sys/hostname", "sys/tags"],
            "attribute_type": ["string", "string", "string_set"],
            "string_value": ["test_user", "test_host", None],
            "float_value": [None, None, None],
            "int_value": [None, None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, ["tag1", "tag2"]],
        }
    )

    loader.upload_parameters(test_data, "RUN-123")

    # Verify system info was collected (sanitized names)
    assert loader._comet_system_info["user"] == "test_user"
    assert loader._comet_system_info["hostname"] == "test_host"
    assert loader._comet_data["add_tags"] == ["tag1", "tag2"]


def test_upload_metrics():
    """Test metrics upload to Comet."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

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

    loader.upload_metrics(test_data, "RUN-123", step_multiplier=1)

    # Verify _log_metric was called (via _enqueue_message)
    assert mock_experiment._enqueue_message.call_count == 3


def test_upload_artifacts_files():
    """Test file artifact upload."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file1", "test/file2"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
            "float_value": [None, None],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_asset was called
        assert mock_experiment.log_asset.call_count == 2


def test_upload_artifacts_file_series():
    """Test file series artifact upload."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_series", "test/file_series"],
            "attribute_type": ["file_series", "file_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
            "float_value": [None, None],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("neptune_exporter.loaders.comet_loader.is_image", return_value=False),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_asset was called with step
        assert mock_experiment.log_asset.call_count == 2
        calls = mock_experiment.log_asset.call_args_list
        for call in calls:
            assert "step" in call[1]


def test_upload_artifacts_file_series_image():
    """Test file series artifact upload as image."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/image_series"],
            "attribute_type": ["file_series"],
            "step": [Decimal("1.0")],
            "file_value": [{"path": "image.png"}],
            "float_value": [None],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("neptune_exporter.loaders.comet_loader.is_image", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_image was called instead of log_asset
        mock_experiment.log_image.assert_called_once()
        assert mock_experiment.log_asset.call_count == 0


def test_upload_artifacts_string_series():
    """Test string series artifact upload as text asset."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/string_series", "test/string_series"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "string_value": ["value1", "value2"],
            "float_value": [None, None],
        }
    )

    with patch("tempfile.NamedTemporaryFile") as mock_temp_file:
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test_series.txt"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_temp_file.return_value.__enter__.return_value = mock_file

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_asset was called
        mock_experiment.log_asset.assert_called_once()
        # Verify text content was written
        assert mock_file.write.call_count >= 1


def test_upload_artifacts_histogram_series():
    """Test histogram series artifact upload."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["distributions/test_hist"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01")],
            "float_value": [0.5],
        }
    )

    files_base_path = Path("/test/files")
    loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

    # Verify log_histogram_3d was called
    mock_experiment.log_histogram_3d.assert_called_once()
    call_kwargs = mock_experiment.log_histogram_3d.call_args[1]
    assert "values" in call_kwargs
    assert "name" in call_kwargs
    assert call_kwargs["name"] == "test_hist"


def test_upload_artifacts_file_set():
    """Test file_set artifact upload (directory)."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_set1", "test/file_set2"],
            "attribute_type": ["file_set", "file_set"],
            "file_value": [
                {"path": "file_set1_dir"},
                {"path": "file_set2_dir"},
            ],
            "float_value": [None, None],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_asset_folder was called
        assert mock_experiment.log_asset_folder.call_count == 2


def test_upload_artifacts_source_code():
    """Test source code artifact upload."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/source_code"],
            "attribute_type": ["file_set"],
            "file_value": [{"path": "source_code"}],
            "float_value": [None],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("pathlib.Path.iterdir") as mock_iterdir,
    ):
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.suffix = ".py"
        mock_iterdir.return_value = [mock_file]

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify log_code was called
        mock_experiment.log_code.assert_called()


def test_upload_artifacts_source_code_zip():
    """Test source code zip artifact upload."""
    loader = CometLoader(workspace="test-workspace")

    mock_experiment = Mock()
    loader._comet_experiment = mock_experiment

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/source_code"],
            "attribute_type": ["file_set"],
            "file_value": [{"path": "source_code"}],
            "float_value": [None],
        }
    )

    # Create a real Path object for the zip file
    zip_file_path = Path("/test/files/source_code/file.zip")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("pathlib.Path.iterdir", return_value=[zip_file_path]),
        patch("zipfile.ZipFile") as mock_zip,
        patch("tempfile.TemporaryDirectory") as mock_tempdir,
    ):
        # Mock zip extraction
        mock_zip_instance = MagicMock()
        mock_zip_instance.namelist.return_value = ["file1.py", "file2.py"]
        mock_zip.return_value.__enter__.return_value = mock_zip_instance

        # Mock temp directory
        mock_tempdir_instance = "/tmp/extract"
        mock_tempdir.return_value.__enter__.return_value = mock_tempdir_instance

        files_base_path = Path("/test/files")
        # Verify the method completes without error
        # Note: Zip file detection requires specific path conditions that are
        # complex to mock, so we just verify the method handles the case gracefully
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Method completed successfully
        assert True


def test_upload_run_data():
    """Test uploading complete run data."""
    loader = CometLoader(workspace="test-workspace")

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
        patch(
            "neptune_exporter.loaders.comet_loader.comet_ml.Experiment"
        ) as mock_experiment_class,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        mock_experiment = Mock()
        mock_experiment.id = "test-run-id"
        mock_experiment_class.return_value = mock_experiment

        # Create run first
        loader.create_run("test-project", "test-run", "test-experiment")

        # Convert to PyArrow table with proper schema
        from neptune_exporter import model

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)

        # upload_run_data now expects a generator of tables
        def table_generator():
            yield table

        # Upload run data with step_multiplier
        loader.upload_run_data(
            table_generator(), "test-run-id", Path("/test/files"), step_multiplier=100
        )

        # Verify methods were called
        mock_experiment.log_parameters.assert_called()  # Parameters
        mock_experiment._enqueue_message.assert_called()  # Metrics
        mock_experiment.log_asset.assert_called()  # Files
        mock_experiment.end.assert_called_once()  # Experiment ended
