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
import pyarrow as pa
from decimal import Decimal
from unittest.mock import Mock, patch
from pathlib import Path
import wandb

from neptune_exporter.loaders.wandb_loader import WandBLoader


@patch("wandb.login", spec=wandb.login)
def test_init(mock_login):
    """Test WandBLoader initialization."""
    loader = WandBLoader(
        entity="test-entity",
        api_key="test-key",
        name_prefix="test-prefix",
    )

    assert loader.entity == "test-entity"
    assert loader.name_prefix == "test-prefix"
    mock_login.assert_called_once_with(key="test-key")


def test_init_with_api_key():
    """Test WandBLoader initialization with API key authentication."""
    with patch("wandb.login", spec=wandb.login) as mock_login:
        loader = WandBLoader(entity="test-entity", api_key="test-api-key")

        mock_login.assert_called_once_with(key="test-api-key")
        assert loader.entity == "test-entity"


def test_sanitize_attribute_name():
    """Test attribute name sanitization for W&B."""
    loader = WandBLoader(entity="test-entity")

    # Test normal name
    assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    # Test name with invalid characters (W&B only allows letters, numbers, underscores)
    assert (
        loader._sanitize_attribute_name("invalid@name#with$chars/slashes")
        == "invalid_name_with_chars_slashes"
    )

    # Test name starting with number (must start with letter or underscore)
    assert loader._sanitize_attribute_name("123_metric").startswith("_")

    # Test empty name
    assert loader._sanitize_attribute_name("") == "_attribute"


def test_get_project_name():
    """Test W&B project name generation."""
    loader = WandBLoader(entity="test-entity", name_prefix="test-prefix")
    loader_no_prefix = WandBLoader(entity="test-entity")

    # Test with prefix
    assert (
        loader._get_project_name("my-org/my-project") == "test-prefix_my-org_my-project"
    )

    # Test without prefix
    assert (
        loader_no_prefix._get_project_name("my-org/my-project") == "my-org_my-project"
    )


def test_convert_step_to_int():
    """Test step conversion from decimal to int."""
    loader = WandBLoader(entity="test-entity")

    # Test normal conversion
    assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    # Test None step
    assert loader._convert_step_to_int(None, 1000) == 0

    # Test zero step
    assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


def test_create_experiment():
    """Test creating a W&B project (experiment)."""
    loader = WandBLoader(entity="test-entity")

    project_name = loader.create_experiment("test-project", "experiment-name")

    assert project_name == "experiment-name"


@patch("wandb.init", spec=wandb.init)
def test_create_run(mock_init):
    """Test creating a W&B run."""
    mock_run = Mock()
    mock_run.id = "wandb-run-123"
    mock_init.return_value = mock_run

    loader = WandBLoader(entity="test-entity")
    run_id = loader.create_run("test-project", "run-name", "experiment-id")

    assert run_id == "wandb-run-123"
    mock_init.assert_called_once_with(
        entity="test-entity",
        project="test-project",
        group="experiment-id",
        name="run-name",
    )


@patch("wandb.init", spec=wandb.init)
def test_create_run_with_parent(mock_init):
    """Test creating a forked W&B run."""
    mock_run = Mock()
    mock_run.id = "wandb-run-child"
    mock_init.return_value = mock_run

    loader = WandBLoader(entity="test-entity")

    # Create child with parent
    run_id = loader.create_run(
        "test-project", "child-run", "experiment-id", "wandb-run-parent"
    )

    assert run_id == "wandb-run-child"

    # Check fork_from parameter
    call_kwargs = mock_init.call_args[1]
    assert "fork_from" in call_kwargs
    assert "wandb-run-parent" in call_kwargs["fork_from"]


def test_upload_parameters():
    """Test parameter upload to W&B."""
    loader = WandBLoader(entity="test-entity")

    # Create mock active run
    mock_run = Mock()
    mock_config = Mock()
    mock_run.config = mock_config
    loader._active_run = mock_run

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

    # Verify config.update was called
    mock_config.update.assert_called_once()
    config_dict = mock_config.update.call_args[0][0]

    assert "test_param1" in config_dict
    assert "test_param2" in config_dict
    assert "test_param3" in config_dict
    assert config_dict["test_param1"] == "test_value"
    assert config_dict["test_param2"] == 3.14
    assert config_dict["test_param3"] == 42


def test_upload_parameters_string_set():
    """Test parameter upload with string_set type."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    mock_config = Mock()
    mock_run.config = mock_config
    loader._active_run = mock_run

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

    mock_config.update.assert_called_once()
    config_dict = mock_config.update.call_args[0][0]

    assert "test_string_set" in config_dict
    assert config_dict["test_string_set"] == ["value1", "value2", "value3"]


def test_upload_metrics():
    """Test metrics upload to W&B."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
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
        }
    )

    loader.upload_metrics(test_data, "RUN-123", step_multiplier=1)

    # Verify log was called twice (once for each step)
    assert mock_run.log.call_count == 2

    # Check the calls
    calls = mock_run.log.call_args_list

    # Both calls should have step parameter
    for call in calls:
        assert "step" in call[1]


def test_upload_artifacts_files():
    """Test file artifact upload."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file1", "test/file2"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2


def test_upload_artifacts_file_series():
    """Test file series artifact upload."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_series", "test/file_series"],
            "attribute_type": ["file_series", "file_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify artifacts include step in name
        assert mock_artifact_class.call_count == 2
        calls = mock_artifact_class.call_args_list

        # Check that step is included in artifact names
        for call in calls:
            artifact_name = call[1]["name"]
            assert "step_" in artifact_name


def test_upload_artifacts_string_series():
    """Test string series artifact upload as text artifact."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

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
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
        patch("tempfile.NamedTemporaryFile") as mock_temp_file,
    ):
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test_series.txt"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_temp_file.return_value.__enter__.return_value = mock_file

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify artifact was created and logged
        mock_artifact_class.assert_called_once_with(
            name="test_string_series", type="string_series"
        )
        mock_artifact.add_file.assert_called_once()
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

        # Verify text content was written
        assert mock_file.write.call_count >= 1
        # Get all written text (in case write is called multiple times)
        written_calls = mock_file.write.call_args_list
        all_written_text = "".join(call[0][0] for call in written_calls)
        assert "1; 2023-01-01T00:00:00; value1" in all_written_text
        assert "2; 2023-01-02T00:00:00; value2" in all_written_text


def test_upload_artifacts_histogram_series():
    """Test histogram series artifact upload as W&B Histogram."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

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

    with patch("wandb.Histogram", spec=wandb.Histogram) as mock_histogram_class:
        mock_histogram = Mock()
        mock_histogram_class.return_value = mock_histogram

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify Histogram was created and logged
        mock_histogram_class.assert_called_once()
        mock_run.log.assert_called_once()

        # Check histogram creation
        call_kwargs = mock_histogram_class.call_args[1]
        assert "np_histogram" in call_kwargs
        values, edges = call_kwargs["np_histogram"]
        assert values == [10, 20]
        assert edges == [0.0, 1.0, 2.0]


def test_upload_artifacts_file_set():
    """Test file_set artifact upload (directory)."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

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
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2

        # Verify artifact types are set correctly
        calls = mock_artifact_class.call_args_list
        assert calls[0][1]["type"] == "file_set"
        assert calls[1][1]["type"] == "file_set"

        # Verify add_dir was called (not add_file) for directories
        assert mock_artifact.add_dir.call_count == 2
        assert mock_artifact.add_file.call_count == 0


def test_upload_artifacts_artifact_type():
    """Test artifact type upload (JSON file containing artifact metadata)."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

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
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(
            test_data, "RUN-123", files_base_path, step_multiplier=1
        )

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2

        # Verify artifact types are set correctly
        calls = mock_artifact_class.call_args_list
        assert calls[0][1]["type"] == "artifact"
        assert calls[1][1]["type"] == "artifact"

        # Verify add_file was called (not add_dir) for files
        assert mock_artifact.add_file.call_count == 2
        assert mock_artifact.add_dir.call_count == 0

        # Verify file paths
        file_paths = [call[0][0] for call in mock_artifact.add_file.call_args_list]
        assert "/test/files/project/run/test/artifact1/files_list.json" in file_paths
        assert "/test/files/project/run/test/artifact2/files_list.json" in file_paths


def test_upload_run_data():
    """Test uploading complete run data."""
    loader = WandBLoader(entity="test-entity")

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
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        mock_init.return_value = mock_run
        mock_artifact_class.return_value = Mock()

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
        mock_run.config.update.assert_called_once()  # Parameters
        mock_run.log.assert_called()  # Metrics
        mock_run.log_artifact.assert_called_once()  # Files
        mock_run.finish.assert_called_once()  # Run finished
