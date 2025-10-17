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
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from neptune_exporter.loaders.mlflow import MLflowLoader


class TestMLflowLoader:
    def test_init(self):
        """Test MLflowLoader initialization."""
        loader = MLflowLoader(
            tracking_uri="http://localhost:5000",
            experiment_name_prefix="test-prefix",
            step_multiplier=1000,
        )

        assert loader.tracking_uri == "http://localhost:5000"
        assert loader.experiment_name_prefix == "test-prefix"
        assert loader.step_multiplier == 1000

    def test_sanitize_attribute_name(self):
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

    def test_convert_step_to_int(self):
        """Test step conversion from decimal to int."""
        loader = MLflowLoader(step_multiplier=1000)

        # Test normal conversion
        assert loader._convert_step_to_int(Decimal("1.5")) == 1500

        # Test None step
        assert loader._convert_step_to_int(None) == 0

        # Test zero step
        assert loader._convert_step_to_int(Decimal("0")) == 0

    def test_get_experiment_name(self):
        """Test experiment name generation."""
        loader = MLflowLoader(experiment_name_prefix="test-prefix")

        # Test with prefix
        assert loader._get_experiment_name("my-project") == "test-prefix/my-project"

        # Test without prefix
        loader_no_prefix = MLflowLoader()
        assert loader_no_prefix._get_experiment_name("my-project") == "my-project"

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.create_experiment")
    def test_create_experiment_new(self, mock_create, mock_get):
        """Test creating a new experiment."""
        mock_get.return_value = None
        mock_create.return_value = "exp-123"

        loader = MLflowLoader()
        experiment_id = loader.create_experiment("test-project")

        assert experiment_id == "exp-123"
        mock_get.assert_called_once()
        mock_create.assert_called_once()

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.create_experiment")
    def test_create_experiment_existing(self, mock_create, mock_get):
        """Test using an existing experiment."""
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp-456"
        mock_get.return_value = mock_experiment

        loader = MLflowLoader()
        experiment_id = loader.create_experiment("test-project")

        assert experiment_id == "exp-456"
        mock_get.assert_called_once()
        mock_create.assert_not_called()

    def test_upload_parameters(self):
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

        with patch("mlflow.log_params") as mock_log_params:
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

    def test_upload_metrics(self):
        """Test metrics upload."""
        loader = MLflowLoader(step_multiplier=1000)

        # Create test data
        test_data = pd.DataFrame(
            {
                "attribute_path": ["test/metric1", "test/metric1", "test/metric2"],
                "attribute_type": ["float_series", "float_series", "float_series"],
                "step": [Decimal("1.0"), Decimal("2.0"), Decimal("1.0")],
                "float_value": [0.5, 0.7, 0.3],
            }
        )

        with patch("mlflow.log_metric") as mock_log_metric:
            loader.upload_metrics(test_data, "RUN-123")

            # Verify metrics were logged
            assert mock_log_metric.call_count == 3

            # Check specific calls
            calls = mock_log_metric.call_args_list
            metric_names = [call[0][0] for call in calls]
            values = [call[0][1] for call in calls]
            steps = [call[1]["step"] for call in calls]

            assert "test/metric1" in metric_names
            assert "test/metric2" in metric_names
            assert 0.5 in values
            assert 0.7 in values
            assert 0.3 in values
            assert 1000 in steps  # 1.0 * 1000
            assert 2000 in steps  # 2.0 * 1000

    def test_upload_artifacts_files(self):
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
            patch("mlflow.log_artifact") as mock_log_artifact,
            patch("pathlib.Path.exists", return_value=True),
        ):
            files_base_path = Path("/test/files")
            loader.upload_artifacts(test_data, "RUN-123", files_base_path)

            # Verify artifacts were logged
            assert mock_log_artifact.call_count == 2

            calls = mock_log_artifact.call_args_list
            file_paths = [call[0][0] for call in calls]
            artifact_paths = [call[1]["artifact_path"] for call in calls]

            assert "/test/files/file1.txt" in file_paths
            assert "/test/files/file2.txt" in file_paths
            assert "test/file1" in artifact_paths
            assert "test/file2" in artifact_paths

    def test_upload_artifacts_file_series(self):
        """Test file series artifact upload."""
        loader = MLflowLoader(step_multiplier=1000)

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
            patch("mlflow.log_artifact") as mock_log_artifact,
            patch("pathlib.Path.exists", return_value=True),
        ):
            files_base_path = Path("/test/files")
            loader.upload_artifacts(test_data, "RUN-123", files_base_path)

            # Verify artifacts were logged with step information
            assert mock_log_artifact.call_count == 2

            calls = mock_log_artifact.call_args_list
            artifact_paths = [call[1]["artifact_path"] for call in calls]

            assert "test/file_series/step_1000" in artifact_paths
            assert "test/file_series/step_2000" in artifact_paths

    def test_upload_artifacts_string_series(self):
        """Test string series artifact upload."""
        loader = MLflowLoader(step_multiplier=1000)

        # Create test data
        test_data = pd.DataFrame(
            {
                "attribute_path": ["test/string_series", "test/string_series"],
                "attribute_type": ["string_series", "string_series"],
                "step": [Decimal("1.0"), Decimal("2.0")],
                "string_value": ["value1", "value2"],
            }
        )

        with (
            patch("mlflow.log_artifact") as mock_log_artifact,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("pathlib.Path.unlink"),
        ):
            # Mock temporary file
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.txt"
            mock_temp_file.return_value.__enter__.return_value = mock_file

            files_base_path = Path("/test/files")
            loader.upload_artifacts(test_data, "RUN-123", files_base_path)

            # Verify artifact was logged
            mock_log_artifact.assert_called_once()
            call_args = mock_log_artifact.call_args
            assert call_args[1]["artifact_path"] == "test/string_series/series.txt"

            # Verify file content was written
            mock_file.write.assert_called_once()
            written_content = mock_file.write.call_args[0][0]
            assert "Step 1000: value1" in written_content
            assert "Step 2000: value2" in written_content

    def test_upload_artifacts_histogram_series(self):
        """Test histogram series artifact upload."""
        loader = MLflowLoader(step_multiplier=1000)

        # Create test data
        test_data = pd.DataFrame(
            {
                "attribute_path": ["test/hist_series"],
                "attribute_type": ["histogram_series"],
                "step": [Decimal("1.0")],
                "histogram_value": [
                    {"type": "histogram", "edges": [0.0, 1.0, 2.0], "values": [10, 20]}
                ],
            }
        )

        with (
            patch("mlflow.log_artifact") as mock_log_artifact,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("pathlib.Path.unlink"),
        ):
            # Mock temporary file
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.csv"
            mock_temp_file.return_value.__enter__.return_value = mock_file

            files_base_path = Path("/test/files")
            loader.upload_artifacts(test_data, "RUN-123", files_base_path)

            # Verify artifact was logged
            mock_log_artifact.assert_called_once()
            call_args = mock_log_artifact.call_args
            assert call_args[1]["artifact_path"] == "test/hist_series/histograms.csv"

            # Verify CSV content was written
            assert mock_file.write.call_count == 2  # Header + data row
            calls = mock_file.write.call_args_list
            written_content = "".join(call[0][0] for call in calls)
            assert "step,type,edges,values" in written_content
            assert '1000,histogram,"0.0,1.0,2.0","10,20"' in written_content
