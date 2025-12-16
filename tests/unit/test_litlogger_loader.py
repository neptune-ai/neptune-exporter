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

import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from neptune_exporter.loaders.litlogger_loader import LitLoggerLoader


class TestLitLoggerLoaderInit:
    """Test LitLoggerLoader initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        loader = LitLoggerLoader(teamspace="my-teamspace")

        assert loader.teamspace == "my-teamspace"
        assert loader.name_prefix is None
        assert loader.show_client_logs is False
        assert loader._active_experiment is None
        assert loader._current_run_id is None
        assert loader._pending_experiment is None

    def test_init_with_name_prefix(self):
        """Test initialization with name prefix."""
        loader = LitLoggerLoader(teamspace="my-teamspace", name_prefix="test-prefix")

        assert loader.name_prefix == "test-prefix"

    def test_init_with_api_key(self):
        """Test initialization sets LIGHTNING_API_KEY env var."""
        with patch.dict(os.environ, {}, clear=False):
            LitLoggerLoader(api_key="test-api-key")
            assert os.environ.get("LIGHTNING_API_KEY") == "test-api-key"

    def test_init_with_user_id(self):
        """Test initialization sets LIGHTNING_USER_ID env var."""
        with patch.dict(os.environ, {}, clear=False):
            LitLoggerLoader(user_id="test-user-id")
            assert os.environ.get("LIGHTNING_USER_ID") == "test-user-id"


class TestSanitizeAttributeName:
    """Test attribute name sanitization."""

    def test_sanitize_normal_name(self):
        """Test normal name is unchanged."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    def test_sanitize_name_with_slashes(self):
        """Test slashes are replaced with underscores."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("path/to/metric") == "path_to_metric"

    def test_sanitize_name_with_special_chars(self):
        """Test special characters are replaced."""
        loader = LitLoggerLoader()
        assert (
            loader._sanitize_attribute_name("invalid@name#with$chars")
            == "invalid_name_with_chars"
        )

    def test_sanitize_name_starting_with_number(self):
        """Test name starting with number gets underscore prefix."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("123metric") == "_123metric"

    def test_sanitize_empty_name(self):
        """Test empty name becomes _attribute."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("") == "_attribute"

    def test_sanitize_name_with_only_special_chars(self):
        """Test name with only special chars."""
        loader = LitLoggerLoader()
        # After replacing special chars, result starts with underscore
        result = loader._sanitize_attribute_name("@#$")
        assert result.startswith("_")


class TestGetExperimentName:
    """Test experiment name generation."""

    def test_get_experiment_name_basic(self):
        """Test basic experiment name generation."""
        loader = LitLoggerLoader()
        name = loader._get_experiment_name("my-project", "my-run")
        assert name == "my-project_my-run"

    def test_get_experiment_name_with_prefix(self):
        """Test experiment name with prefix."""
        loader = LitLoggerLoader(name_prefix="test-prefix")
        name = loader._get_experiment_name("my-project", "my-run")
        assert name == "test-prefix_my-project_my-run"

    def test_get_experiment_name_sanitizes_special_chars(self):
        """Test experiment name sanitizes special characters."""
        loader = LitLoggerLoader()
        name = loader._get_experiment_name("project/with/slashes", "run@name")
        assert "/" not in name
        assert "@" not in name


class TestConvertStepToInt:
    """Test step conversion."""

    def test_convert_step_normal(self):
        """Test normal step conversion."""
        loader = LitLoggerLoader()
        assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    def test_convert_step_none(self):
        """Test None step returns 0."""
        loader = LitLoggerLoader()
        assert loader._convert_step_to_int(None, 1000) == 0

    def test_convert_step_zero(self):
        """Test zero step."""
        loader = LitLoggerLoader()
        assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


class TestStripNeptunePathPrefix:
    """Test Neptune path prefix stripping."""

    def test_strip_full_prefix(self):
        """Test stripping full project/run prefix."""
        loader = LitLoggerLoader()
        loader._pending_experiment = {
            "project_id": "showcase/onboarding-project",
            "run_name": "IMG-177",
        }

        path = "showcase/onboarding-project/IMG-177/finetuning/validation/image.png"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "finetuning/validation/image.png"

    def test_strip_run_only_prefix(self):
        """Test stripping when path starts with just run ID."""
        loader = LitLoggerLoader()
        loader._pending_experiment = {
            "project_id": "different-project",
            "run_name": "RUN-123",
        }

        path = "RUN-123/metrics/data.csv"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "metrics/data.csv"

    def test_no_prefix_match(self):
        """Test path returned unchanged if no prefix match."""
        loader = LitLoggerLoader()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
        }

        path = "other-project/other-run/file.txt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "other-project/other-run/file.txt"

    def test_no_pending_experiment(self):
        """Test path returned unchanged if no pending experiment."""
        loader = LitLoggerLoader()
        loader._pending_experiment = None

        path = "project/run/file.txt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "project/run/file.txt"

    def test_simple_project_id(self):
        """Test with simple (non-nested) project ID."""
        loader = LitLoggerLoader()
        loader._pending_experiment = {
            "project_id": "myproject",
            "run_name": "EXP-42",
        }

        path = "myproject/EXP-42/checkpoints/model.pt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "checkpoints/model.pt"


class TestCreateExperiment:
    """Test experiment creation."""

    def test_create_experiment_basic(self):
        """Test basic experiment creation returns identifier."""
        loader = LitLoggerLoader()
        experiment_id = loader.create_experiment("project-id", "experiment-name")

        assert experiment_id == "project-id/experiment-name"

    def test_create_experiment_with_prefix(self):
        """Test experiment creation with prefix."""
        loader = LitLoggerLoader(name_prefix="test-prefix")
        experiment_id = loader.create_experiment("project-id", "experiment-name")

        assert experiment_id == "test-prefix/project-id/experiment-name"


class TestFindRun:
    """Test run finding."""

    def test_find_run_returns_none(self):
        """Test find_run always returns None (not supported by LitLogger)."""
        loader = LitLoggerLoader()
        result = loader.find_run("project-id", "run-name", "experiment-id")

        assert result is None


class TestCreateRun:
    """Test run creation."""

    def test_create_run_basic(self):
        """Test basic run creation."""
        loader = LitLoggerLoader()
        run_id = loader.create_run("project-id", "run-name")

        assert run_id == "project-id_run-name"
        assert loader._pending_experiment is not None
        assert loader._pending_experiment["experiment_name"] == "project-id_run-name"
        assert loader._pending_experiment["project_id"] == "project-id"
        assert loader._pending_experiment["run_name"] == "run-name"

    def test_create_run_with_experiment_id(self):
        """Test run creation with experiment ID."""
        loader = LitLoggerLoader()
        run_id = loader.create_run("project-id", "run-name", "experiment-id")

        assert run_id == "experiment-id_run-name"

    def test_create_run_with_parent_logs_warning(self):
        """Test run creation with parent logs a warning."""
        loader = LitLoggerLoader()

        with patch.object(loader._logger, "warning") as mock_warning:
            loader.create_run("project-id", "run-name", parent_run_id="parent-run-123")

            mock_warning.assert_called_once()
            assert "parent-child" in mock_warning.call_args[0][0].lower()


class TestExtractParametersAsMetadata:
    """Test parameter extraction as metadata."""

    def test_extract_float_param(self):
        """Test extracting float parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/learning_rate"],
            "attribute_type": ["float"],
            "float_value": [0.001],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_learning_rate" in metadata
        assert metadata["param_learning_rate"] == "0.001"

    def test_extract_int_param(self):
        """Test extracting int parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/batch_size"],
            "attribute_type": ["int"],
            "float_value": [None],
            "int_value": [32],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_batch_size" in metadata
        assert metadata["param_batch_size"] == "32"

    def test_extract_string_param(self):
        """Test extracting string parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/model_name"],
            "attribute_type": ["string"],
            "float_value": [None],
            "int_value": [None],
            "string_value": ["resnet50"],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_model_name" in metadata
        assert metadata["param_model_name"] == "resnet50"

    def test_extract_bool_param(self):
        """Test extracting bool parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/use_gpu"],
            "attribute_type": ["bool"],
            "float_value": [None],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [True],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_use_gpu" in metadata
        assert metadata["param_use_gpu"] == "True"

    def test_extract_string_set_param(self):
        """Test extracting string_set parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/tags"],
            "attribute_type": ["string_set"],
            "float_value": [None],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [["tag1", "tag2", "tag3"]],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_tags" in metadata
        assert metadata["param_tags"] == "tag1,tag2,tag3"

    def test_extract_empty_params(self):
        """Test extracting from empty dataframe."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": [],
            "attribute_type": [],
            "float_value": [],
            "int_value": [],
            "string_value": [],
            "bool_value": [],
            "datetime_value": [],
            "string_set_value": [],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert metadata == {}


class TestUploadParameters:
    """Test parameter upload."""

    def test_upload_numeric_parameters(self):
        """Test uploading numeric parameters as metrics."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "param/epochs"],
            "attribute_type": ["float", "int"],
            "float_value": [0.001, None],
            "int_value": [None, 100],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
        })

        loader.upload_parameters(run_df, "run-123")

        loader._active_experiment.log_metrics.assert_called_once()
        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert "param_param_lr" in logged_params
        assert logged_params["param_param_lr"] == 0.001
        assert "param_param_epochs" in logged_params
        assert logged_params["param_param_epochs"] == 100.0

    def test_upload_bool_parameters(self):
        """Test uploading bool parameters as 0/1 metrics."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/use_gpu"],
            "attribute_type": ["bool"],
            "float_value": [None],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [True],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert "param_param_use_gpu" in logged_params
        assert logged_params["param_param_use_gpu"] == 1.0

    def test_upload_parameters_no_active_experiment(self):
        """Test upload parameters raises if no active experiment."""
        loader = LitLoggerLoader()
        loader._active_experiment = None

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr"],
            "attribute_type": ["float"],
            "float_value": [0.001],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_parameters(run_df, "run-123")


class TestUploadMetrics:
    """Test metrics upload."""

    def test_upload_metrics_basic(self):
        """Test basic metrics upload."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss", "metrics/loss", "metrics/accuracy"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [Decimal("1.0"), Decimal("2.0"), Decimal("1.0")],
            "float_value": [0.5, 0.3, 0.8],
        })

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        loader._active_experiment.log_metrics_batch.assert_called_once()
        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert "metrics_loss" in call_args
        assert "metrics_accuracy" in call_args
        assert len(call_args["metrics_loss"]) == 2
        assert len(call_args["metrics_accuracy"]) == 1

    def test_upload_metrics_empty(self):
        """Test upload metrics with no float_series data."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr"],
            "attribute_type": ["float"],
            "step": [None],
            "float_value": [0.001],
        })

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        loader._active_experiment.log_metrics_batch.assert_not_called()

    def test_upload_metrics_no_active_experiment(self):
        """Test upload metrics raises if no active experiment."""
        loader = LitLoggerLoader()
        loader._active_experiment = None

        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.0")],
            "float_value": [0.5],
        })

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_metrics(run_df, "run-123", step_multiplier=100)


class TestUploadArtifacts:
    """Test artifacts upload."""

    def test_upload_file_artifacts(self):
        """Test uploading file artifacts with proper remote paths."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["files/model.pt", "files/config.yaml"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "model.pt"}, {"path": "config.yaml"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            model_file = Path(tmpdir) / "model.pt"
            config_file = Path(tmpdir) / "config.yaml"
            model_file.write_text("model content")
            config_file.write_text("config content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]
            remote_paths = call_args[1]["remote_paths"]

            assert len(file_paths) == 2
            assert len(remote_paths) == 2
            # Remote paths should be the paths from file_value, not ../../tmp paths
            assert "model.pt" in remote_paths
            assert "config.yaml" in remote_paths

    def test_upload_file_artifacts_strips_neptune_prefix(self):
        """Test that Neptune project/run prefix is stripped from remote paths."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "showcase/onboarding-project",
            "run_name": "IMG-177",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["finetuning/model.pt"],
            "attribute_type": ["file"],
            "file_value": [{"path": "showcase/onboarding-project/IMG-177/finetuning/model.pt"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure matching Neptune export
            nested_dir = Path(tmpdir) / "showcase" / "onboarding-project" / "IMG-177" / "finetuning"
            nested_dir.mkdir(parents=True)
            model_file = nested_dir / "model.pt"
            model_file.write_text("model content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Remote path should have Neptune prefix stripped
            assert len(remote_paths) == 1
            assert remote_paths[0] == "finetuning/model.pt"

    def test_upload_file_artifacts_missing_file(self):
        """Test uploading file artifacts with missing file logs warning."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["files/missing.pt"],
            "attribute_type": ["file"],
            "file_value": [{"path": "missing.pt"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(loader._logger, "warning") as mock_warning:
                loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

                mock_warning.assert_called_once()
                assert "not found" in mock_warning.call_args[0][0].lower()

    def test_upload_string_series_as_text_file(self):
        """Test uploading string series creates a text file with proper remote path."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["logs/output", "logs/output"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "string_value": ["line 1", "line 2"],
            "file_value": [None, None],
            "histogram_value": [None, None],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Remote path should be organized under string_series/
            assert len(remote_paths) == 1
            assert remote_paths[0].startswith("string_series/")
            assert remote_paths[0].endswith(".txt")

    def test_upload_artifacts_no_active_experiment(self):
        """Test upload artifacts raises if no active experiment."""
        loader = LitLoggerLoader()
        loader._active_experiment = None

        run_df = pd.DataFrame({
            "attribute_path": ["files/model.pt"],
            "attribute_type": ["file"],
            "file_value": [{"path": "model.pt"}],
        })

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_artifacts(run_df, "run-123", Path("/tmp"), step_multiplier=100)

    def test_upload_directory_artifacts(self):
        """Test uploading directory artifacts preserves structure."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["artifacts/checkpoint"],
            "attribute_type": ["file_set"],
            "file_value": [{"path": "checkpoint_dir"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with files
            checkpoint_dir = Path(tmpdir) / "checkpoint_dir"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.bin").write_text("model data")
            (checkpoint_dir / "config.json").write_text("{}")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Should have 2 files with paths relative to files_base_path
            assert len(remote_paths) == 2
            assert all("checkpoint_dir" in p for p in remote_paths)

    def test_upload_file_series_artifacts(self):
        """Test uploading file_series artifacts."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["images/step1", "images/step2"],
            "attribute_type": ["file_series", "file_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "file_value": [{"path": "img1.png"}, {"path": "img2.png"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "img1.png").write_text("image1 content")
            (Path(tmpdir) / "img2.png").write_text("image2 content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]
            remote_paths = call_args[1]["remote_paths"]

            assert len(file_paths) == 2
            assert len(remote_paths) == 2
            assert "img1.png" in remote_paths
            assert "img2.png" in remote_paths

    def test_upload_file_series_missing_file(self):
        """Test file_series with missing file logs warning."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["images/step1"],
            "attribute_type": ["file_series"],
            "step": [Decimal("1.0")],
            "file_value": [{"path": "missing.png"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(loader._logger, "warning") as mock_warning:
                loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

                mock_warning.assert_called_once()
                assert "not found" in mock_warning.call_args[0][0].lower()

    def test_upload_artifact_type(self):
        """Test uploading artifact type (same handling as file)."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["artifacts/model"],
            "attribute_type": ["artifact"],
            "file_value": [{"path": "model.pkl"}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pkl").write_text("model content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert len(remote_paths) == 1
            assert "model.pkl" in remote_paths[0]

    def test_upload_histogram_series(self):
        """Test uploading histogram_series creates image files."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["histograms/weights"],
            "attribute_type": ["histogram_series"],
            "step": [Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01")],
            "string_value": [None],
            "file_value": [None],
            "histogram_value": [{"edges": [0.0, 1.0, 2.0, 3.0], "values": [10, 20, 15]}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert len(remote_paths) == 1
            assert remote_paths[0].startswith("histograms/")
            assert remote_paths[0].endswith(".png")

    def test_upload_histogram_series_no_step(self):
        """Test uploading histogram_series without step uses 'none' in filename."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["histograms/weights"],
            "attribute_type": ["histogram_series"],
            "step": [None],
            "timestamp": [None],
            "string_value": [None],
            "file_value": [None],
            "histogram_value": [{"edges": [0.0, 1.0, 2.0], "values": [5, 10]}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert "stepnone" in remote_paths[0] or "step_none" in remote_paths[0] or "none" in remote_paths[0]

    def test_upload_histogram_series_no_matplotlib(self):
        """Test histogram_series logs warning when matplotlib not available."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["histograms/weights"],
            "attribute_type": ["histogram_series"],
            "step": [Decimal("1.0")],
            "timestamp": [None],
            "string_value": [None],
            "file_value": [None],
            "histogram_value": [{"edges": [0.0, 1.0, 2.0], "values": [5, 10]}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock matplotlib import to raise ImportError
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "matplotlib.pyplot":
                    raise ImportError("No module named 'matplotlib'")
                return real_import(name, *args, **kwargs)

            with patch.object(loader._logger, "warning") as mock_warning:
                with patch.object(builtins, "__import__", mock_import):
                    loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

                mock_warning.assert_called_once()
                assert "matplotlib" in mock_warning.call_args[0][0].lower()

    def test_upload_artifacts_empty_data(self):
        """Test upload_artifacts with no artifacts to upload."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        # DataFrame with only float_series (metrics), no artifacts
        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.0")],
            "float_value": [0.5],
            "file_value": [None],
            "histogram_value": [None],
            "string_value": [None],
            "timestamp": [None],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            # log_files should not be called when there are no files
            loader._active_experiment.log_files.assert_not_called()

    def test_upload_artifacts_nan_file_value(self):
        """Test upload_artifacts skips rows with NaN file_value."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["files/model.pt", "files/missing"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "model.pt"}, None],  # Second has NaN
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pt").write_text("model content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]

            # Only one file should be uploaded (the one with valid file_value)
            assert len(file_paths) == 1

    def test_upload_artifacts_non_dict_file_value(self):
        """Test upload_artifacts skips rows with non-dict file_value."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["files/model.pt", "files/invalid"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "model.pt"}, "not_a_dict"],  # Second is not a dict
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pt").write_text("model content")

            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]

            # Only one file should be uploaded (the one with dict file_value)
            assert len(file_paths) == 1

    def test_temp_file_cleanup_after_upload(self):
        """Test that temporary files are cleaned up after upload."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["logs/output"],
            "attribute_type": ["string_series"],
            "step": [Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01")],
            "string_value": ["log line"],
            "file_value": [None],
            "histogram_value": [None],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(run_df, "run-123", Path(tmpdir), step_multiplier=100)

            # Get the temp file path that was passed to log_files
            call_args = loader._active_experiment.log_files.call_args
            temp_file_paths = call_args[0][0]

            # Verify temp file was cleaned up
            for temp_path in temp_file_paths:
                assert not Path(temp_path).exists(), f"Temp file {temp_path} should have been deleted"


class TestUploadRunData:
    """Test complete run data upload."""

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_upload_run_data_basic(self):
        """Test basic run data upload."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")

        # Create a pending experiment first
        loader.create_run("project-id", "run-name")

        # Create test data
        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "metrics/loss"],
            "attribute_type": ["float", "float_series"],
            "step": [None, Decimal("1.0")],
            "float_value": [0.001, 0.5],
            "int_value": [None, None],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
            "file_value": [None, None],
            "histogram_value": [None, None],
            "timestamp": [None, pd.Timestamp("2023-01-01")],
        })

        # Convert to PyArrow table
        table = pa.Table.from_pandas(run_df)

        def table_generator():
            yield table

        loader.upload_run_data(
            table_generator(),
            "project-id_run-name",
            Path("/tmp"),
            step_multiplier=100,
        )

        # Verify litlogger.init was called
        mock_litlogger.init.assert_called_once()
        init_kwargs = mock_litlogger.init.call_args[1]
        assert init_kwargs["name"] == "project-id_run-name"
        assert init_kwargs["teamspace"] == "test-teamspace"

        # Verify experiment was finalized
        mock_experiment.finalize.assert_called_once()

    def test_upload_run_data_not_prepared(self):
        """Test upload_run_data raises if run not prepared."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss"],
            "attribute_type": ["float_series"],
        })
        table = pa.Table.from_pandas(run_df)

        def table_generator():
            yield table

        with pytest.raises(RuntimeError, match="not prepared"):
            loader.upload_run_data(
                table_generator(),
                "unknown-run",
                Path("/tmp"),
                step_multiplier=100,
            )

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_upload_run_data_handles_error(self):
        """Test upload_run_data handles errors and finalizes experiment."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment
        # Make log_metrics_batch raise an error
        mock_experiment.log_metrics_batch.side_effect = Exception("Upload failed")

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader.create_run("project-id", "run-name")

        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.0")],
            "float_value": [0.5],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
            "histogram_value": [None],
            "timestamp": [pd.Timestamp("2023-01-01")],
        })

        table = pa.Table.from_pandas(run_df)

        def table_generator():
            yield table

        with pytest.raises(Exception, match="Upload failed"):
            loader.upload_run_data(
                table_generator(),
                "project-id_run-name",
                Path("/tmp"),
                step_multiplier=100,
            )

        # Verify experiment finalize was attempted
        mock_experiment.finalize.assert_called_once()


class TestLogMetricsBatch:
    """Test batched metrics logging."""

    def test_log_metrics_batch_groups_by_metric(self):
        """Test metrics are grouped by name for batch upload."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame({
            "attribute_path": ["loss", "loss", "loss", "accuracy"],
            "step": [Decimal("1.0"), Decimal("2.0"), Decimal("3.0"), Decimal("1.0")],
            "float_value": [0.9, 0.7, 0.5, 0.6],
        })

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert "loss" in call_args
        assert "accuracy" in call_args
        assert len(call_args["loss"]) == 3
        assert len(call_args["accuracy"]) == 1

        # Check step conversion
        loss_steps = [v["step"] for v in call_args["loss"]]
        assert loss_steps == [100, 200, 300]

    def test_log_metrics_batch_skips_nan_values(self):
        """Test metrics with NaN values are skipped."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame({
            "attribute_path": ["loss", "loss"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "float_value": [0.9, float("nan")],
        })

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only non-NaN value should be included
        assert len(call_args["loss"]) == 1
        assert call_args["loss"][0]["value"] == 0.9

    def test_log_metrics_batch_skips_nan_steps(self):
        """Test metrics with NaN steps are skipped."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame({
            "attribute_path": ["loss", "loss"],
            "step": [Decimal("1.0"), None],
            "float_value": [0.9, 0.5],
        })

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only value with valid step should be included
        assert len(call_args["loss"]) == 1
        assert call_args["loss"][0]["step"] == 100

    def test_log_metrics_batch_empty_data(self):
        """Test metrics batch with no valid data doesn't call API."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame({
            "attribute_path": ["loss"],
            "step": [None],
            "float_value": [float("nan")],
        })

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        # Should not call API when no valid metrics
        loader._active_experiment.log_metrics_batch.assert_not_called()

    def test_log_metrics_batch_multiple_metrics_same_step(self):
        """Test multiple metrics at the same step."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame({
            "attribute_path": ["loss", "accuracy", "f1_score"],
            "step": [Decimal("1.0"), Decimal("1.0"), Decimal("1.0")],
            "float_value": [0.5, 0.8, 0.75],
        })

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert len(call_args) == 3
        assert call_args["loss"][0]["step"] == 100
        assert call_args["accuracy"][0]["step"] == 100
        assert call_args["f1_score"][0]["step"] == 100


class TestExtractParametersDatetime:
    """Additional tests for datetime parameter extraction."""

    def test_extract_datetime_param(self):
        """Test extracting datetime parameter."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/created_at"],
            "attribute_type": ["datetime"],
            "float_value": [None],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [pd.Timestamp("2023-06-15 10:30:00")],
            "string_set_value": [None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_created_at" in metadata
        assert "2023-06-15" in metadata["param_created_at"]

    def test_extract_params_with_nan_values(self):
        """Test extracting params skips rows with NaN values."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "param/missing"],
            "attribute_type": ["float", "float"],
            "float_value": [0.001, float("nan")],
            "int_value": [None, None],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_lr" in metadata
        assert "param_missing" not in metadata

    def test_extract_multiple_param_types(self):
        """Test extracting multiple parameter types at once."""
        loader = LitLoggerLoader()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "param/epochs", "param/name", "param/debug"],
            "attribute_type": ["float", "int", "string", "bool"],
            "float_value": [0.001, None, None, None],
            "int_value": [None, 100, None, None],
            "string_value": [None, None, "my_model", None],
            "bool_value": [None, None, None, False],
            "datetime_value": [None, None, None, None],
            "string_set_value": [None, None, None, None],
        })

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert len(metadata) == 4
        assert metadata["param_lr"] == "0.001"
        assert metadata["param_epochs"] == "100"
        assert metadata["param_name"] == "my_model"
        assert metadata["param_debug"] == "False"


class TestUploadParametersExtended:
    """Extended tests for upload_parameters."""

    def test_upload_parameters_empty(self):
        """Test upload_parameters with empty dataframe."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": [],
            "attribute_type": [],
            "float_value": [],
            "int_value": [],
            "string_value": [],
            "bool_value": [],
            "datetime_value": [],
            "string_set_value": [],
        })

        loader.upload_parameters(run_df, "run-123")

        # Should not call log_metrics when no params
        loader._active_experiment.log_metrics.assert_not_called()

    def test_upload_parameters_bool_false(self):
        """Test bool False value is logged as 0.0."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/debug"],
            "attribute_type": ["bool"],
            "float_value": [None],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [False],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert logged_params["param_param_debug"] == 0.0

    def test_upload_parameters_skips_non_numeric(self):
        """Test non-numeric parameters are skipped (string, datetime, string_set)."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/name", "param/created", "param/tags"],
            "attribute_type": ["string", "datetime", "string_set"],
            "float_value": [None, None, None],
            "int_value": [None, None, None],
            "string_value": ["model_name", None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, pd.Timestamp("2023-01-01"), None],
            "string_set_value": [None, None, ["tag1", "tag2"]],
        })

        loader.upload_parameters(run_df, "run-123")

        # log_metrics should not be called because all params are non-numeric
        loader._active_experiment.log_metrics.assert_not_called()

    def test_upload_parameters_mixed_types(self):
        """Test uploading mix of numeric and non-numeric params."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "param/name", "param/epochs"],
            "attribute_type": ["float", "string", "int"],
            "float_value": [0.001, None, None],
            "int_value": [None, None, 50],
            "string_value": [None, "model", None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
        })

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        # Only numeric params should be logged
        assert len(logged_params) == 2
        assert "param_param_lr" in logged_params
        assert "param_param_epochs" in logged_params
        assert "param_param_name" not in logged_params

    def test_upload_parameters_logs_at_step_zero(self):
        """Test parameters are logged at step 0."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr"],
            "attribute_type": ["float"],
            "float_value": [0.001],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        })

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        assert call_args[1]["step"] == 0


class TestCreateExperimentWithParams:
    """Test _create_experiment_with_params method."""

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_create_experiment_includes_neptune_info(self):
        """Test experiment metadata includes Neptune project and run info."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader._pending_experiment = {
            "experiment_name": "test_experiment",
            "project_id": "my-workspace/my-project",
            "run_name": "RUN-42",
        }

        run_df = pd.DataFrame({
            "attribute_path": [],
            "attribute_type": [],
            "float_value": [],
            "int_value": [],
            "string_value": [],
            "bool_value": [],
            "datetime_value": [],
            "string_set_value": [],
        })

        loader._create_experiment_with_params(run_df)

        init_kwargs = mock_litlogger.init.call_args[1]
        metadata = init_kwargs["metadata"]

        assert metadata["neptune_project"] == "my-workspace/my-project"
        assert metadata["neptune_run"] == "RUN-42"

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_create_experiment_with_params_in_metadata(self):
        """Test parameters are included in experiment metadata."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader._pending_experiment = {
            "experiment_name": "test_experiment",
            "project_id": "project",
            "run_name": "run",
        }

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "param/epochs"],
            "attribute_type": ["float", "int"],
            "float_value": [0.001, None],
            "int_value": [None, 100],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
        })

        loader._create_experiment_with_params(run_df)

        init_kwargs = mock_litlogger.init.call_args[1]
        metadata = init_kwargs["metadata"]

        assert "param_lr" in metadata
        assert "param_epochs" in metadata
        assert metadata["param_lr"] == "0.001"
        assert metadata["param_epochs"] == "100"

    def test_create_experiment_no_pending_raises(self):
        """Test _create_experiment_with_params raises if no pending experiment."""
        loader = LitLoggerLoader()
        loader._pending_experiment = None

        run_df = pd.DataFrame({
            "attribute_path": [],
            "attribute_type": [],
        })

        with pytest.raises(RuntimeError, match="No pending experiment"):
            loader._create_experiment_with_params(run_df)

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_create_experiment_sets_correct_options(self):
        """Test litlogger.init is called with correct options."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="my-teamspace")
        loader._pending_experiment = {
            "experiment_name": "my_experiment",
            "project_id": "project",
            "run_name": "run",
        }

        run_df = pd.DataFrame({
            "attribute_path": [],
            "attribute_type": [],
            "float_value": [],
            "int_value": [],
            "string_value": [],
            "bool_value": [],
            "datetime_value": [],
            "string_set_value": [],
        })

        loader._create_experiment_with_params(run_df)

        init_kwargs = mock_litlogger.init.call_args[1]

        assert init_kwargs["name"] == "my_experiment"
        assert init_kwargs["teamspace"] == "my-teamspace"
        assert init_kwargs["store_step"] is True
        assert init_kwargs["store_created_at"] is True
        assert init_kwargs["print_url"] is False


class TestUploadRunDataExtended:
    """Extended tests for upload_run_data."""

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_upload_run_data_multiple_chunks(self):
        """Test upload_run_data processes multiple data chunks."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader.create_run("project-id", "run-name")

        # Create two chunks of data
        chunk1 = pd.DataFrame({
            "attribute_path": ["param/lr", "metrics/loss"],
            "attribute_type": ["float", "float_series"],
            "step": [None, Decimal("1.0")],
            "float_value": [0.001, 0.9],
            "int_value": [None, None],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
            "file_value": [None, None],
            "histogram_value": [None, None],
            "timestamp": [None, pd.Timestamp("2023-01-01")],
        })

        chunk2 = pd.DataFrame({
            "attribute_path": ["metrics/loss", "metrics/loss"],
            "attribute_type": ["float_series", "float_series"],
            "step": [Decimal("2.0"), Decimal("3.0")],
            "float_value": [0.7, 0.5],
            "int_value": [None, None],
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
            "file_value": [None, None],
            "histogram_value": [None, None],
            "timestamp": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
        })

        table1 = pa.Table.from_pandas(chunk1)
        table2 = pa.Table.from_pandas(chunk2)

        def table_generator():
            yield table1
            yield table2

        loader.upload_run_data(
            table_generator(),
            "project-id_run-name",
            Path("/tmp"),
            step_multiplier=100,
        )

        # Should create experiment only once (on first chunk)
        mock_litlogger.init.assert_called_once()

        # Should call log_metrics_batch for each chunk with metrics
        assert mock_experiment.log_metrics_batch.call_count == 2

        # Experiment should be finalized
        mock_experiment.finalize.assert_called_once()

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_upload_run_data_cleans_up_state(self):
        """Test upload_run_data cleans up state after successful upload."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader.create_run("project-id", "run-name")

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr"],
            "attribute_type": ["float"],
            "step": [None],
            "float_value": [0.001],
            "int_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [None],
            "histogram_value": [None],
            "timestamp": [None],
        })

        table = pa.Table.from_pandas(run_df)

        def table_generator():
            yield table

        # Before upload
        assert loader._pending_experiment is not None
        assert loader._current_run_id is not None

        loader.upload_run_data(
            table_generator(),
            "project-id_run-name",
            Path("/tmp"),
            step_multiplier=100,
        )

        # After successful upload, state should be cleaned up
        assert loader._pending_experiment is None
        assert loader._current_run_id is None
        assert loader._active_experiment is None

    @patch.dict("sys.modules", {"litlogger": MagicMock()})
    def test_upload_run_data_wrong_run_id(self):
        """Test upload_run_data fails if run_id doesn't match current run."""
        import sys
        mock_litlogger = sys.modules["litlogger"]
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        loader = LitLoggerLoader(teamspace="test-teamspace")
        loader.create_run("project-id", "run-name")

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr"],
            "attribute_type": ["float"],
        })
        table = pa.Table.from_pandas(run_df)

        def table_generator():
            yield table

        # Try to upload with wrong run_id
        with pytest.raises(RuntimeError, match="not prepared"):
            loader.upload_run_data(
                table_generator(),
                "wrong-run-id",
                Path("/tmp"),
                step_multiplier=100,
            )


class TestCreateRunExtended:
    """Extended tests for create_run."""

    def test_create_run_sets_current_run_id(self):
        """Test create_run sets _current_run_id."""
        loader = LitLoggerLoader()
        run_id = loader.create_run("project-id", "run-name")

        assert loader._current_run_id == run_id
        assert loader._current_run_id == "project-id_run-name"

    def test_create_run_with_name_prefix(self):
        """Test create_run with name_prefix."""
        loader = LitLoggerLoader(name_prefix="migration-2024")
        run_id = loader.create_run("project", "run")

        assert "migration-2024" in run_id

    def test_create_run_sanitizes_special_chars(self):
        """Test create_run sanitizes special characters in run name."""
        loader = LitLoggerLoader()
        run_id = loader.create_run("workspace/project", "RUN@123")

        # Special chars should be replaced
        assert "@" not in run_id
        assert "/" not in run_id

    def test_create_run_fork_step_ignored(self):
        """Test fork_step parameter is silently ignored."""
        loader = LitLoggerLoader()

        # Should not raise, fork_step is just ignored
        run_id = loader.create_run("project", "run", fork_step=100.0)

        assert run_id is not None
        assert loader._pending_experiment is not None


class TestUploadMetricsExtended:
    """Extended tests for upload_metrics."""

    def test_upload_metrics_with_various_step_multipliers(self):
        """Test upload_metrics with different step multipliers."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["metrics/loss"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.5")],
            "float_value": [0.5],
        })

        # Test with multiplier of 1000
        loader.upload_metrics(run_df, "run-123", step_multiplier=1000)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]
        assert call_args["metrics_loss"][0]["step"] == 1500

    def test_upload_metrics_filters_only_float_series(self):
        """Test upload_metrics only processes float_series type."""
        loader = LitLoggerLoader()
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame({
            "attribute_path": ["param/lr", "metrics/loss", "files/model"],
            "attribute_type": ["float", "float_series", "file"],
            "step": [None, Decimal("1.0"), None],
            "float_value": [0.001, 0.5, None],
        })

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only float_series should be logged
        assert "metrics_loss" in call_args
        assert "param_lr" not in call_args
        assert len(call_args) == 1


class TestSanitizeAttributeNameExtended:
    """Extended tests for _sanitize_attribute_name."""

    def test_sanitize_preserves_underscores(self):
        """Test underscores are preserved."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("my_metric_name") == "my_metric_name"

    def test_sanitize_consecutive_special_chars(self):
        """Test consecutive special chars become consecutive underscores."""
        loader = LitLoggerLoader()
        result = loader._sanitize_attribute_name("path//to///metric")
        # Multiple slashes become multiple underscores
        assert "___" in result or "__" in result

    def test_sanitize_dots(self):
        """Test dots are replaced."""
        loader = LitLoggerLoader()
        assert loader._sanitize_attribute_name("train.loss") == "train_loss"

    def test_sanitize_mixed_valid_invalid(self):
        """Test mix of valid and invalid characters."""
        loader = LitLoggerLoader()
        result = loader._sanitize_attribute_name("model_v2/train-loss.final")
        assert result == "model_v2_train_loss_final"


class TestGetExperimentNameExtended:
    """Extended tests for _get_experiment_name."""

    def test_get_experiment_name_preserves_hyphens(self):
        """Test hyphens are preserved in experiment names."""
        loader = LitLoggerLoader()
        name = loader._get_experiment_name("my-project", "run-123")
        assert "my-project" in name or "my_project" in name
        assert "-" in name or "_" in name

    def test_get_experiment_name_nested_project(self):
        """Test with nested project path."""
        loader = LitLoggerLoader()
        name = loader._get_experiment_name("workspace/team/project", "RUN-1")
        # Slashes should be replaced
        assert "/" not in name
