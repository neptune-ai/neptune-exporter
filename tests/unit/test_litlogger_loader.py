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


@pytest.fixture
def mock_owner():
    """Create a mock owner (User) for tests."""
    owner = MagicMock()
    owner.name = "test-user"
    owner.organizations = []
    return owner


@pytest.fixture
def mock_teamspace():
    """Create a mock Teamspace for tests."""
    teamspace = MagicMock()
    teamspace.name = "test-teamspace"
    return teamspace


@pytest.fixture
def loader_with_mocks(mock_owner, mock_teamspace):
    """Create a LitLoggerLoader with mocked dependencies."""
    with patch(
        "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
        return_value=mock_owner,
    ):
        with patch(
            "neptune_exporter.loaders.litlogger_loader.Teamspace",
            return_value=mock_teamspace,
        ):
            loader = LitLoggerLoader()
            yield loader, mock_owner, mock_teamspace


class TestLitLoggerLoaderInit:
    """Test LitLoggerLoader initialization."""

    def test_init_basic(self, mock_owner):
        """Test basic initialization."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            loader = LitLoggerLoader()

            assert loader.owner == mock_owner
            assert loader.name_prefix is None
            assert loader.show_client_logs is False
            assert loader._active_experiment is None
            assert loader._current_run_id is None
            assert loader._pending_experiment is None

    def test_init_with_name_prefix(self, mock_owner):
        """Test initialization with name prefix."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            loader = LitLoggerLoader(name_prefix="test-prefix")

            assert loader.name_prefix == "test-prefix"

    def test_init_with_api_key(self, mock_owner):
        """Test initialization sets LIGHTNING_API_KEY env var."""
        with patch.dict(os.environ, {}, clear=False):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
                return_value=mock_owner,
            ):
                LitLoggerLoader(api_key="test-api-key")
                assert os.environ.get("LIGHTNING_API_KEY") == "test-api-key"

    def test_init_with_user_id(self, mock_owner):
        """Test initialization sets LIGHTNING_USER_ID env var."""
        with patch.dict(os.environ, {}, clear=False):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
                return_value=mock_owner,
            ):
                LitLoggerLoader(user_id="test-user-id")
                assert os.environ.get("LIGHTNING_USER_ID") == "test-user-id"

    def test_init_with_owner(self, mock_owner):
        """Test initialization with owner parameter."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ) as mock_validate:
            loader = LitLoggerLoader(owner="my-org")
            mock_validate.assert_called_once_with(owner_name="my-org")
            assert loader.owner == mock_owner


class TestSanitizeAttributeName:
    """Test attribute name sanitization."""

    def test_sanitize_normal_name(self, loader_with_mocks):
        """Test normal name is unchanged."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    def test_sanitize_name_with_slashes(self, loader_with_mocks):
        """Test slashes are replaced with underscores."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("path/to/metric") == "path_to_metric"

    def test_sanitize_name_with_special_chars(self, loader_with_mocks):
        """Test special characters are replaced."""
        loader, _, _ = loader_with_mocks
        assert (
            loader._sanitize_attribute_name("invalid@name#with$chars")
            == "invalid_name_with_chars"
        )

    def test_sanitize_name_starting_with_number(self, loader_with_mocks):
        """Test name starting with number gets underscore prefix."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("123metric") == "_123metric"

    def test_sanitize_empty_name(self, loader_with_mocks):
        """Test empty name becomes _attribute."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("") == "_attribute"

    def test_sanitize_name_with_only_special_chars(self, loader_with_mocks):
        """Test name with only special chars."""
        loader, _, _ = loader_with_mocks
        # After replacing special chars, result starts with underscore
        result = loader._sanitize_attribute_name("@#$")
        assert result.startswith("_")


class TestGetExperimentName:
    """Test experiment name generation."""

    def test_get_experiment_name_basic(self, loader_with_mocks):
        """Test basic experiment name generation (just run_name)."""
        loader, _, _ = loader_with_mocks
        name = loader._get_experiment_name("my-run")
        assert name == "my-run"

    def test_get_experiment_name_with_prefix(self, mock_owner):
        """Test experiment name with prefix."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            loader = LitLoggerLoader(name_prefix="test-prefix")
            name = loader._get_experiment_name("my-run")
            assert name == "test-prefix_my-run"

    def test_get_experiment_name_sanitizes_special_chars(self, loader_with_mocks):
        """Test experiment name sanitizes special characters."""
        loader, _, _ = loader_with_mocks
        name = loader._get_experiment_name("run@name/with/slashes")
        assert "/" not in name
        assert "@" not in name

    def test_get_experiment_name_truncates_long_names(self, loader_with_mocks):
        """Test experiment name is truncated to default max length of 64."""
        loader, _, _ = loader_with_mocks
        long_name = "a" * 100
        name = loader._get_experiment_name(long_name)
        assert len(name) == 64

    def test_get_experiment_name_custom_max_length(self, loader_with_mocks):
        """Test experiment name with custom max length."""
        loader, _, _ = loader_with_mocks
        long_name = "a" * 50
        name = loader._get_experiment_name(long_name, max_length=30)
        assert len(name) == 30

    def test_get_experiment_name_short_name_unchanged(self, loader_with_mocks):
        """Test short names are not truncated."""
        loader, _, _ = loader_with_mocks
        short_name = "short-run"
        name = loader._get_experiment_name(short_name)
        assert name == short_name
        assert len(name) < 64


class TestGetTeamspaceName:
    """Test teamspace name generation from project ID."""

    def test_get_teamspace_name_basic(self, loader_with_mocks):
        """Test basic teamspace name generation."""
        loader, _, _ = loader_with_mocks
        name = loader._get_teamspace_name("my-project")
        assert name == "my-project"

    def test_get_teamspace_name_with_workspace(self, loader_with_mocks):
        """Test teamspace name with workspace prefix."""
        loader, _, _ = loader_with_mocks
        name = loader._get_teamspace_name("workspace/project-name")
        assert name == "workspace_project-name"

    def test_get_teamspace_name_sanitizes_special_chars(self, loader_with_mocks):
        """Test teamspace name sanitizes special characters."""
        loader, _, _ = loader_with_mocks
        name = loader._get_teamspace_name("workspace/project@name")
        assert "/" not in name
        assert "@" not in name
        assert name == "workspace_project_name"

    def test_get_teamspace_name_truncates_long_names(self, loader_with_mocks):
        """Test teamspace name is truncated to default max length of 64."""
        loader, _, _ = loader_with_mocks
        long_project_id = "workspace/" + "a" * 100
        name = loader._get_teamspace_name(long_project_id)
        assert len(name) == 64

    def test_get_teamspace_name_custom_max_length(self, loader_with_mocks):
        """Test teamspace name with custom max length."""
        loader, _, _ = loader_with_mocks
        long_project_id = "a" * 50
        name = loader._get_teamspace_name(long_project_id, max_length=30)
        assert len(name) == 30

    def test_get_teamspace_name_short_name_unchanged(self, loader_with_mocks):
        """Test short teamspace names are not truncated."""
        loader, _, _ = loader_with_mocks
        short_project_id = "my-project"
        name = loader._get_teamspace_name(short_project_id)
        assert name == short_project_id
        assert len(name) < 64


class TestConvertStepToInt:
    """Test step conversion."""

    def test_convert_step_normal(self, loader_with_mocks):
        """Test normal step conversion."""
        loader, _, _ = loader_with_mocks
        assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    def test_convert_step_none(self, loader_with_mocks):
        """Test None step returns 0."""
        loader, _, _ = loader_with_mocks
        assert loader._convert_step_to_int(None, 1000) == 0

    def test_convert_step_zero(self, loader_with_mocks):
        """Test zero step."""
        loader, _, _ = loader_with_mocks
        assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


class TestStripNeptunePathPrefix:
    """Test Neptune path prefix stripping."""

    def test_strip_full_prefix(self, loader_with_mocks):
        """Test stripping full project/run prefix."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._pending_experiment = {
            "project_id": "showcase/onboarding-project",
            "run_name": "IMG-177",
            "teamspace": mock_teamspace,
        }

        path = "showcase/onboarding-project/IMG-177/finetuning/validation/image.png"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "finetuning/validation/image.png"

    def test_strip_run_only_prefix(self, loader_with_mocks):
        """Test stripping when path starts with just run ID."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._pending_experiment = {
            "project_id": "different-project",
            "run_name": "RUN-123",
            "teamspace": mock_teamspace,
        }

        path = "RUN-123/metrics/data.csv"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "metrics/data.csv"

    def test_no_prefix_match(self, loader_with_mocks):
        """Test path returned unchanged if no prefix match."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "teamspace": mock_teamspace,
        }

        path = "other-project/other-run/file.txt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "other-project/other-run/file.txt"

    def test_no_pending_experiment(self, loader_with_mocks):
        """Test path returned unchanged if no pending experiment."""
        loader, _, _ = loader_with_mocks
        loader._pending_experiment = None

        path = "project/run/file.txt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "project/run/file.txt"

    def test_simple_project_id(self, loader_with_mocks):
        """Test with simple (non-nested) project ID."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._pending_experiment = {
            "project_id": "myproject",
            "run_name": "EXP-42",
            "teamspace": mock_teamspace,
        }

        path = "myproject/EXP-42/checkpoints/model.pt"
        result = loader._strip_neptune_path_prefix(path)

        assert result == "checkpoints/model.pt"


class TestCreateExperiment:
    """Test experiment creation."""

    def test_create_experiment_basic(self, loader_with_mocks):
        """Test basic experiment creation returns identifier."""
        loader, _, _ = loader_with_mocks
        experiment_id = loader.create_experiment("project-id", "experiment-name")

        assert experiment_id == "project-id/experiment-name"

    def test_create_experiment_with_prefix(self, mock_owner):
        """Test experiment creation with prefix."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            loader = LitLoggerLoader(name_prefix="test-prefix")
            experiment_id = loader.create_experiment("project-id", "experiment-name")

            assert experiment_id == "test-prefix/project-id/experiment-name"


class TestFindRun:
    """Test run finding."""

    def test_find_run_returns_none(self, loader_with_mocks):
        """Test find_run always returns None (not supported by LitLogger)."""
        loader, _, _ = loader_with_mocks
        result = loader.find_run("project-id", "run-name", "experiment-id")

        assert result is None


class TestCreateRun:
    """Test run creation."""

    def test_create_run_basic(self, mock_owner, mock_teamspace):
        """Test basic run creation."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                run_id = loader.create_run("project-id", "run-name")

                # Run ID should now be just the run_name (not project_id_run_name)
                assert run_id == "run-name"
                assert loader._pending_experiment is not None
                assert loader._pending_experiment["experiment_name"] == "run-name"
                assert loader._pending_experiment["project_id"] == "project-id"
                assert loader._pending_experiment["run_name"] == "run-name"
                assert loader._pending_experiment["teamspace"] == mock_teamspace.name

    def test_create_run_stores_teamspace_name_not_object(self, mock_owner, mock_teamspace):
        """Regression test: teamspace must be stored as a string, not a Teamspace object.

        litlogger.init() expects teamspace to be a string and resolves it internally
        via _resolve_teamspace(). Passing a Teamspace object causes _resolve_teamspace
        to fall back to the user's default teamspace, so experiments end up in the
        wrong place.
        """
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader.create_run("workspace/my-project", "run-name")

                teamspace_value = loader._pending_experiment["teamspace"]
                assert isinstance(teamspace_value, str), (
                    "teamspace must be a string for litlogger.init(), "
                    f"got {type(teamspace_value)}"
                )
                assert teamspace_value == mock_teamspace.name

    def test_create_run_creates_teamspace_from_project_id(
        self, mock_owner, mock_teamspace
    ):
        """Test that teamspace is derived from project_id."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ) as mock_teamspace_class:
                loader = LitLoggerLoader()
                loader.create_run("workspace/my-project", "run-name")

                # Verify Teamspace was called with the sanitized project name
                mock_teamspace_class.assert_called_once()
                call_kwargs = mock_teamspace_class.call_args[1]
                assert call_kwargs["name"] == "workspace_my-project"

    def test_create_run_creates_teamspace_if_not_found(
        self, mock_owner, mock_teamspace
    ):
        """Test that teamspace is created if it doesn't exist."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                side_effect=Exception("Teamspace not found"),
            ):
                mock_owner.create_teamspace.return_value = mock_teamspace
                loader = LitLoggerLoader()
                loader.create_run("project-id", "run-name")

                # Verify create_teamspace was called
                mock_owner.create_teamspace.assert_called_once_with("project-id")

    def test_create_run_with_parent_logs_warning(self, mock_owner, mock_teamspace):
        """Test run creation with parent logs a warning."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()

                with patch.object(loader._logger, "warning") as mock_warning:
                    loader.create_run(
                        "project-id", "run-name", parent_run_id="parent-run-123"
                    )

                    mock_warning.assert_called_once()
                    assert "parent-child" in mock_warning.call_args[0][0].lower()


class TestExtractParametersAsMetadata:
    """Test parameter extraction as metadata."""

    def test_extract_float_param(self, loader_with_mocks):
        """Test extracting float parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/learning_rate"],
                "attribute_type": ["float"],
                "float_value": [0.001],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_learning_rate" in metadata
        assert metadata["param_learning_rate"] == "0.001"

    def test_extract_int_param(self, loader_with_mocks):
        """Test extracting int parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/batch_size"],
                "attribute_type": ["int"],
                "float_value": [None],
                "int_value": [32],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_batch_size" in metadata
        assert metadata["param_batch_size"] == "32"

    def test_extract_string_param(self, loader_with_mocks):
        """Test extracting string parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/model_name"],
                "attribute_type": ["string"],
                "float_value": [None],
                "int_value": [None],
                "string_value": ["resnet50"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_model_name" in metadata
        assert metadata["param_model_name"] == "resnet50"

    def test_extract_bool_param(self, loader_with_mocks):
        """Test extracting bool parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/use_gpu"],
                "attribute_type": ["bool"],
                "float_value": [None],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [True],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_use_gpu" in metadata
        assert metadata["param_use_gpu"] == "True"

    def test_extract_string_set_param(self, loader_with_mocks):
        """Test extracting string_set parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/tags"],
                "attribute_type": ["string_set"],
                "float_value": [None],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [["tag1", "tag2", "tag3"]],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_tags" in metadata
        assert metadata["param_tags"] == "tag1,tag2,tag3"

    def test_extract_empty_params(self, loader_with_mocks):
        """Test extracting from empty dataframe."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": [],
                "attribute_type": [],
                "float_value": [],
                "int_value": [],
                "string_value": [],
                "bool_value": [],
                "datetime_value": [],
                "string_set_value": [],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert metadata == {}


class TestUploadParameters:
    """Test parameter upload."""

    def test_upload_numeric_parameters(self, loader_with_mocks):
        """Test uploading numeric parameters as metrics."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr", "param/epochs"],
                "attribute_type": ["float", "int"],
                "float_value": [0.001, None],
                "int_value": [None, 100],
                "string_value": [None, None],
                "bool_value": [None, None],
                "datetime_value": [None, None],
                "string_set_value": [None, None],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        loader._active_experiment.log_metrics.assert_called_once()
        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert "param_param_lr" in logged_params
        assert logged_params["param_param_lr"] == 0.001
        assert "param_param_epochs" in logged_params
        assert logged_params["param_param_epochs"] == 100.0

    def test_upload_bool_parameters(self, loader_with_mocks):
        """Test uploading bool parameters as 0/1 metrics."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/use_gpu"],
                "attribute_type": ["bool"],
                "float_value": [None],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [True],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert "param_param_use_gpu" in logged_params
        assert logged_params["param_param_use_gpu"] == 1.0

    def test_upload_parameters_no_active_experiment(self, loader_with_mocks):
        """Test upload parameters raises if no active experiment."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = None

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr"],
                "attribute_type": ["float"],
                "float_value": [0.001],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_parameters(run_df, "run-123")


class TestUploadMetrics:
    """Test metrics upload."""

    def test_upload_metrics_basic(self, loader_with_mocks):
        """Test basic metrics upload."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["metrics/loss", "metrics/loss", "metrics/accuracy"],
                "attribute_type": ["float_series", "float_series", "float_series"],
                "step": [Decimal("1.0"), Decimal("2.0"), Decimal("1.0")],
                "float_value": [0.5, 0.3, 0.8],
            }
        )

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        loader._active_experiment.log_metrics_batch.assert_called_once()
        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert "metrics_loss" in call_args
        assert "metrics_accuracy" in call_args
        assert len(call_args["metrics_loss"]) == 2
        assert len(call_args["metrics_accuracy"]) == 1

    def test_upload_metrics_empty(self, loader_with_mocks):
        """Test upload metrics with no float_series data."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr"],
                "attribute_type": ["float"],
                "step": [None],
                "float_value": [0.001],
            }
        )

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        loader._active_experiment.log_metrics_batch.assert_not_called()

    def test_upload_metrics_no_active_experiment(self, loader_with_mocks):
        """Test upload metrics raises if no active experiment."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = None

        run_df = pd.DataFrame(
            {
                "attribute_path": ["metrics/loss"],
                "attribute_type": ["float_series"],
                "step": [Decimal("1.0")],
                "float_value": [0.5],
            }
        )

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_metrics(run_df, "run-123", step_multiplier=100)


class TestUploadArtifacts:
    """Test artifacts upload."""

    def test_upload_file_artifacts(self, loader_with_mocks):
        """Test uploading file artifacts with proper remote paths."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["files/model.pt", "files/config.yaml"],
                "attribute_type": ["file", "file"],
                "file_value": [{"path": "model.pt"}, {"path": "config.yaml"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            model_file = Path(tmpdir) / "model.pt"
            config_file = Path(tmpdir) / "config.yaml"
            model_file.write_text("model content")
            config_file.write_text("config content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]
            remote_paths = call_args[1]["remote_paths"]

            assert len(file_paths) == 2
            assert len(remote_paths) == 2
            # Remote paths should be the paths from file_value, not ../../tmp paths
            assert "model.pt" in remote_paths
            assert "config.yaml" in remote_paths

    def test_upload_file_artifacts_strips_neptune_prefix(self, loader_with_mocks):
        """Test that Neptune project/run prefix is stripped from remote paths."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "showcase/onboarding-project",
            "run_name": "IMG-177",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["finetuning/model.pt"],
                "attribute_type": ["file"],
                "file_value": [
                    {"path": "showcase/onboarding-project/IMG-177/finetuning/model.pt"}
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure matching Neptune export
            nested_dir = (
                Path(tmpdir)
                / "showcase"
                / "onboarding-project"
                / "IMG-177"
                / "finetuning"
            )
            nested_dir.mkdir(parents=True)
            model_file = nested_dir / "model.pt"
            model_file.write_text("model content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Remote path should have Neptune prefix stripped
            assert len(remote_paths) == 1
            assert remote_paths[0] == "finetuning/model.pt"

    def test_upload_file_artifacts_missing_file(self, loader_with_mocks):
        """Test uploading file artifacts with missing file logs warning."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["files/missing.pt"],
                "attribute_type": ["file"],
                "file_value": [{"path": "missing.pt"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(loader._logger, "warning") as mock_warning:
                loader.upload_artifacts(
                    run_df, "run-123", Path(tmpdir), step_multiplier=100
                )

                mock_warning.assert_called_once()
                assert "not found" in mock_warning.call_args[0][0].lower()

    def test_upload_string_series_as_text_file(self, loader_with_mocks):
        """Test uploading string series creates a text file with proper remote path."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["logs/output", "logs/output"],
                "attribute_type": ["string_series", "string_series"],
                "step": [Decimal("1.0"), Decimal("2.0")],
                "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
                "string_value": ["line 1", "line 2"],
                "file_value": [None, None],
                "histogram_value": [None, None],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Remote path should be organized under string_series/
            assert len(remote_paths) == 1
            assert remote_paths[0].startswith("string_series/")
            assert remote_paths[0].endswith(".txt")

    def test_upload_artifacts_no_active_experiment(self, loader_with_mocks):
        """Test upload artifacts raises if no active experiment."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = None

        run_df = pd.DataFrame(
            {
                "attribute_path": ["files/model.pt"],
                "attribute_type": ["file"],
                "file_value": [{"path": "model.pt"}],
            }
        )

        with pytest.raises(RuntimeError, match="No active experiment"):
            loader.upload_artifacts(
                run_df, "run-123", Path("/tmp"), step_multiplier=100
            )

    def test_upload_directory_artifacts(self, loader_with_mocks):
        """Test uploading directory artifacts preserves structure."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["artifacts/checkpoint"],
                "attribute_type": ["file_set"],
                "file_value": [{"path": "checkpoint_dir"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory with files
            checkpoint_dir = Path(tmpdir) / "checkpoint_dir"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.bin").write_text("model data")
            (checkpoint_dir / "config.json").write_text("{}")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            # Should have 2 files with paths relative to files_base_path
            assert len(remote_paths) == 2
            assert all("checkpoint_dir" in p for p in remote_paths)

    def test_upload_file_series_artifacts(self, loader_with_mocks):
        """Test uploading file_series artifacts."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["images/step1", "images/step2"],
                "attribute_type": ["file_series", "file_series"],
                "step": [Decimal("1.0"), Decimal("2.0")],
                "file_value": [{"path": "img1.png"}, {"path": "img2.png"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "img1.png").write_text("image1 content")
            (Path(tmpdir) / "img2.png").write_text("image2 content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]
            remote_paths = call_args[1]["remote_paths"]

            assert len(file_paths) == 2
            assert len(remote_paths) == 2
            assert "img1.png" in remote_paths
            assert "img2.png" in remote_paths

    def test_upload_file_series_missing_file(self, loader_with_mocks):
        """Test file_series with missing file logs warning."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["images/step1"],
                "attribute_type": ["file_series"],
                "step": [Decimal("1.0")],
                "file_value": [{"path": "missing.png"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(loader._logger, "warning") as mock_warning:
                loader.upload_artifacts(
                    run_df, "run-123", Path(tmpdir), step_multiplier=100
                )

                mock_warning.assert_called_once()
                assert "not found" in mock_warning.call_args[0][0].lower()

    def test_upload_artifact_type(self, loader_with_mocks):
        """Test uploading artifact type (same handling as file)."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["artifacts/model"],
                "attribute_type": ["artifact"],
                "file_value": [{"path": "model.pkl"}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pkl").write_text("model content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert len(remote_paths) == 1
            assert "model.pkl" in remote_paths[0]

    def test_upload_histogram_series(self, loader_with_mocks):
        """Test uploading histogram_series creates image files."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["histograms/weights"],
                "attribute_type": ["histogram_series"],
                "step": [Decimal("1.0")],
                "timestamp": [pd.Timestamp("2023-01-01")],
                "string_value": [None],
                "file_value": [None],
                "histogram_value": [
                    {"edges": [0.0, 1.0, 2.0, 3.0], "values": [10, 20, 15]}
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert len(remote_paths) == 1
            assert remote_paths[0].startswith("histograms/")
            assert remote_paths[0].endswith(".png")

    def test_upload_histogram_series_no_step(self, loader_with_mocks):
        """Test uploading histogram_series without step uses 'none' in filename."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["histograms/weights"],
                "attribute_type": ["histogram_series"],
                "step": [None],
                "timestamp": [None],
                "string_value": [None],
                "file_value": [None],
                "histogram_value": [{"edges": [0.0, 1.0, 2.0], "values": [5, 10]}],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            remote_paths = call_args[1]["remote_paths"]

            assert (
                "stepnone" in remote_paths[0]
                or "step_none" in remote_paths[0]
                or "none" in remote_paths[0]
            )

    def test_upload_histogram_series_no_matplotlib(self, loader_with_mocks):
        """Test histogram_series logs warning when matplotlib not available."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["histograms/weights"],
                "attribute_type": ["histogram_series"],
                "step": [Decimal("1.0")],
                "timestamp": [None],
                "string_value": [None],
                "file_value": [None],
                "histogram_value": [{"edges": [0.0, 1.0, 2.0], "values": [5, 10]}],
            }
        )

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
                    loader.upload_artifacts(
                        run_df, "run-123", Path(tmpdir), step_multiplier=100
                    )

                mock_warning.assert_called_once()
                assert "matplotlib" in mock_warning.call_args[0][0].lower()

    def test_upload_artifacts_empty_data(self, loader_with_mocks):
        """Test upload_artifacts with no artifacts to upload."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        # DataFrame with only float_series (metrics), no artifacts
        run_df = pd.DataFrame(
            {
                "attribute_path": ["metrics/loss"],
                "attribute_type": ["float_series"],
                "step": [Decimal("1.0")],
                "float_value": [0.5],
                "file_value": [None],
                "histogram_value": [None],
                "string_value": [None],
                "timestamp": [None],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            # log_files should not be called when there are no files
            loader._active_experiment.log_files.assert_not_called()

    def test_upload_artifacts_nan_file_value(self, loader_with_mocks):
        """Test upload_artifacts skips rows with NaN file_value."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["files/model.pt", "files/missing"],
                "attribute_type": ["file", "file"],
                "file_value": [{"path": "model.pt"}, None],  # Second has NaN
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pt").write_text("model content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]

            # Only one file should be uploaded (the one with valid file_value)
            assert len(file_paths) == 1

    def test_upload_artifacts_non_dict_file_value(self, loader_with_mocks):
        """Test upload_artifacts skips rows with non-dict file_value."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["files/model.pt", "files/invalid"],
                "attribute_type": ["file", "file"],
                "file_value": [
                    {"path": "model.pt"},
                    "not_a_dict",
                ],  # Second is not a dict
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pt").write_text("model content")

            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            loader._active_experiment.log_files.assert_called_once()
            call_args = loader._active_experiment.log_files.call_args
            file_paths = call_args[0][0]

            # Only one file should be uploaded (the one with dict file_value)
            assert len(file_paths) == 1

    def test_temp_file_cleanup_after_upload(self, loader_with_mocks):
        """Test that temporary files are cleaned up after upload."""
        loader, _, mock_teamspace = loader_with_mocks
        loader._active_experiment = MagicMock()
        loader._pending_experiment = {
            "project_id": "my-project",
            "run_name": "RUN-1",
            "experiment_name": "test",
            "teamspace": mock_teamspace,
        }

        run_df = pd.DataFrame(
            {
                "attribute_path": ["logs/output"],
                "attribute_type": ["string_series"],
                "step": [Decimal("1.0")],
                "timestamp": [pd.Timestamp("2023-01-01")],
                "string_value": ["log line"],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader.upload_artifacts(
                run_df, "run-123", Path(tmpdir), step_multiplier=100
            )

            # Get the temp file path that was passed to log_files
            call_args = loader._active_experiment.log_files.call_args
            temp_file_paths = call_args[0][0]

            # Verify temp file was cleaned up
            for temp_path in temp_file_paths:
                assert not Path(temp_path).exists(), (
                    f"Temp file {temp_path} should have been deleted"
                )


class TestUploadRunData:
    """Test complete run data upload."""

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_upload_run_data_basic(self, mock_litlogger, mock_owner, mock_teamspace):
        """Test basic run data upload."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()

                # Create a pending experiment first
                loader.create_run("project-id", "run-name")

                # Create test data
                run_df = pd.DataFrame(
                    {
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
                    }
                )

                # Convert to PyArrow table
                table = pa.Table.from_pandas(run_df)

                def table_generator():
                    yield table

                loader.upload_run_data(
                    table_generator(),
                    "run-name",
                    Path("/tmp"),
                    step_multiplier=100,
                )

                # Verify litlogger.init was called with teamspace object
                mock_litlogger.init.assert_called_once()
                init_kwargs = mock_litlogger.init.call_args[1]
                assert init_kwargs["name"] == "run-name"
                assert init_kwargs["teamspace"] == mock_teamspace.name

                # Verify experiment was finalized
                mock_experiment.finalize.assert_called_once()

    def test_upload_run_data_not_prepared(self, loader_with_mocks):
        """Test upload_run_data raises if run not prepared."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["metrics/loss"],
                "attribute_type": ["float_series"],
            }
        )
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

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_upload_run_data_handles_error(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test upload_run_data handles errors and finalizes experiment."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment
        # Make log_metrics_batch raise an error
        mock_experiment.log_metrics_batch.side_effect = Exception("Upload failed")

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader.create_run("project-id", "run-name")

                run_df = pd.DataFrame(
                    {
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
                    }
                )

                table = pa.Table.from_pandas(run_df)

                def table_generator():
                    yield table

                with pytest.raises(Exception, match="Upload failed"):
                    loader.upload_run_data(
                        table_generator(),
                        "run-name",
                        Path("/tmp"),
                        step_multiplier=100,
                    )

                # Verify experiment finalize was attempted
                mock_experiment.finalize.assert_called_once()


class TestLogMetricsBatch:
    """Test batched metrics logging."""

    def test_log_metrics_batch_groups_by_metric(self, loader_with_mocks):
        """Test metrics are grouped by name for batch upload."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame(
            {
                "attribute_path": ["loss", "loss", "loss", "accuracy"],
                "step": [
                    Decimal("1.0"),
                    Decimal("2.0"),
                    Decimal("3.0"),
                    Decimal("1.0"),
                ],
                "float_value": [0.9, 0.7, 0.5, 0.6],
            }
        )

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert "loss" in call_args
        assert "accuracy" in call_args
        assert len(call_args["loss"]) == 3
        assert len(call_args["accuracy"]) == 1

        # Check step conversion
        loss_steps = [v["step"] for v in call_args["loss"]]
        assert loss_steps == [100, 200, 300]

    def test_log_metrics_batch_skips_nan_values(self, loader_with_mocks):
        """Test metrics with NaN values are skipped."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame(
            {
                "attribute_path": ["loss", "loss"],
                "step": [Decimal("1.0"), Decimal("2.0")],
                "float_value": [0.9, float("nan")],
            }
        )

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only non-NaN value should be included
        assert len(call_args["loss"]) == 1
        assert call_args["loss"][0]["value"] == 0.9

    def test_log_metrics_batch_skips_nan_steps(self, loader_with_mocks):
        """Test metrics with NaN steps are skipped."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame(
            {
                "attribute_path": ["loss", "loss"],
                "step": [Decimal("1.0"), None],
                "float_value": [0.9, 0.5],
            }
        )

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only value with valid step should be included
        assert len(call_args["loss"]) == 1
        assert call_args["loss"][0]["step"] == 100

    def test_log_metrics_batch_empty_data(self, loader_with_mocks):
        """Test metrics batch with no valid data doesn't call API."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame(
            {
                "attribute_path": ["loss"],
                "step": [None],
                "float_value": [float("nan")],
            }
        )

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        # Should not call API when no valid metrics
        loader._active_experiment.log_metrics_batch.assert_not_called()

    def test_log_metrics_batch_multiple_metrics_same_step(self, loader_with_mocks):
        """Test multiple metrics at the same step."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        metrics_data = pd.DataFrame(
            {
                "attribute_path": ["loss", "accuracy", "f1_score"],
                "step": [Decimal("1.0"), Decimal("1.0"), Decimal("1.0")],
                "float_value": [0.5, 0.8, 0.75],
            }
        )

        loader._log_metrics_batch(metrics_data, step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        assert len(call_args) == 3
        assert call_args["loss"][0]["step"] == 100
        assert call_args["accuracy"][0]["step"] == 100
        assert call_args["f1_score"][0]["step"] == 100


class TestExtractParametersDatetime:
    """Additional tests for datetime parameter extraction."""

    def test_extract_datetime_param(self, loader_with_mocks):
        """Test extracting datetime parameter."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/created_at"],
                "attribute_type": ["datetime"],
                "float_value": [None],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [pd.Timestamp("2023-06-15 10:30:00")],
                "string_set_value": [None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_created_at" in metadata
        assert "2023-06-15" in metadata["param_created_at"]

    def test_extract_params_with_nan_values(self, loader_with_mocks):
        """Test extracting params skips rows with NaN values."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr", "param/missing"],
                "attribute_type": ["float", "float"],
                "float_value": [0.001, float("nan")],
                "int_value": [None, None],
                "string_value": [None, None],
                "bool_value": [None, None],
                "datetime_value": [None, None],
                "string_set_value": [None, None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert "param_lr" in metadata
        assert "param_missing" not in metadata

    def test_extract_multiple_param_types(self, loader_with_mocks):
        """Test extracting multiple parameter types at once."""
        loader, _, _ = loader_with_mocks

        run_df = pd.DataFrame(
            {
                "attribute_path": [
                    "param/lr",
                    "param/epochs",
                    "param/name",
                    "param/debug",
                ],
                "attribute_type": ["float", "int", "string", "bool"],
                "float_value": [0.001, None, None, None],
                "int_value": [None, 100, None, None],
                "string_value": [None, None, "my_model", None],
                "bool_value": [None, None, None, False],
                "datetime_value": [None, None, None, None],
                "string_set_value": [None, None, None, None],
            }
        )

        metadata = loader._extract_parameters_as_metadata(run_df)

        assert len(metadata) == 4
        assert metadata["param_lr"] == "0.001"
        assert metadata["param_epochs"] == "100"
        assert metadata["param_name"] == "my_model"
        assert metadata["param_debug"] == "False"


class TestUploadParametersExtended:
    """Extended tests for upload_parameters."""

    def test_upload_parameters_empty(self, loader_with_mocks):
        """Test upload_parameters with empty dataframe."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": [],
                "attribute_type": [],
                "float_value": [],
                "int_value": [],
                "string_value": [],
                "bool_value": [],
                "datetime_value": [],
                "string_set_value": [],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        # Should not call log_metrics when no params
        loader._active_experiment.log_metrics.assert_not_called()

    def test_upload_parameters_bool_false(self, loader_with_mocks):
        """Test bool False value is logged as 0.0."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/debug"],
                "attribute_type": ["bool"],
                "float_value": [None],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [False],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        assert logged_params["param_param_debug"] == 0.0

    def test_upload_parameters_skips_non_numeric(self, loader_with_mocks):
        """Test non-numeric parameters are skipped (string, datetime, string_set)."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/name", "param/created", "param/tags"],
                "attribute_type": ["string", "datetime", "string_set"],
                "float_value": [None, None, None],
                "int_value": [None, None, None],
                "string_value": ["model_name", None, None],
                "bool_value": [None, None, None],
                "datetime_value": [None, pd.Timestamp("2023-01-01"), None],
                "string_set_value": [None, None, ["tag1", "tag2"]],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        # log_metrics should not be called because all params are non-numeric
        loader._active_experiment.log_metrics.assert_not_called()

    def test_upload_parameters_mixed_types(self, loader_with_mocks):
        """Test uploading mix of numeric and non-numeric params."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr", "param/name", "param/epochs"],
                "attribute_type": ["float", "string", "int"],
                "float_value": [0.001, None, None],
                "int_value": [None, None, 50],
                "string_value": [None, "model", None],
                "bool_value": [None, None, None],
                "datetime_value": [None, None, None],
                "string_set_value": [None, None, None],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        logged_params = call_args[0][0]

        # Only numeric params should be logged
        assert len(logged_params) == 2
        assert "param_param_lr" in logged_params
        assert "param_param_epochs" in logged_params
        assert "param_param_name" not in logged_params

    def test_upload_parameters_logs_at_step_zero(self, loader_with_mocks):
        """Test parameters are logged at step 0."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr"],
                "attribute_type": ["float"],
                "float_value": [0.001],
                "int_value": [None],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
            }
        )

        loader.upload_parameters(run_df, "run-123")

        call_args = loader._active_experiment.log_metrics.call_args
        assert call_args[1]["step"] == 0


class TestCreateExperimentWithParams:
    """Test _create_experiment_with_params method."""

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_create_experiment_includes_neptune_info(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test experiment metadata includes Neptune project and run info."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader._pending_experiment = {
                    "experiment_name": "test_experiment",
                    "project_id": "my-workspace/my-project",
                    "run_name": "RUN-42",
                    "teamspace": mock_teamspace,
                }

                run_df = pd.DataFrame(
                    {
                        "attribute_path": [],
                        "attribute_type": [],
                        "float_value": [],
                        "int_value": [],
                        "string_value": [],
                        "bool_value": [],
                        "datetime_value": [],
                        "string_set_value": [],
                    }
                )

                loader._create_experiment_with_params(run_df)

                init_kwargs = mock_litlogger.init.call_args[1]
                metadata = init_kwargs["metadata"]

                assert metadata["neptune_project"] == "my-workspace/my-project"
                assert metadata["neptune_run"] == "RUN-42"

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_create_experiment_with_params_in_metadata(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test parameters are included in experiment metadata."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader._pending_experiment = {
                    "experiment_name": "test_experiment",
                    "project_id": "project",
                    "run_name": "run",
                    "teamspace": mock_teamspace,
                }

                run_df = pd.DataFrame(
                    {
                        "attribute_path": ["param/lr", "param/epochs"],
                        "attribute_type": ["float", "int"],
                        "float_value": [0.001, None],
                        "int_value": [None, 100],
                        "string_value": [None, None],
                        "bool_value": [None, None],
                        "datetime_value": [None, None],
                        "string_set_value": [None, None],
                    }
                )

                loader._create_experiment_with_params(run_df)

                init_kwargs = mock_litlogger.init.call_args[1]
                metadata = init_kwargs["metadata"]

                assert "param_lr" in metadata
                assert "param_epochs" in metadata
                assert metadata["param_lr"] == "0.001"
                assert metadata["param_epochs"] == "100"

    def test_create_experiment_no_pending_raises(self, loader_with_mocks):
        """Test _create_experiment_with_params raises if no pending experiment."""
        loader, _, _ = loader_with_mocks
        loader._pending_experiment = None

        run_df = pd.DataFrame(
            {
                "attribute_path": [],
                "attribute_type": [],
            }
        )

        with pytest.raises(RuntimeError, match="No pending experiment"):
            loader._create_experiment_with_params(run_df)

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_create_experiment_sets_correct_options(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test litlogger.init is called with correct options."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader._pending_experiment = {
                    "experiment_name": "my_experiment",
                    "project_id": "project",
                    "run_name": "run",
                    "teamspace": mock_teamspace,
                }

                run_df = pd.DataFrame(
                    {
                        "attribute_path": [],
                        "attribute_type": [],
                        "float_value": [],
                        "int_value": [],
                        "string_value": [],
                        "bool_value": [],
                        "datetime_value": [],
                        "string_set_value": [],
                    }
                )

                loader._create_experiment_with_params(run_df)

                init_kwargs = mock_litlogger.init.call_args[1]

                assert init_kwargs["name"] == "my_experiment"
                assert init_kwargs["teamspace"] == mock_teamspace
                assert init_kwargs["store_step"] is True
                assert init_kwargs["store_created_at"] is True
                assert init_kwargs["print_url"] is False


class TestUploadRunDataExtended:
    """Extended tests for upload_run_data."""

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_upload_run_data_multiple_chunks(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test upload_run_data processes multiple data chunks."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader.create_run("project-id", "run-name")

                # Create two chunks of data
                chunk1 = pd.DataFrame(
                    {
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
                    }
                )

                chunk2 = pd.DataFrame(
                    {
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
                        "timestamp": [
                            pd.Timestamp("2023-01-02"),
                            pd.Timestamp("2023-01-03"),
                        ],
                    }
                )

                table1 = pa.Table.from_pandas(chunk1)
                table2 = pa.Table.from_pandas(chunk2)

                def table_generator():
                    yield table1
                    yield table2

                loader.upload_run_data(
                    table_generator(),
                    "run-name",
                    Path("/tmp"),
                    step_multiplier=100,
                )

                # Should create experiment only once (on first chunk)
                mock_litlogger.init.assert_called_once()

                # Should call log_metrics_batch for each chunk with metrics
                assert mock_experiment.log_metrics_batch.call_count == 2

                # Experiment should be finalized
                mock_experiment.finalize.assert_called_once()

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_upload_run_data_cleans_up_state(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test upload_run_data cleans up state after successful upload."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader.create_run("project-id", "run-name")

                run_df = pd.DataFrame(
                    {
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
                    }
                )

                table = pa.Table.from_pandas(run_df)

                def table_generator():
                    yield table

                # Before upload
                assert loader._pending_experiment is not None
                assert loader._current_run_id is not None

                loader.upload_run_data(
                    table_generator(),
                    "run-name",
                    Path("/tmp"),
                    step_multiplier=100,
                )

                # After successful upload, state should be cleaned up
                assert loader._pending_experiment is None
                assert loader._current_run_id is None
                assert loader._active_experiment is None

    @patch("neptune_exporter.loaders.litlogger_loader.litlogger")
    def test_upload_run_data_wrong_run_id(
        self, mock_litlogger, mock_owner, mock_teamspace
    ):
        """Test upload_run_data fails if run_id doesn't match current run."""
        mock_experiment = MagicMock()
        mock_litlogger.init.return_value = mock_experiment

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                loader.create_run("project-id", "run-name")

                run_df = pd.DataFrame(
                    {
                        "attribute_path": ["param/lr"],
                        "attribute_type": ["float"],
                    }
                )
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

    def test_create_run_sets_current_run_id(self, mock_owner, mock_teamspace):
        """Test create_run sets _current_run_id."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                run_id = loader.create_run("project-id", "run-name")

                assert loader._current_run_id == run_id
                assert loader._current_run_id == "run-name"

    def test_create_run_with_name_prefix(self, mock_owner, mock_teamspace):
        """Test create_run with name_prefix."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader(name_prefix="migration-2024")
                run_id = loader.create_run("project", "run")

                assert "migration-2024" in run_id

    def test_create_run_sanitizes_special_chars(self, mock_owner, mock_teamspace):
        """Test create_run sanitizes special characters in run name."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()
                run_id = loader.create_run("workspace/project", "RUN@123")

                # Special chars should be replaced
                assert "@" not in run_id

    def test_create_run_fork_step_ignored(self, mock_owner, mock_teamspace):
        """Test fork_step parameter is silently ignored."""
        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_owner,
        ):
            with patch(
                "neptune_exporter.loaders.litlogger_loader.Teamspace",
                return_value=mock_teamspace,
            ):
                loader = LitLoggerLoader()

                # Should not raise, fork_step is just ignored
                run_id = loader.create_run("project", "run", fork_step=100.0)

                assert run_id is not None
                assert loader._pending_experiment is not None


class TestUploadMetricsExtended:
    """Extended tests for upload_metrics."""

    def test_upload_metrics_with_various_step_multipliers(self, loader_with_mocks):
        """Test upload_metrics with different step multipliers."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["metrics/loss"],
                "attribute_type": ["float_series"],
                "step": [Decimal("1.5")],
                "float_value": [0.5],
            }
        )

        # Test with multiplier of 1000
        loader.upload_metrics(run_df, "run-123", step_multiplier=1000)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]
        assert call_args["metrics_loss"][0]["step"] == 1500

    def test_upload_metrics_filters_only_float_series(self, loader_with_mocks):
        """Test upload_metrics only processes float_series type."""
        loader, _, _ = loader_with_mocks
        loader._active_experiment = MagicMock()

        run_df = pd.DataFrame(
            {
                "attribute_path": ["param/lr", "metrics/loss", "files/model"],
                "attribute_type": ["float", "float_series", "file"],
                "step": [None, Decimal("1.0"), None],
                "float_value": [0.001, 0.5, None],
            }
        )

        loader.upload_metrics(run_df, "run-123", step_multiplier=100)

        call_args = loader._active_experiment.log_metrics_batch.call_args[0][0]

        # Only float_series should be logged
        assert "metrics_loss" in call_args
        assert "param_lr" not in call_args
        assert len(call_args) == 1


class TestSanitizeAttributeNameExtended:
    """Extended tests for _sanitize_attribute_name."""

    def test_sanitize_preserves_underscores(self, loader_with_mocks):
        """Test underscores are preserved."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("my_metric_name") == "my_metric_name"

    def test_sanitize_consecutive_special_chars(self, loader_with_mocks):
        """Test consecutive special chars become consecutive underscores."""
        loader, _, _ = loader_with_mocks
        result = loader._sanitize_attribute_name("path//to///metric")
        # Multiple slashes become multiple underscores
        assert "___" in result or "__" in result

    def test_sanitize_dots(self, loader_with_mocks):
        """Test dots are replaced."""
        loader, _, _ = loader_with_mocks
        assert loader._sanitize_attribute_name("train.loss") == "train_loss"

    def test_sanitize_mixed_valid_invalid(self, loader_with_mocks):
        """Test mix of valid and invalid characters."""
        loader, _, _ = loader_with_mocks
        result = loader._sanitize_attribute_name("model_v2/train-loss.final")
        assert result == "model_v2_train_loss_final"


class TestGetExperimentNameExtended:
    """Extended tests for _get_experiment_name."""

    def test_get_experiment_name_preserves_hyphens(self, loader_with_mocks):
        """Test hyphens are preserved in experiment names."""
        loader, _, _ = loader_with_mocks
        name = loader._get_experiment_name("run-123")
        assert "run-123" == name

    def test_get_experiment_name_run_only(self, loader_with_mocks):
        """Test experiment name is just the run name."""
        loader, _, _ = loader_with_mocks
        name = loader._get_experiment_name("RUN-1")
        # Should just be the run name, not project_id_run_name
        assert name == "RUN-1"


class TestValidateOwner:
    """Tests for _validate_owner method."""

    def test_validate_owner_returns_authed_user_when_none(self):
        """Test that authenticated user is returned when owner is None."""
        mock_authed_user = MagicMock()
        mock_authed_user.name = "test-user"
        mock_authed_user.organizations = []

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_authed_user,
        ):
            loader = LitLoggerLoader()
            assert loader.owner == mock_authed_user

    def test_validate_owner_returns_authed_user_when_name_matches(self):
        """Test that authenticated user is returned when owner name matches."""
        mock_authed_user = MagicMock()
        mock_authed_user.name = "test-user"
        mock_authed_user.organizations = []

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_authed_user,
        ):
            loader = LitLoggerLoader(owner="test-user")
            assert loader.owner == mock_authed_user

    def test_validate_owner_returns_org_when_name_matches(self):
        """Test that organization is returned when owner name matches an org."""
        mock_org = MagicMock()
        mock_org.name = "my-org"

        with patch(
            "neptune_exporter.loaders.litlogger_loader.LitLoggerLoader._validate_owner",
            return_value=mock_org,
        ):
            loader = LitLoggerLoader(owner="my-org")
            assert loader.owner == mock_org
