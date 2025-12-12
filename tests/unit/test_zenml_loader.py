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
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pytest

from neptune_exporter.loaders.zenml_loader import ZenMLLoader


def _create_loader(name_prefix: str | None = None) -> tuple[ZenMLLoader, Mock]:
    """Helper to create a ZenMLLoader instance with mocked ZenML Client.

    This helper patches ZENML_AVAILABLE to True and replaces Client with a Mock,
    so tests can instantiate ZenMLLoader without a real ZenML installation.
    """
    with (
        patch(
            "neptune_exporter.loaders.zenml_loader.ZENML_AVAILABLE",
            True,
        ),
        patch(
            "neptune_exporter.loaders.zenml_loader.Client",
        ) as mock_client_class,
    ):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        loader = ZenMLLoader(name_prefix=name_prefix, show_client_logs=False)

    return loader, mock_client


@patch("neptune_exporter.loaders.zenml_loader.Client")
def test_init_with_zenml_available(mock_client_class: Mock) -> None:
    """Test ZenMLLoader initialization when ZenML is available."""
    with patch(
        "neptune_exporter.loaders.zenml_loader.ZENML_AVAILABLE",
        True,
    ):
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        loader = ZenMLLoader(name_prefix="test-prefix", show_client_logs=True)

    assert loader.name_prefix == "test-prefix"
    mock_client_class.assert_called_once_with()
    assert loader._client is mock_client_instance  # type: ignore[attr-defined]


def test_init_without_zenml_available() -> None:
    """Test ZenMLLoader initialization fails when ZenML is not available."""
    with patch(
        "neptune_exporter.loaders.zenml_loader.ZENML_AVAILABLE",
        False,
    ):
        with pytest.raises(RuntimeError):
            ZenMLLoader()


def test_sanitize_attribute_name() -> None:
    """Test attribute name sanitization for ZenML metadata keys."""
    loader, _ = _create_loader()

    # Normal name should remain unchanged
    assert loader._sanitize_attribute_name("normal/name") == "normal/name"

    # Invalid characters should be replaced with underscores
    assert (
        loader._sanitize_attribute_name("invalid@name#with$chars/slashes")
        == "invalid_name_with_chars/slashes"
    )

    # Very long names should be truncated to 250 characters
    long_name = "a" * 300
    sanitized = loader._sanitize_attribute_name(long_name)
    assert len(sanitized) == 250
    assert sanitized == "a" * 250

    # Empty name should get a fallback key
    assert loader._sanitize_attribute_name("") == "_attribute"


def test_sanitize_name() -> None:
    """Test generic name sanitization for models and model versions."""
    loader, _ = _create_loader()

    # Invalid characters replaced with underscores
    assert (
        loader._sanitize_name("My Name/With*Invalid#Chars")
        == "My Name/With_Invalid_Chars"
    )

    # Very long names should be truncated
    long_name = "x" * 300
    sanitized = loader._sanitize_name(long_name)
    assert len(sanitized) == 250
    assert sanitized == "x" * 250

    # Empty name should get a default fallback
    assert loader._sanitize_name("") == "neptune-import"


def test_get_model_name_with_and_without_prefix() -> None:
    """Test model name derivation from project id with and without a prefix."""
    loader_with_prefix, _ = _create_loader(name_prefix="prefix")
    loader_no_prefix, _ = _create_loader(name_prefix=None)

    # Project IDs have slashes replaced with dashes
    assert (
        loader_with_prefix._get_model_name("org/my-project")
        == "prefix-neptune-export-org-my-project"
    )
    assert (
        loader_no_prefix._get_model_name("org/my-project")
        == "neptune-export-org-my-project"
    )


def test_create_experiment_creates_new_model() -> None:
    """Test create_experiment creates a new ZenML Model when none exists."""
    loader, mock_client = _create_loader()

    # No existing models with the target name
    mock_client.list_models.return_value = []
    mock_model = Mock()
    mock_model.id = "model-123"
    mock_model.name = "neptune-export-org-my-project"
    mock_client.create_model.return_value = mock_model

    experiment_id = loader.create_experiment("org/my-project", "experiment-name")

    expected_model_name = "neptune-export-org-my-project"
    mock_client.list_models.assert_called_once_with(name=expected_model_name)
    mock_client.create_model.assert_called_once()
    assert experiment_id == "model-123"


def test_create_experiment_uses_existing_model() -> None:
    """Test create_experiment reuses an existing ZenML Model."""
    loader, mock_client = _create_loader()

    mock_model = Mock()
    mock_model.id = "model-456"
    mock_model.name = "neptune-export-org-my-project"
    mock_client.list_models.return_value = [mock_model]

    experiment_id = loader.create_experiment("org/my-project", "experiment-name")

    mock_client.list_models.assert_called_once_with(
        name="neptune-export-org-my-project"
    )
    mock_client.create_model.assert_not_called()
    assert experiment_id == "model-456"


def test_find_run_existing_model_version() -> None:
    """Test find_run returns existing model version ID when found."""
    loader, mock_client = _create_loader()

    mock_mv = Mock()
    mock_mv.id = "mv-123"
    mock_mv.name = "run-name"
    mock_client.list_model_versions.return_value = [mock_mv]

    run_id = loader.find_run("org/my-project", "run-name", "model-123")

    mock_client.list_model_versions.assert_called_once_with(
        model_name_or_id="model-123",
        name="run-name",
    )
    assert run_id == "mv-123"


def test_find_run_not_found() -> None:
    """Test find_run returns None when no model version matches."""
    loader, mock_client = _create_loader()

    mock_client.list_model_versions.return_value = []

    run_id = loader.find_run("org/my-project", "missing-run", "model-123")

    mock_client.list_model_versions.assert_called_once_with(
        model_name_or_id="model-123",
        name="missing-run",
    )
    assert run_id is None


def test_create_run_with_experiment_id() -> None:
    """Test create_run creates a model version under an existing model."""
    loader, mock_client = _create_loader()

    mock_mv = Mock()
    mock_mv.id = "mv-123"
    mock_mv.name = "run-name"
    mock_client.create_model_version.return_value = mock_mv

    run_id = loader.create_run(
        project_id="org/my-project",
        run_name="run-name",
        experiment_id="model-123",
    )

    mock_client.create_model_version.assert_called_once()
    assert run_id == "mv-123"


def test_create_run_without_experiment_id_calls_get_or_create_model() -> None:
    """Test create_run obtains a model when experiment_id is not provided."""
    loader, mock_client = _create_loader()

    # Ensure _get_or_create_model is used
    mock_get_or_create = Mock(return_value="model-999")
    loader._get_or_create_model = mock_get_or_create  # type: ignore[assignment]

    mock_mv = Mock()
    mock_mv.id = "mv-999"
    mock_mv.name = "run-name"
    mock_client.create_model_version.return_value = mock_mv

    run_id = loader.create_run(
        project_id="org/my-project",
        run_name="run-name",
        experiment_id=None,
    )

    mock_get_or_create.assert_called_once_with("org/my-project")
    mock_client.create_model_version.assert_called_once()
    assert run_id == "mv-999"


def test_log_metadata_to_zenml_uses_model_version_id() -> None:
    """Test _log_metadata_to_zenml calls zenml.log_metadata with expected args."""
    from uuid import UUID as PyUUID

    loader, _ = _create_loader()

    # Use a valid UUID since the implementation converts to UUID
    model_version_uuid_str = "669e2121-d812-4ce2-9738-6b6be3a004a3"
    expected_uuid = PyUUID(model_version_uuid_str)

    with patch(
        "neptune_exporter.loaders.zenml_loader.log_metadata"
    ) as mock_log_metadata:
        metadata = {"a": 1, "b": "value"}
        loader._log_metadata_to_zenml(run_id=model_version_uuid_str, metadata=metadata)  # type: ignore[arg-type]

    # Primary API should be used (metadata + model_version_id)
    assert mock_log_metadata.call_count == 1
    _, kwargs = mock_log_metadata.call_args
    assert kwargs["metadata"] == metadata
    assert kwargs["model_version_id"] == expected_uuid


def test_upload_run_data_aggregates_and_logs_metadata() -> None:
    """Test upload_run_data aggregates scalars, series, files, and tags correctly."""
    loader, _ = _create_loader()

    # Build a DataFrame with all relevant attribute types
    num_rows = 13
    project_ids = ["test-project"] * num_rows
    run_ids = ["RUN-123"] * num_rows

    attribute_paths = [
        "parameters/float_param",  # 0 float
        "parameters/int_param",  # 1 int
        "parameters/string_param",  # 2 string
        "parameters/bool_param",  # 3 bool
        "parameters/datetime_param",  # 4 datetime
        "parameters/string_set_param",  # 5 string_set
        "sys/tags",  # 6 tags string_set
        "sys/group_tags",  # 7 group tags string_set
        "sys/name",  # 8 run display name
        "sys/description",  # 9 description
        "metrics/accuracy",  # 10 float_series step 1
        "metrics/accuracy",  # 11 float_series step 2
        "artifacts/model",  # 12 file reference
    ]

    attribute_types = [
        "float",  # 0
        "int",  # 1
        "string",  # 2
        "bool",  # 3
        "datetime",  # 4
        "string_set",  # 5
        "string_set",  # 6
        "string_set",  # 7
        "string",  # 8
        "string",  # 9
        "float_series",  # 10
        "float_series",  # 11
        "file",  # 12
    ]

    steps = [
        None,  # 0
        None,  # 1
        None,  # 2
        None,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        Decimal("1.0"),  # 10
        Decimal("2.0"),  # 11
        None,  # 12
    ]

    timestamps = [
        pd.NaT,  # 0
        pd.NaT,  # 1
        pd.NaT,  # 2
        pd.NaT,  # 3
        pd.NaT,  # 4
        pd.NaT,  # 5
        pd.NaT,  # 6
        pd.NaT,  # 7
        pd.NaT,  # 8
        pd.NaT,  # 9
        pd.NaT,  # 10
        pd.NaT,  # 11
        pd.NaT,  # 12
    ]

    int_values = [
        None,  # 0
        5,  # 1
        None,  # 2
        None,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        None,  # 12
    ]

    float_values = [
        1.23,  # 0
        None,  # 1
        None,  # 2
        None,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        0.8,  # 10
        0.9,  # 11
        None,  # 12
    ]

    string_values = [
        None,  # 0
        None,  # 1
        "hello",  # 2
        None,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        "Run Name",  # 8
        "Run description",  # 9
        None,  # 10
        None,  # 11
        None,  # 12
    ]

    bool_values = [
        None,  # 0
        None,  # 1
        None,  # 2
        True,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        None,  # 12
    ]

    datetime_values = [
        None,  # 0
        None,  # 1
        None,  # 2
        None,  # 3
        pd.Timestamp("2023-01-02T12:00:00Z"),  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        None,  # 12
    ]

    string_set_values = [
        None,  # 0
        None,  # 1
        None,  # 2
        None,  # 3
        None,  # 4
        ["a", "b"],  # 5
        ["tag1", "tag2"],  # 6
        ["group1", "group2"],  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        None,  # 12
    ]

    file_values = [
        None,  # 0
        None,  # 1
        None,  # 2
        None,  # 3
        None,  # 4
        None,  # 5
        None,  # 6
        None,  # 7
        None,  # 8
        None,  # 9
        None,  # 10
        None,  # 11
        {"path": "artifact/model.pkl"},  # 12
    ]

    histogram_values = [None] * num_rows

    df = pd.DataFrame(
        {
            "project_id": project_ids,
            "run_id": run_ids,
            "attribute_path": attribute_paths,
            "attribute_type": attribute_types,
            "step": steps,
            "timestamp": timestamps,
            "int_value": int_values,
            "float_value": float_values,
            "string_value": string_values,
            "bool_value": bool_values,
            "datetime_value": datetime_values,
            "string_set_value": string_set_values,
            "file_value": file_values,
            "histogram_value": histogram_values,
        }
    )

    from neptune_exporter import model

    table = pa.Table.from_pandas(df, schema=model.SCHEMA)

    def table_generator():
        yield table

    # Patch helper methods to intercept description/tags and metadata payload
    # Note: save_artifact is None in tests (ZenML not installed), so _upload_files_as_artifacts
    # will fall back to storing file paths as metadata only.
    with (
        patch.object(
            loader,
            "_update_model_version_description_and_tags",
        ) as mock_update_desc_tags,
        patch.object(
            loader,
            "_log_metadata_to_zenml",
        ) as mock_log_metadata,
        patch(
            "neptune_exporter.loaders.zenml_loader.save_artifact",
            None,  # Simulate ZenML not fully available
        ),
    ):
        loader.upload_run_data(
            run_data=table_generator(),
            run_id="mv-123",
            files_directory=Path("/exports/files"),
            step_multiplier=100,
        )

    # Description and tags should be derived from sys/ attributes
    assert mock_update_desc_tags.call_count == 1
    _, update_kwargs = mock_update_desc_tags.call_args
    assert update_kwargs["run_id"] == "mv-123"
    assert update_kwargs["description"] == "Run description"
    assert set(update_kwargs["tags"]) == {"tag1", "tag2"}
    assert set(update_kwargs["experiment_tags"]) == {"group1", "group2"}

    # Metadata logged to ZenML should include scalars, series stats, file refs, and tag metadata
    assert mock_log_metadata.call_count == 1
    _, log_kwargs = mock_log_metadata.call_args
    assert log_kwargs["run_id"] == "mv-123"
    metadata = log_kwargs["metadata"]

    # Import provenance metadata (nested under "neptune_import" card)
    assert metadata["neptune_import"]["source"] == "neptune"
    assert metadata["neptune_import"]["tool"] == "neptune-exporter"
    assert metadata["neptune_import"]["loader"] == "zenml"
    assert metadata["neptune_import"]["target_run_id"] == "mv-123"

    # Scalar parameters (nested under "parameters" card)
    assert metadata["parameters"]["float_param"] == pytest.approx(1.23)
    assert metadata["parameters"]["int_param"] == 5
    assert metadata["parameters"]["string_param"] == "hello"
    assert metadata["parameters"]["bool_param"] is True
    assert "datetime_param" in metadata["parameters"]
    assert metadata["parameters"]["string_set_param"] == ["a", "b"]

    # Tags preserved as metadata (nested under "sys" card)
    assert metadata["sys"]["tags"] == ["tag1", "tag2"]
    assert metadata["sys"]["group_tags"] == ["group1", "group2"]

    # Float series summary statistics (nested under "series" card)
    assert metadata["series"]["metrics"]["accuracy"]["min"] == pytest.approx(0.8)
    assert metadata["series"]["metrics"]["accuracy"]["max"] == pytest.approx(0.9)
    assert metadata["series"]["metrics"]["accuracy"]["final"] == pytest.approx(0.9)
    assert metadata["series"]["metrics"]["accuracy"]["count"] == 2

    # File references stored as metadata (nested under "files" card)
    assert metadata["files"]["artifacts"]["model"] == "artifact/model.pkl"


def test_upload_files_as_artifacts_with_save_artifact(tmp_path: Path) -> None:
    """Test _upload_files_as_artifacts uploads files when save_artifact is available."""
    from uuid import UUID as PyUUID

    loader, mock_client = _create_loader()

    # Create a mock file in the temporary directory
    test_file = tmp_path / "artifact" / "model.pkl"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("mock model content")

    # Use valid UUIDs for all IDs
    artifact_version_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    model_version_uuid = "669e2121-d812-4ce2-9738-6b6be3a004a3"

    # Mock the ZenML artifact version response
    mock_artifact_version = Mock()
    mock_artifact_version.id = PyUUID(artifact_version_uuid)

    # Mock the model version response for get_model_version
    mock_model_version = Mock()
    mock_model_version.id = PyUUID(model_version_uuid)
    mock_client.get_model_version.return_value = mock_model_version

    file_refs = {"files/artifacts/model": ["artifact/model.pkl"]}
    all_metadata: dict = {}

    with (
        patch(
            "neptune_exporter.loaders.zenml_loader.save_artifact",
        ) as mock_save_artifact,
        patch(
            "neptune_exporter.loaders.zenml_loader.link_artifact_version_to_model_version",
        ) as mock_link_artifact,
    ):
        mock_save_artifact.return_value = mock_artifact_version

        loader._upload_files_as_artifacts(
            file_refs=file_refs,
            files_directory=tmp_path,
            model_version_id=model_version_uuid,
            all_metadata=all_metadata,
        )

    # Verify save_artifact was called with correct arguments
    assert mock_save_artifact.call_count == 1
    save_call_kwargs = mock_save_artifact.call_args[1]
    assert save_call_kwargs["name"] == "neptune-artifacts-model"
    assert save_call_kwargs["data"] == test_file
    assert "neptune-import" in save_call_kwargs["tags"]
    assert (
        save_call_kwargs["user_metadata"]["neptune_attribute_path"] == "artifacts/model"
    )

    # Verify link_artifact_version_to_model_version was called with proper objects
    assert mock_link_artifact.call_count == 1
    link_call_kwargs = mock_link_artifact.call_args[1]
    assert link_call_kwargs["artifact_version"] == mock_artifact_version
    assert link_call_kwargs["model_version"] == mock_model_version

    # Verify artifact ID is stored in metadata (stored as string)
    assert (
        all_metadata["artifacts"]["artifacts"]["model"]["zenml_artifact_id"]
        == artifact_version_uuid
    )

    # Verify file paths are still stored as fallback
    assert all_metadata["files"]["artifacts"]["model"] == "artifact/model.pkl"


def test_upload_files_as_artifacts_skips_missing_files(tmp_path: Path) -> None:
    """Test _upload_files_as_artifacts skips files that don't exist."""
    loader, mock_client = _create_loader()

    # Don't create the file - it should be skipped
    file_refs = {"files/artifacts/missing": ["missing/file.pkl"]}
    all_metadata: dict = {}

    with patch(
        "neptune_exporter.loaders.zenml_loader.save_artifact",
    ) as mock_save_artifact:
        loader._upload_files_as_artifacts(
            file_refs=file_refs,
            files_directory=tmp_path,
            model_version_id="mv-456",
            all_metadata=all_metadata,
        )

    # save_artifact should not be called since file doesn't exist
    assert mock_save_artifact.call_count == 0

    # But file path should still be stored in metadata
    assert all_metadata["files"]["artifacts"]["missing"] == "missing/file.pkl"


def test_upload_artifact_to_zenml_handles_directory(tmp_path: Path) -> None:
    """Test _upload_artifact_to_zenml handles directory artifacts correctly."""
    from uuid import UUID as PyUUID

    loader, mock_client = _create_loader()

    # Create a directory with files
    test_dir = tmp_path / "model_dir"
    test_dir.mkdir()
    (test_dir / "weights.bin").write_text("weights")
    (test_dir / "config.json").write_text("{}")

    # Use valid UUIDs for all IDs
    artifact_version_uuid = "b1b2c3d4-e5f6-7890-abcd-ef1234567890"
    model_version_uuid = "669e2121-d812-4ce2-9738-6b6be3a004a3"

    mock_artifact_version = Mock()
    mock_artifact_version.id = PyUUID(artifact_version_uuid)

    # Mock the zen_store for linking
    mock_zen_store = Mock()
    mock_client.zen_store = mock_zen_store

    with patch(
        "neptune_exporter.loaders.zenml_loader.save_artifact",
    ) as mock_save_artifact:
        mock_save_artifact.return_value = mock_artifact_version

        result = loader._upload_artifact_to_zenml(
            local_path=test_dir,
            artifact_name="model-dir-artifact",
            model_version_id=model_version_uuid,
            neptune_attr_path="artifacts/model_dir",
        )

    assert result == artifact_version_uuid

    # Verify DirectoryMaterializer was used (check materializer argument)
    save_call_kwargs = mock_save_artifact.call_args[1]
    # The materializer should be DirectoryMaterializer for directories
    from neptune_exporter.loaders.zenml_loader import DirectoryMaterializer

    assert save_call_kwargs["materializer"] == DirectoryMaterializer


def test_upload_artifact_to_zenml_handles_single_file(tmp_path: Path) -> None:
    """Test _upload_artifact_to_zenml handles single file artifacts correctly."""
    from uuid import UUID as PyUUID

    loader, mock_client = _create_loader()

    # Create a single file
    test_file = tmp_path / "model.pkl"
    test_file.write_text("model content")

    # Use valid UUIDs for all IDs
    artifact_version_uuid = "c1b2c3d4-e5f6-7890-abcd-ef1234567890"
    model_version_uuid = "669e2121-d812-4ce2-9738-6b6be3a004a3"

    mock_artifact_version = Mock()
    mock_artifact_version.id = PyUUID(artifact_version_uuid)

    # Mock the zen_store for linking
    mock_zen_store = Mock()
    mock_client.zen_store = mock_zen_store

    with patch(
        "neptune_exporter.loaders.zenml_loader.save_artifact",
    ) as mock_save_artifact:
        mock_save_artifact.return_value = mock_artifact_version

        result = loader._upload_artifact_to_zenml(
            local_path=test_file,
            artifact_name="single-file-artifact",
            model_version_id=model_version_uuid,
            neptune_attr_path="artifacts/model.pkl",
        )

    assert result == artifact_version_uuid

    # Verify FileMaterializer was used for files
    save_call_kwargs = mock_save_artifact.call_args[1]
    from neptune_exporter.loaders.zenml_loader import FileMaterializer

    assert save_call_kwargs["materializer"] == FileMaterializer
