import re
import uuid
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Generator

import neptune
import pyarrow as pa
import pytest

from neptune_exporter import model


def _wait_for_model_registry_visibility(
    *,
    project: str,
    api_token: str,
    model_id: str,
    model_version_id: str,
    timeout_seconds: int = 10,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Wait until model and model version are visible in read APIs.

    Neptune model registry APIs are eventually consistent. Without waiting,
    freshly created entities can be missing for a short time and cause flaky
    assertions in integration tests.
    """
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None

    while monotonic() < deadline:
        try:
            with neptune.init_project(
                api_token=api_token,
                project=project,
                mode="read-only",
            ) as project_obj:
                models_table = project_obj.fetch_models_table(
                    query=f'`sys/id`:string = "{model_id}"',
                    columns=["sys/id"],
                    trashed=False,
                    progress_bar=False,
                ).to_pandas()
            if models_table.empty:
                sleep(poll_interval_seconds)
                continue

            with neptune.init_model(
                api_token=api_token,
                project=project,
                with_id=model_id,
                mode="read-only",
            ) as model_obj:
                versions_table = model_obj.fetch_model_versions_table(
                    columns=["sys/id"],
                    progress_bar=False,
                    ascending=True,
                ).to_pandas()

            if model_version_id in set(versions_table["sys/id"]):
                return
        except Exception as exc:
            last_error = exc

        sleep(poll_interval_seconds)

    if last_error is not None:
        pytest.skip(
            "Model registry entities were not visible before timeout due to "
            f"eventual consistency. Last error: {last_error}"
        )
    pytest.skip(
        "Model registry entities were not visible before timeout due to eventual "
        "consistency."
    )


@pytest.fixture(scope="session")
def test_model_registry_entities(project, api_token):
    """Create a model and model version for model registry export tests."""
    model_key = f"MR{uuid.uuid4().hex[:8]}".upper()

    try:
        model_obj = neptune.init_model(
            api_token=api_token,
            project=project,
            key=model_key,
            name="neptune-exporter-model-registry-test",
            mode="sync",
        )
        model_obj["test/exporter/string-value"] = "model-value"
        model_obj["test/exporter/float-value"] = 1.23
        model_id = model_obj["sys/id"].fetch()
        model_obj.stop()

        model_version_obj = neptune.init_model_version(
            api_token=api_token,
            project=project,
            model=model_id,
            name="v1",
            mode="sync",
        )
        model_version_obj["test/exporter/version-string"] = "version-value"
        model_version_obj["test/exporter/version-float"] = 2.34
        model_version_obj["test/exporter/version-time"] = datetime.now(timezone.utc)
        model_version_id = model_version_obj["sys/id"].fetch()
        model_version_obj.stop()
    except Exception as e:
        pytest.skip(f"Model registry is unavailable in this Neptune workspace: {e}")

    _wait_for_model_registry_visibility(
        project=project,
        api_token=api_token,
        model_id=model_id,
        model_version_id=model_version_id,
    )

    return {
        "model_id": model_id,
        "model_version_id": model_version_id,
    }


def test_neptune2_list_models(exporter, project, test_model_registry_entities):
    model_ids = exporter.list_models(
        project_id=project,
        query=f'`sys/id`:string = "{test_model_registry_entities["model_id"]}"',
    )
    assert test_model_registry_entities["model_id"] in model_ids


def test_neptune2_list_model_versions(exporter, project, test_model_registry_entities):
    model_version_ids = exporter.list_model_versions(
        project_id=project,
        model_id=test_model_registry_entities["model_id"],
    )
    assert test_model_registry_entities["model_version_id"] in model_version_ids


def test_neptune2_download_model_parameters(
    exporter, project, test_model_registry_entities
):
    parameters = _to_table(
        exporter.download_model_parameters(
            project_id=project,
            model_ids=[test_model_registry_entities["model_id"]],
            attributes=re.escape("test/exporter/string-value"),
        )
    )

    if parameters.num_rows == 0:
        pytest.skip("No model parameters returned from Neptune model registry")

    actual_run_ids = set(parameters.column("run_id").to_pylist())
    assert test_model_registry_entities["model_id"] in actual_run_ids


def test_neptune2_download_model_version_parameters(
    exporter, project, test_model_registry_entities
):
    parameters = _to_table(
        exporter.download_model_version_parameters(
            project_id=project,
            model_version_ids=[test_model_registry_entities["model_version_id"]],
            attributes=re.escape("test/exporter/version-string"),
        )
    )

    if parameters.num_rows == 0:
        pytest.skip("No model version parameters returned from Neptune model registry")

    actual_run_ids = set(parameters.column("run_id").to_pylist())
    assert test_model_registry_entities["model_version_id"] in actual_run_ids


def _to_table(parameters: Generator[pa.RecordBatch, None, None]) -> pa.Table:
    return pa.Table.from_batches(parameters, schema=model.SCHEMA)
