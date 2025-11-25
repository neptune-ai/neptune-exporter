import os
import pathlib
import tempfile
import time
from datetime import timedelta

import pytest
from neptune_api import AuthenticatedClient
from neptune_api.credentials import Credentials
from neptune_scale import Run

from neptune_query.internal import identifiers
from neptune_query.internal.api_utils import (
    create_auth_api_client,
    get_config_and_token_urls,
)
from neptune_query.internal.context import (
    set_project,
)
from neptune_query.internal.filters import _Filter
from neptune_query.internal.retrieval.search import fetch_experiment_sys_attrs
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from .data import (
    TEST_DATA,
    TEST_NOW,
)


@pytest.fixture(scope="session")
def api_token() -> str:
    api_token = os.getenv("NEPTUNE3_E2E_API_TOKEN")
    if api_token is None:
        raise RuntimeError("NEPTUNE3_E2E_API_TOKEN environment variable is not set")
    return api_token


@pytest.fixture(scope="session")
def client(api_token) -> AuthenticatedClient:
    credentials = Credentials.from_api_key(api_key=api_token)
    config, token_urls = get_config_and_token_urls(
        credentials=credentials, proxies=None
    )
    client = create_auth_api_client(
        credentials=credentials,
        config=config,
        token_refreshing_urls=token_urls,
        proxies=None,
    )

    return client


@pytest.fixture(scope="session")
def project() -> str:
    project_identifier = os.getenv("NEPTUNE3_E2E_PROJECT")
    if project_identifier is None:
        raise RuntimeError("NEPTUNE3_E2E_PROJECT environment variable is not set")
    return project_identifier


@pytest.fixture(autouse=True)
def context(project) -> None:
    set_project(project)


@pytest.fixture(scope="session")
def test_runs(project, api_token, client) -> None:
    runs = {}

    for experiment in TEST_DATA:
        # Create new experiment with all data
        run = Run(
            api_token=api_token,
            project=project,
            run_id=experiment.run_id,
            experiment_name=experiment.name,
            source_tracking_config=None,
            enable_console_log_capture=False,
        )

        run.log_configs(experiment.config)

        run.log(tags_add=experiment.string_sets)

        for path, series in experiment.float_series.items():
            for step, value in series:
                run.log_metrics(
                    data={path: value},
                    step=step,
                    timestamp=TEST_NOW + timedelta(seconds=step),
                )

        for path, series in experiment.string_series.items():
            for step, value in series:
                run.log_string_series(
                    data={path: value},
                    step=step,
                    timestamp=TEST_NOW + timedelta(seconds=step),
                )

        for path, series in experiment.histogram_series.items():
            for step, value in series:
                run.log_histograms(
                    histograms={path: value},
                    step=step,
                    timestamp=TEST_NOW + timedelta(seconds=step),
                )

        run.assign_files(experiment.files)

        for path, series in experiment.file_series.items():
            for step, value in series:
                run.log_files(
                    files={path: value},
                    step=step,
                    timestamp=TEST_NOW + timedelta(seconds=step),
                )

        runs[experiment.name] = run

    for run in runs.values():
        run.close()

    timeout = 15  # seconds
    poll_interval = 0.2  # seconds
    start_time = time.monotonic()
    while True:
        existing = next(
            fetch_experiment_sys_attrs(
                client,
                identifiers.ProjectIdentifier(project),
                _Filter.any(
                    [_Filter.name_eq(experiment.name) for experiment in TEST_DATA]
                ),
            )
        )
        if len(existing.items) == len(TEST_DATA):
            return [item.run_id for item in TEST_DATA]
        if time.monotonic() - start_time > timeout:
            break
        time.sleep(poll_interval)

    raise RuntimeError("Experiments did not appear in the system in time")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def exporter(api_token, temp_dir):
    """Fixture providing a Neptune3Exporter instance."""
    return Neptune3Exporter(
        error_reporter=ErrorReporter(path=temp_dir / "errors.jsonl"),
        api_token=api_token,
    )
