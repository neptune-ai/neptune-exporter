from datetime import timedelta
import os
import pathlib
import tempfile

from neptune_exporter.exporters.error_reporter import ErrorReporter
from .data import TEST_DATA, TEST_NOW

import pytest

import neptune
from neptune_exporter.exporters.neptune2 import Neptune2Exporter


@pytest.fixture(scope="session")
def api_token() -> str:
    api_token = os.getenv("NEPTUNE2_E2E_API_TOKEN")
    if api_token is None:
        raise RuntimeError("NEPTUNE2_E2E_API_TOKEN environment variable is not set")
    return api_token


@pytest.fixture(scope="session")
def project() -> str:
    project_identifier = os.getenv("NEPTUNE2_E2E_PROJECT")
    if project_identifier is None:
        raise RuntimeError("NEPTUNE2_E2E_PROJECT environment variable is not set")
    return project_identifier


@pytest.fixture(scope="session")
def test_runs(project, api_token) -> None:
    runs = {}
    run_data = {}

    for experiment in TEST_DATA:
        # Create new experiment with all data
        run = neptune.init_run(
            api_token=api_token,
            project=project,
            name=experiment.name,
            custom_run_id=experiment.run_id,
            mode="sync",
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            git_ref=False,
        )

        run.assign(experiment.config)

        for key, values in experiment.string_sets.items():
            run[key].add(values)

        for path, series in experiment.float_series.items():
            run[path].extend(
                values=[value for _, value in series],
                steps=[step for step, _ in series],
                timestamps=[
                    (TEST_NOW + timedelta(seconds=step)).timestamp()
                    for step, _ in series
                ],
            )

        for path, series in experiment.string_series.items():
            run[path].extend(
                values=[value for _, value in series],
                steps=[step for step, _ in series],
                timestamps=[
                    (TEST_NOW + timedelta(seconds=step)).timestamp()
                    for step, _ in series
                ],
            )

        for path, value in experiment.files.items():
            run[path].upload(value)

        for path, series in experiment.file_series.items():
            run[path].extend(
                values=[value for _, value in series],
                steps=[step for step, _ in series],
                timestamps=[
                    (TEST_NOW + timedelta(seconds=step)).timestamp()
                    for step, _ in series
                ],
            )

        for path, value in experiment.file_sets.items():
            run[path].upload_files(value)

        for path, value in experiment.artifacts.items():
            run[path].track_files(value)

        runs[run._sys_id] = run
        run_data[run._sys_id] = experiment

    for run in runs.values():
        run.stop()

    return run_data


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def exporter(api_token, temp_dir):
    """Fixture providing a Neptune2Exporter instance."""
    return Neptune2Exporter(
        error_reporter=ErrorReporter(path=temp_dir / "errors.jsonl"),
        api_token=api_token,
    )
