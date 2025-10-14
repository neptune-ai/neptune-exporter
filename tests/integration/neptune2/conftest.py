import os
import pathlib
import tempfile
from .data import TEST_DATA

import pytest

import neptune


@pytest.fixture(scope="session")
def api_token() -> str:
    api_token = os.getenv("NEPTUNE_E2E_API_TOKEN")
    if api_token is None:
        raise RuntimeError("NEPTUNE_E2E_API_TOKEN environment variable is not set")
    return api_token


@pytest.fixture(scope="session")
def project() -> str:
    project_identifier = os.getenv("NEPTUNE_E2E_PROJECT")
    if project_identifier is None:
        raise RuntimeError("NEPTUNE_E2E_PROJECT environment variable is not set")
    return project_identifier


@pytest.fixture(scope="session")
def test_runs(project, api_token) -> None:
    runs = {}

    for experiment in TEST_DATA:
        # Create new experiment with all data
        run = neptune.init_run(
            api_token=api_token,
            project=project,
            name=experiment.name,
            custom_run_id=experiment.run_id,
            mode="async",
            capture_stdout=False,
            capture_stderr=False,
            capture_hardware_metrics=False,
            git_ref=False,
        )

        run.assign(experiment.config)

        # for key, values in experiment.string_sets.items():
        #     run[key].add(values)

        # for path, series in experiment.float_series.items():
        #     run[path].extend(
        #         values=[value for _, value in series],
        #         steps=[step for step, _ in series],
        #         timestamps=[(TEST_NOW + timedelta(seconds=step)).timestamp() * 1000.0 for step, _ in series],
        #     )

        # for path, series in experiment.string_series.items():
        #     run[path].extend(
        #         values=[value for _, value in series],
        #         steps=[step for step, _ in series],
        #         timestamps=[(TEST_NOW + timedelta(seconds=step)).timestamp() * 1000.0 for step, _ in series],
        #     )

        # for path, value in experiment.files.items():
        # run[path].upload(value)

        # TODO: FileSeries supports only image files for now
        # for path, series in experiment.file_series.items():
        #     run[path].extend(
        #         values=[value for _, value in series],
        #         steps=[step for step, _ in series],
        #         timestamps=[(TEST_NOW + timedelta(seconds=step)).timestamp() * 1000.0 for step, _ in series],
        #     )

        runs[experiment.name] = run

    for run in runs.values():
        run.stop()

    return [run._sys_id for run in runs.values()]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)
