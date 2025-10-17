from typing import Generator
import pyarrow as pa
from neptune_exporter import model
from neptune_exporter.exporters.neptune2 import Neptune2Exporter
from .data import TEST_DATA


def test_neptune2_download_parameters_empty(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)

    parameters = _to_table(
        exporter.download_parameters(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )

    assert parameters.num_rows == 0


def test_neptune2_download_parameters(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)

    parameters = _to_table(
        exporter.download_parameters(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {run_id for run_id in test_runs}
    actual_run_ids = set(parameters.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected parameter paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.config.keys():
            expected_paths.add(path)

    actual_paths = set(parameters.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_metrics_empty(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)

    metrics = _to_table(
        exporter.download_metrics(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )

    assert metrics.num_rows == 0


def test_neptune2_download_metrics(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)

    metrics = _to_table(
        exporter.download_metrics(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {run_id for run_id in test_runs}
    actual_run_ids = set(metrics.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected metric paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.float_series.keys():
            expected_paths.add(path)

    actual_paths = set(metrics.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_series_empty(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)
    series = _to_table(
        exporter.download_series(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )
    assert series.num_rows == 0


def test_neptune2_download_series(api_token, project, test_runs):
    exporter = Neptune2Exporter(api_token=api_token)
    series = _to_table(
        exporter.download_series(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {run_id for run_id in test_runs}
    actual_run_ids = set(series.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected series paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.string_series.keys():
            expected_paths.add(path)

    actual_paths = set(series.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_files_empty(api_token, project, test_runs, temp_dir):
    exporter = Neptune2Exporter(api_token=api_token)
    files = _to_table(
        exporter.download_files(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
            destination=temp_dir,
        )
    )
    assert files.num_rows == 0


def test_neptune2_download_files(api_token, project, test_runs, temp_dir):
    exporter = Neptune2Exporter(api_token=api_token)
    files = _to_table(
        exporter.download_files(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
            destination=temp_dir,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {run_id for run_id in test_runs}
    actual_run_ids = set(files.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected file paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.files.keys():
            expected_paths.add(path)
        for path in item.file_series.keys():
            expected_paths.add(path)
        for path in item.file_sets.keys():
            expected_paths.add(path)

    actual_paths = set(files.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def _to_table(parameters: Generator[pa.RecordBatch, None, None]) -> pa.Table:
    return pa.Table.from_batches(parameters, schema=model.SCHEMA)
