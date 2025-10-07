from typing import Generator
import pyarrow as pa
from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from .data import TEST_DATA


def test_neptune3_download_parameters_empty(api_token, project, test_runs):
    exporter = Neptune3Exporter(api_token=api_token)

    parameters = _to_table(
        exporter.download_parameters(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )

    assert parameters.num_rows == 0


def test_neptune3_download_parameters(api_token, project, test_runs):
    exporter = Neptune3Exporter(api_token=api_token)

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


def test_neptune3_download_metrics(api_token, project, test_runs):
    exporter = Neptune3Exporter(api_token=api_token)

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


def _to_table(parameters: Generator[pa.RecordBatch, None, None]) -> pa.Table:
    return pa.Table.from_batches(parameters)
