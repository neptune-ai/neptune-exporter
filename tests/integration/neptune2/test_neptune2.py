import re
from typing import Generator
import json
import pyarrow as pa
import pyarrow.compute as pc
from neptune_exporter import model
from .data import TEST_DATA


def test_neptune2_list_runs(exporter, project, test_runs):
    runs = exporter.list_runs(
        project_id=project, runs="|".join(re.escape(run_id) for run_id in test_runs)
    )

    assert len(runs) == len(test_runs)


def test_neptune2_download_parameters_empty(exporter, project, test_runs):
    parameters = _to_table(
        exporter.download_parameters(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )

    assert parameters.num_rows == 0


def test_neptune2_download_parameters(exporter, project, test_runs):
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
        for path in item.string_sets.keys():
            expected_paths.add(path)

    actual_paths = set(parameters.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_metrics_empty(exporter, project, test_runs):
    metrics = _to_table(
        exporter.download_metrics(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )

    assert metrics.num_rows == 0


def test_neptune2_download_metrics(exporter, project, test_runs):
    metrics = _to_table(
        exporter.download_metrics(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {
        run_id for run_id in test_runs if test_runs[run_id].float_series
    }
    actual_run_ids = set(metrics.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected metric paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.float_series.keys():
            expected_paths.add(path)

    actual_paths = set(metrics.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_series_empty(exporter, project, test_runs):
    series = _to_table(
        exporter.download_series(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
        )
    )
    assert series.num_rows == 0


def test_neptune2_download_series(exporter, project, test_runs):
    series = _to_table(
        exporter.download_series(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {
        run_id for run_id in test_runs if test_runs[run_id].string_series
    }
    actual_run_ids = set(series.column("run_id").to_pylist())
    assert expected_run_ids == actual_run_ids

    # Verify all expected series paths are present
    expected_paths = set()
    for item in TEST_DATA:
        for path in item.string_series.keys():
            expected_paths.add(path)

    actual_paths = set(series.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)


def test_neptune2_download_files_empty(exporter, project, test_runs, temp_dir):
    files = _to_table(
        exporter.download_files(
            project_id=project,
            run_ids=["does-not-exist"],
            attributes=None,
            destination=temp_dir,
        )
    )
    assert files.num_rows == 0


def test_neptune2_download_files(exporter, project, test_runs, temp_dir):
    files = _to_table(
        exporter.download_files(
            project_id=project,
            run_ids=test_runs,
            attributes=None,
            destination=temp_dir,
        )
    )

    # Verify we have data for all test runs
    expected_run_ids = {run_id for run_id in test_runs if test_runs[run_id].files}
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
        for path in item.artifacts.keys():
            expected_paths.add(path)

    actual_paths = set(files.column("attribute_path").to_pylist())
    assert expected_paths.issubset(actual_paths)

    # Verify artifact JSON files are created and contain valid data
    mask = pc.equal(files["attribute_type"], "artifact")
    artifact_data = files.filter(mask)
    for row in artifact_data.to_pylist():
        file_path = temp_dir / row["file_value"]["path"]
        assert file_path.exists(), f"Artifact JSON file not found: {file_path}"
        assert file_path.name == "files_list.json", (
            f"Expected files_list.json, got {file_path.name}"
        )

        # Verify JSON file is valid and contains expected structure
        with open(file_path, "r") as f:
            artifact_list = json.load(f)
        assert isinstance(artifact_list, list), "Artifact list should be a list"
        # Each item should have the expected structure from ArtifactFileData.to_dto()
        for item in artifact_list:
            assert "filePath" in item, "Artifact file data should have filePath"
            assert "fileHash" in item, "Artifact file data should have fileHash"
            assert "type" in item, "Artifact file data should have type"
            assert "metadata" in item, "Artifact file data should have metadata"


def test_neptune2_list_runs_with_regex_filter(exporter, project, test_runs):
    """Test list_runs filters runs using regex pattern."""
    # Test with pattern that matches all (should return at least test_runs)
    all_matching = exporter.list_runs(project_id=project, runs=".*")
    assert len(all_matching) >= len(test_runs)
    assert set(test_runs).issubset(set(all_matching))

    first_run_id = next(iter(test_runs.keys()))
    prefix_pattern = f".*{first_run_id[4:]}$"
    prefix_matching = exporter.list_runs(project_id=project, runs=prefix_pattern)
    assert first_run_id in prefix_matching

    # Test with pattern that matches none
    no_matching = exporter.list_runs(project_id=project, runs="^NONEXISTENT-")
    assert len(no_matching) == 0


def _to_table(parameters: Generator[pa.RecordBatch, None, None]) -> pa.Table:
    return pa.Table.from_batches(parameters, schema=model.SCHEMA)
