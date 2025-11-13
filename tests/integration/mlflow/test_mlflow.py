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

import re

from mlflow.tracking import MlflowClient

from .data import TEST_PROJECT_ID, TEST_RUNS


def _sanitize_attribute_name(attribute_path: str) -> str:
    """Sanitize attribute path to match MLflowLoader's sanitization."""
    # Match MLflowLoader._sanitize_attribute_name logic
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s/]", "_", attribute_path)
    if len(sanitized) > 250:
        sanitized = sanitized[:250]
    return sanitized


def _list_all_artifacts(
    mlflow_client: MlflowClient, run_id: str, path: str = ""
) -> set[str]:
    """Recursively list all artifact paths in an MLflow run."""
    artifacts = mlflow_client.list_artifacts(run_id, path=path)
    all_paths = set()

    for artifact in artifacts:
        artifact_path = artifact.path
        all_paths.add(artifact_path)

        # If it's a directory (is_dir is True), recursively list its contents
        if artifact.is_dir:
            all_paths.update(_list_all_artifacts(mlflow_client, run_id, artifact_path))

    return all_paths


def test_mlflow_load_all_runs(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test loading all runs from parquet files to MLflow."""
    # Load all data to MLflow
    loader_manager.load(project_ids=None, runs=None)

    # Verify experiments were created
    experiments = mlflow_client.search_experiments()
    assert len(experiments) > 0

    # Verify all test runs were loaded
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )
    assert len(all_runs) >= len(TEST_RUNS)

    # Verify each run has data
    for run in all_runs:
        # MLflow status is a string, not an enum
        assert run.info.status == "FINISHED"
        # Verify run has parameters (configs)
        assert len(run.data.params) > 0
        # Verify run has metrics
        assert len(run.data.metrics) > 0


def test_mlflow_load_specific_project(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test loading specific project to MLflow."""
    # Load specific project
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Verify experiments were created
    experiments = mlflow_client.search_experiments()
    assert len(experiments) > 0

    # Verify runs exist
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )
    assert len(all_runs) > 0


def test_mlflow_load_specific_runs(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test loading specific runs to MLflow."""
    # Load only first run
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=[TEST_RUNS[0].run_id])

    # Verify one new run was added
    experiments_after = mlflow_client.search_experiments()
    found_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments_after],
        filter_string=f"attributes.run_name = '{TEST_RUNS[0].run_id}'",
        max_results=100,
    )
    assert len(found_runs) == 1
    assert found_runs[0].info.run_name == TEST_RUNS[0].run_id


def test_mlflow_parameters_loaded(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that parameters (configs) are correctly loaded to MLflow."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs
    experiments = mlflow_client.search_experiments()
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )

    # Create a mapping of run name to test data (MLflow generates its own run IDs)
    test_data_by_run_name = {run_data.run_id: run_data for run_data in TEST_RUNS}

    # Verify parameters for each run match test data
    for run in all_runs:
        # Match by run name instead of run_id (MLflow generates its own IDs)
        run_name = run.info.run_name
        test_data = test_data_by_run_name[run_name]
        params = run.data.params

        # Helper to find parameter by partial name match (MLflow may sanitize)
        def find_param(key_pattern: str) -> str:
            key_lower = key_pattern.lower()
            for key in params:
                if key_lower in key.lower() or key.lower() in key_lower:
                    return key
            raise ValueError(f"Parameter {key_pattern} not found in {params}")

        # Verify int parameter
        int_key = find_param("test/param/int")
        # MLflow may store ints as "42" or "42.0", so compare numerically
        assert float(params[int_key]) == float(test_data.params["test/param/int"])

        # Verify float parameter
        float_key = find_param("test/param/float")
        # MLflow stores floats as strings, compare with tolerance
        expected = test_data.params["test/param/float"]
        actual = float(params[float_key])
        assert abs(actual - expected) < 0.001

        # Verify string parameter
        string_key = find_param("test/param/string")
        assert params[string_key] == str(test_data.params["test/param/string"])

        # Verify bool parameter
        bool_key = find_param("test/param/bool")
        # MLflow stores bools as strings
        assert params[bool_key].lower() in ("true", "false")
        assert (params[bool_key].lower() == "true") == test_data.params[
            "test/param/bool"
        ]

        # Verify datetime parameter (MLflow may convert to string)
        datetime_key = find_param("test/param/datetime")
        # Just verify it exists and is a string (datetime conversion format may vary)
        assert isinstance(params[datetime_key], str)
        assert len(params[datetime_key]) > 0

        # Verify string_set parameter (MLflow may convert to string representation)
        string_set_key = find_param("test/param/string_set")
        # MLflow may store string_set as a JSON string or comma-separated
        value = params[string_set_key]
        assert isinstance(value, str)
        assert len(value) > 0


def test_mlflow_metrics_loaded(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that metrics are correctly loaded to MLflow."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs
    experiments = mlflow_client.search_experiments()
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )

    # Create a mapping of run name to test data (MLflow generates its own run IDs)
    test_data_by_run_name = {run_data.run_id: run_data for run_data in TEST_RUNS}

    # Verify metrics for each run match test data
    for run in all_runs:
        # Match by run name instead of run_id (MLflow generates its own IDs)
        run_name = run.info.run_name
        test_data = test_data_by_run_name[run_name]
        assert len(run.data.metrics) > 0

        # Verify each metric from test data
        for metric_attr_path, expected_series in test_data.metrics.items():
            # Find metric by name (MLflow may sanitize the path)
            metric_name = None
            for key in run.data.metrics:
                if (
                    metric_attr_path.lower() in key.lower()
                    or key.lower() in metric_attr_path.lower()
                ):
                    metric_name = key
                    break
            if not metric_name:
                raise ValueError(
                    f"Metric {metric_attr_path} not found in {run.data.metrics}"
                )

            metric_history = mlflow_client.get_metric_history(
                run.info.run_id, metric_name
            )
            assert len(metric_history) > 0

            # Verify steps are integers and non-negative
            for metric in metric_history:
                assert isinstance(metric.step, int)
                assert metric.step >= 0

            # Build expected and actual sequences sorted by step
            expected_by_step = {int(step): value for step, value in expected_series}
            actual_by_step = {metric.step: metric.value for metric in metric_history}

            # Round to 3 decimal places for comparison (handles floating point tolerance)
            # This allows pytest to show a proper diff when lists don't match
            expected_rounded = [round(v, 3) for v in expected_by_step.values()]
            actual_rounded = [round(v, 3) for v in actual_by_step.values()]
            assert expected_rounded == actual_rounded


def test_mlflow_artifacts_loaded(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that artifacts (files) are correctly loaded to MLflow."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs
    experiments = mlflow_client.search_experiments()
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )

    # Create a mapping of run name to test data (MLflow generates its own run IDs)
    test_data_by_run_name = {run_data.run_id: run_data for run_data in TEST_RUNS}

    # Verify artifacts for each run
    runs_with_artifacts = 0
    for run in all_runs:
        # Match by run name instead of run_id (MLflow generates its own IDs)
        run_name = run.info.run_name
        if run_name not in test_data_by_run_name:
            continue

        test_data = test_data_by_run_name[run_name]
        artifact_paths = _list_all_artifacts(mlflow_client, run.info.run_id)

        # Collect expected artifact attribute paths (MLflow stores them under sanitized paths)
        # MLflowLoader uses attribute_path as artifact_path, so artifacts are under subdirectories
        expected_artifact_paths = set()
        for attr_path in test_data.files.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            expected_artifact_paths.add(sanitized)
        for attr_path in test_data.artifacts.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            expected_artifact_paths.add(sanitized)
        for attr_path in test_data.file_series.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            expected_artifact_paths.add(sanitized)
        for attr_path in test_data.file_sets.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            expected_artifact_paths.add(sanitized)

        if expected_artifact_paths:
            runs_with_artifacts += 1

            # For each expected path, verify there's at least one artifact that matches it
            # Artifacts can be stored as:
            # - Direct match: `sanitized_path` (directory)
            # - File in directory: `sanitized_path/filename`
            # - For file_series: `sanitized_path/step_X/filename`
            missing_paths = []
            for expected_path in expected_artifact_paths:
                # Check if any artifact path starts with the expected path or is the expected path
                found = any(
                    artifact_path == expected_path
                    or artifact_path.startswith(f"{expected_path}/")
                    for artifact_path in artifact_paths
                )
                if not found:
                    missing_paths.append(expected_path)

            assert len(missing_paths) == 0, (
                f"Missing artifact paths for run {run_name}: {missing_paths}. "
                f"Expected: {expected_artifact_paths}, "
                f"Found: {artifact_paths}"
            )

    # At least one run should have artifacts
    assert runs_with_artifacts > 0


def test_mlflow_string_series_loaded(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that string series are correctly loaded to MLflow."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs
    experiments = mlflow_client.search_experiments()
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )

    # Create a mapping of run name to test data (MLflow generates its own run IDs)
    test_data_by_run_name = {run_data.run_id: run_data for run_data in TEST_RUNS}

    # Verify string series are logged as artifacts or text files
    runs_with_string_series = 0
    for run in all_runs:
        # Match by run name instead of run_id (MLflow generates its own IDs)
        run_name = run.info.run_name
        test_data = test_data_by_run_name[run_name]

        # Recursively list all artifacts to find nested paths
        artifact_paths = _list_all_artifacts(mlflow_client, run.info.run_id)

        # Collect expected artifact paths (string series are logged as {sanitized_path}/series.txt)
        expected_artifact_paths = set()
        for attr_path in test_data.string_series.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            # String series are logged at {sanitized_path}/series.txt
            expected_artifact_paths.add(f"{sanitized}/series.txt")

        if expected_artifact_paths:
            runs_with_string_series += 1

            # For each expected path, verify it exists
            # String series artifacts are stored as {sanitized_path}/series.txt
            missing_paths = []
            for expected_path in expected_artifact_paths:
                found = expected_path in artifact_paths
                if not found:
                    missing_paths.append(expected_path)

            assert len(missing_paths) == 0, (
                f"Missing string series artifact paths for run {run_name}: {missing_paths}. "
                f"Expected: {expected_artifact_paths}, "
                f"Found: {artifact_paths}"
            )

    # At least one run should have string series
    assert runs_with_string_series > 0


def test_mlflow_histogram_series_loaded(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that histogram series are correctly loaded to MLflow."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs
    experiments = mlflow_client.search_experiments()
    all_runs = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )

    # Create a mapping of run name to test data (MLflow generates its own run IDs)
    test_data_by_run_name = {run_data.run_id: run_data for run_data in TEST_RUNS}

    # Verify histogram series are logged as artifacts
    runs_with_histogram_series = 0
    for run in all_runs:
        # Match by run name instead of run_id (MLflow generates its own IDs)
        run_name = run.info.run_name
        test_data = test_data_by_run_name[run_name]

        # Recursively list all artifacts to find nested paths
        artifact_paths = _list_all_artifacts(mlflow_client, run.info.run_id)

        # Collect expected artifact paths (histogram series are logged as {sanitized_path}/histograms.json)
        expected_artifact_paths = set()
        for attr_path in test_data.histogram_series.keys():
            sanitized = _sanitize_attribute_name(attr_path)
            # Histogram series are logged at {sanitized_path}/histograms.json
            expected_artifact_paths.add(f"{sanitized}/histograms.json")

        if expected_artifact_paths:
            runs_with_histogram_series += 1

            # For each expected path, verify it exists
            # Histogram series artifacts are stored as {sanitized_path}/histograms.json
            missing_paths = []
            for expected_path in expected_artifact_paths:
                found = expected_path in artifact_paths
                if not found:
                    missing_paths.append(expected_path)

            assert len(missing_paths) == 0, (
                f"Missing histogram series artifact paths for run {run_name}: {missing_paths}. "
                f"Expected: {expected_artifact_paths}, "
                f"Found: {artifact_paths}"
            )

    # At least one run should have histogram series
    assert runs_with_histogram_series > 0


def test_mlflow_resumable_loading(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that loading is resumable - existing runs are found and skipped."""
    # Load data first time
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs after first load
    experiments = mlflow_client.search_experiments()
    all_runs_first = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments], max_results=100
    )
    runs_count_first = len(all_runs_first)

    # Load again - should find existing runs and skip them
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get all runs after second load
    experiments_after = mlflow_client.search_experiments()
    all_runs_second = mlflow_client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments_after], max_results=100
    )
    runs_count_second = len(all_runs_second)

    # Should have the same number of runs (no duplicates created)
    assert runs_count_second == runs_count_first

    # Verify run names match (runs were found, not recreated)
    first_run_names = {run.info.run_name for run in all_runs_first}
    second_run_names = {run.info.run_name for run in all_runs_second}
    assert first_run_names == second_run_names


def test_mlflow_find_run_by_name(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that find_run can locate runs by name."""
    # Load data
    loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)

    # Get experiment
    experiments = mlflow_client.search_experiments()
    assert len(experiments) > 0
    experiment_id = experiments[0].experiment_id

    # Get a run to test with
    all_runs = mlflow_client.search_runs(experiment_ids=[experiment_id], max_results=1)
    assert len(all_runs) > 0
    test_run = all_runs[0]
    test_run_name = test_run.info.run_name

    # Test find_run
    loader = loader_manager._data_loader
    found_run_id = loader.find_run(
        project_id=TEST_PROJECT_ID,
        run_name=test_run_name,
        experiment_id=experiment_id,
    )

    # Should find the run
    assert found_run_id is not None
    assert found_run_id == test_run.info.run_id


def test_mlflow_find_run_not_exists(
    loader_manager, mlflow_client: MlflowClient, test_data_dir
):
    """Test that find_run returns None for non-existent runs."""
    # Get experiment
    experiments = mlflow_client.search_experiments()
    if len(experiments) == 0:
        # Create an experiment if none exists
        loader_manager.load(project_ids=[TEST_PROJECT_ID], runs=None)
        experiments = mlflow_client.search_experiments()

    assert len(experiments) > 0
    experiment_id = experiments[0].experiment_id

    # Test find_run with non-existent name
    loader = loader_manager._data_loader
    found_run_id = loader.find_run(
        project_id=TEST_PROJECT_ID,
        run_name="non-existent-run-name-12345",
        experiment_id=experiment_id,
    )

    # Should return None
    assert found_run_id is None
