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

from collections import defaultdict
import json
import pathlib
import tempfile
import shutil
from decimal import Decimal
from datetime import datetime
from typing import Any

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient

from neptune_exporter import model
from neptune_exporter.loaders.mlflow_loader import MLflowLoader
from neptune_exporter.loader_manager import LoaderManager
from neptune_exporter.storage.parquet_reader import ParquetReader
from .data import TEST_PROJECT_ID, TEST_RUNS, TEST_NOW
from neptune_exporter.utils import sanitize_path_part


def _create_base_row(
    run_id: str,
    attr_path: str,
    attr_type: str,
    step: Decimal | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create a base row with all fields initialized to None."""
    return {
        "project_id": TEST_PROJECT_ID,
        "run_id": run_id,
        "attribute_path": attr_path,
        "attribute_type": attr_type,
        "step": step,
        "timestamp": timestamp,
        "int_value": None,
        "float_value": None,
        "string_value": None,
        "bool_value": None,
        "datetime_value": None,
        "string_set_value": None,
        "file_value": None,
        "histogram_value": None,
    }


def _create_series_rows(
    run_id: str,
    attr_path: str,
    attr_type: str,
    series: list[tuple[Decimal, Any]],
    value_field: str,
    value_extractor: Any = lambda x: x,
) -> list[dict[str, Any]]:
    """Create rows for a series (metrics, string_series, histogram_series, file_series)."""
    rows = []
    for step, value in series:
        row = _create_base_row(
            run_id=run_id,
            attr_path=attr_path,
            attr_type=attr_type,
            step=step,
            timestamp=TEST_NOW,
        )
        row[value_field] = value_extractor(value)
        rows.append(row)
    return rows


@pytest.fixture(scope="session")
def mlflow_tracking_uri() -> str:
    """Create a temporary directory for MLflow tracking."""
    temp_dir = tempfile.mkdtemp(prefix="mlflow_tests_")
    tracking_uri = f"file://{temp_dir}"
    yield tracking_uri
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mlflow_client(mlflow_tracking_uri: str) -> MlflowClient:
    """Create MLflow client for verification."""
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    """Create parquet files with test data."""
    temp_dir = tempfile.mkdtemp(prefix="mlflow_test_data_")
    data_path = pathlib.Path(temp_dir) / "data"
    files_path = pathlib.Path(temp_dir) / "files"
    data_path.mkdir(parents=True)
    files_path.mkdir(parents=True)

    # Create project directory
    project_dir = data_path / sanitize_path_part(TEST_PROJECT_ID)
    project_dir.mkdir(parents=True)
    project_files_dir = files_path / sanitize_path_part(TEST_PROJECT_ID)
    project_files_dir.mkdir(parents=True)

    # Prepare test data from data.py
    all_rows = []

    for run_data in TEST_RUNS:
        # Parameters (configs)
        for attr_path, value in run_data.params.items():
            row = _create_base_row(
                run_id=run_data.run_id,
                attr_path=attr_path,
                attr_type="string",
            )
            # Set the appropriate value field based on type
            if isinstance(value, bool):
                row["attribute_type"] = "bool"
                row["bool_value"] = value
            elif isinstance(value, int):
                row["attribute_type"] = "int"
                row["int_value"] = value
            elif isinstance(value, float):
                row["attribute_type"] = "float"
                row["float_value"] = value
            elif isinstance(value, datetime):
                row["attribute_type"] = "datetime"
                row["datetime_value"] = value
            elif isinstance(value, list):
                # string_set is a list of strings
                row["attribute_type"] = "string_set"
                row["string_set_value"] = value
            else:
                row["string_value"] = str(value)
            all_rows.append(row)

        # Metrics (float series)
        for attr_path, series in run_data.metrics.items():
            all_rows.extend(
                _create_series_rows(
                    run_id=run_data.run_id,
                    attr_path=attr_path,
                    attr_type="float_series",
                    series=series,
                    value_field="float_value",
                )
            )

        # String series
        for attr_path, series in run_data.string_series.items():
            all_rows.extend(
                _create_series_rows(
                    run_id=run_data.run_id,
                    attr_path=attr_path,
                    attr_type="string_series",
                    series=series,
                    value_field="string_value",
                )
            )

        # Histogram series
        for attr_path, series in run_data.histogram_series.items():
            all_rows.extend(
                _create_series_rows(
                    run_id=run_data.run_id,
                    attr_path=attr_path,
                    attr_type="histogram_series",
                    series=series,
                    value_field="histogram_value",
                )
            )

        # Files
        for attr_path, filename in run_data.files.items():
            test_file_path = project_files_dir / filename
            test_file_path.write_text(f"Test file content for {run_data.run_id}")
            row = _create_base_row(
                run_id=run_data.run_id,
                attr_path=attr_path,
                attr_type="file",
            )
            row["file_value"] = {"path": filename}
            all_rows.append(row)

        # Artifacts
        for attr_path, filename in run_data.artifacts.items():
            test_file_path = project_files_dir / filename
            test_file_path.parent.mkdir(parents=True, exist_ok=True)
            test_file_path.write_text(
                json.dumps(
                    [
                        {
                            "filePath": f"test_artifact_{run_data.run_id}.txt",
                            "fileHash": "1234567890",
                            "type": "file",
                            "metadata": {},
                        }
                    ]
                )
            )
            row = _create_base_row(
                run_id=run_data.run_id,
                attr_path=attr_path,
                attr_type="artifact",
            )
            row["file_value"] = {"path": filename}
            all_rows.append(row)

        # File series
        for attr_path, series in run_data.file_series.items():
            for step, filename in series:
                test_file_path = project_files_dir / filename
                test_file_path.write_text(
                    f"Test file series content for {run_data.run_id} at step {step}"
                )
                row = _create_base_row(
                    run_id=run_data.run_id,
                    attr_path=attr_path,
                    attr_type="file_series",
                    step=step,
                    timestamp=TEST_NOW,
                )
                row["file_value"] = {"path": filename}
                all_rows.append(row)

        # File sets
        for attr_path, dirname in run_data.file_sets.items():
            # Create a directory with multiple files for file_set
            file_set_dir = project_files_dir / dirname
            file_set_dir.mkdir(parents=True, exist_ok=True)
            # Create a few files in the directory
            for file_idx in range(2):
                file_path = file_set_dir / f"file_{file_idx}.txt"
                file_path.write_text(
                    f"Test file set content for {run_data.run_id} file {file_idx}"
                )
            row = _create_base_row(
                run_id=run_data.run_id,
                attr_path=attr_path,
                attr_type="file_set",
            )
            row["file_value"] = {"path": dirname}
            all_rows.append(row)

    # Group rows by run_id and write separate files for each run
    # New naming pattern: {run_id}_part_0.parquet
    rows_by_run = defaultdict(list)
    for row in all_rows:
        rows_by_run[row["run_id"]].append(row)

    # Write a separate parquet file for each run
    for run_id, run_rows in rows_by_run.items():
        run_table = pa.Table.from_pylist(run_rows, schema=model.SCHEMA)
        sanitized_run_id = sanitize_path_part(run_id)
        parquet_file = project_dir / f"{sanitized_run_id}_part_0.parquet"
        pq.write_table(run_table, parquet_file)

    yield data_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mlflow_loader(mlflow_tracking_uri: str) -> MLflowLoader:
    """Create MLflowLoader instance."""
    return MLflowLoader(tracking_uri=mlflow_tracking_uri)


@pytest.fixture
def loader_manager(
    test_data_dir: pathlib.Path,
    mlflow_loader: MLflowLoader,
) -> LoaderManager:
    """Create LoaderManager instance with MLflowLoader."""
    parquet_reader = ParquetReader(base_path=test_data_dir)
    files_directory = test_data_dir.parent / "files"
    return LoaderManager(
        parquet_reader=parquet_reader,
        data_loader=mlflow_loader,
        files_directory=files_directory,
        step_multiplier=100,
    )
