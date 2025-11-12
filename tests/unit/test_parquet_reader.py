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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from tempfile import TemporaryDirectory
from decimal import Decimal

from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter import model
from neptune_exporter.utils import sanitize_path_part


def test_list_projects_empty_directory():
    """Test listing projects when directory is empty."""
    with TemporaryDirectory() as temp_dir:
        reader = ParquetReader(Path(temp_dir))
        projects = reader.list_project_directories()
        assert projects == []


def test_list_projects_with_data():
    """Test listing projects when parquet files exist."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["test_value"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        # Create project directory and parquet file
        project_dir = temp_path / "test-project"
        project_dir.mkdir()
        parquet_file = project_dir / "part_0.parquet"

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)
        pq.write_table(table, parquet_file)

        # Test reader
        reader = ParquetReader(temp_path)
        projects = reader.list_project_directories()
        assert len(projects) == 1
        assert projects[0].name == "test-project"


def test_read_project_data():
    """Test reading project data with filters."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data for two runs
        test_data_run1 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param1"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["test_value"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        test_data_run2 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-456"],
                "attribute_path": ["test/param2"],
                "attribute_type": ["float"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [3.14],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        # Create project directory and parquet files with new naming pattern
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()
        parquet_file_run1 = (
            project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet"
        )
        parquet_file_run2 = (
            project_dir / f"{sanitize_path_part('RUN-456')}_part_0.parquet"
        )

        table_run1 = pa.Table.from_pandas(test_data_run1, schema=model.SCHEMA)
        table_run2 = pa.Table.from_pandas(test_data_run2, schema=model.SCHEMA)
        pq.write_table(table_run1, parquet_file_run1)
        pq.write_table(table_run2, parquet_file_run2)

        # Test reader
        reader = ParquetReader(temp_path)

        # Read all data (should yield parts separately)
        all_data = list(reader.read_project_data(project_dir))
        assert len(all_data) == 2  # Two parts (one per run)
        # Check that we have data from both runs
        all_run_ids = set()
        for table in all_data:
            run_ids = pc.unique(table["run_id"]).to_pylist()
            all_run_ids.update(run_ids)
        assert all_run_ids == {"RUN-123", "RUN-456"}

        # Read filtered data
        filtered_data = list(reader.read_project_data(project_dir, runs=["RUN-123"]))
        assert len(filtered_data) == 1  # One part for RUN-123
        assert filtered_data[0]["run_id"][0].as_py() == "RUN-123"


def test_check_run_exists():
    """Test checking if a run exists."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data for a single run
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param1"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["test_value"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        # Create project directory and parquet file with new naming scheme
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()
        parquet_file = project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet"

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)
        pq.write_table(table, parquet_file)

        # Test reader
        reader = ParquetReader(temp_path)

        # Test that existing run is found
        assert reader.check_run_exists("test-project", "RUN-123"), (
            "Existing run should be found"
        )

        # Test that non-existent run is not found
        assert not reader.check_run_exists("test-project", "RUN-456"), (
            "Non-existent run should not be found"
        )

        # Test with non-existent project
        assert not reader.check_run_exists("non-existent-project", "RUN-123"), (
            "Non-existent project should return False"
        )

        # Test with empty directory
        empty_dir = temp_path / "empty-project"
        empty_dir.mkdir()
        assert not reader.check_run_exists("empty-project", "RUN-123"), (
            "Empty project should return False"
        )


def test_read_project_data_with_attribute_type_filter():
    """Test reading project data with attribute type filter."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data with different attribute types
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project"] * 3,
                "run_id": ["RUN-123"] * 3,
                "attribute_path": ["test/param1", "test/param2", "test/metric"],
                "attribute_type": ["string", "float", "float_series"],
                "step": [None, None, Decimal("1.0")],
                "timestamp": [None, None, None],
                "int_value": [None, None, None],
                "float_value": [None, 3.14, 0.5],
                "string_value": ["test_value", None, None],
                "bool_value": [None, None, None],
                "datetime_value": [None, None, None],
                "string_set_value": [None, None, None],
                "file_value": [None, None, None],
                "histogram_value": [None, None, None],
            }
        )

        # Create project directory and parquet file with new naming pattern
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()
        parquet_file = project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet"

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)
        pq.write_table(table, parquet_file)

        # Test reader
        reader = ParquetReader(temp_path)

        # Read only string attributes
        string_data = list(
            reader.read_project_data(project_dir, attribute_types=["string"])
        )
        assert len(string_data) == 1
        assert len(string_data[0]) == 1
        assert string_data[0]["attribute_type"][0].as_py() == "string"

        # Read only float_series attributes
        series_data = list(
            reader.read_project_data(project_dir, attribute_types=["float_series"])
        )
        assert len(series_data) == 1
        assert len(series_data[0]) == 1
        assert series_data[0]["attribute_type"][0].as_py() == "float_series"


def test_list_runs():
    """Test listing runs in a project."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()

        # Create test data for two runs
        test_data_run1 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param1"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["test_value"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        test_data_run2 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-456"],
                "attribute_path": ["test/param2"],
                "attribute_type": ["float"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [3.14],
                "string_value": [None],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        parquet_file_run1 = (
            project_dir / f"{sanitize_path_part('RUN-123')}_part_0.parquet"
        )
        parquet_file_run2 = (
            project_dir / f"{sanitize_path_part('RUN-456')}_part_0.parquet"
        )

        table_run1 = pa.Table.from_pandas(test_data_run1, schema=model.SCHEMA)
        table_run2 = pa.Table.from_pandas(test_data_run2, schema=model.SCHEMA)
        pq.write_table(table_run1, parquet_file_run1)
        pq.write_table(table_run2, parquet_file_run2)

        reader = ParquetReader(temp_path)
        runs = reader.list_run_files(project_dir)

        assert len(runs) == 2
        # list_runs returns sanitized run IDs (with digest suffix)
        assert sanitize_path_part("RUN-123") in runs
        assert sanitize_path_part("RUN-456") in runs


def test_read_run_data():
    """Test reading run data part by part."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / sanitize_path_part("test-project")
        project_dir.mkdir()

        # Create test data split across two parts
        test_data_part0 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param1"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["value1"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        test_data_part1 = pd.DataFrame(
            {
                "project_id": ["test-project"],
                "run_id": ["RUN-123"],
                "attribute_path": ["test/param2"],
                "attribute_type": ["string"],
                "step": [None],
                "timestamp": [None],
                "int_value": [None],
                "float_value": [None],
                "string_value": ["value2"],
                "bool_value": [None],
                "datetime_value": [None],
                "string_set_value": [None],
                "file_value": [None],
                "histogram_value": [None],
            }
        )

        sanitized_run_id = sanitize_path_part("RUN-123")
        parquet_file_part0 = project_dir / f"{sanitized_run_id}_part_0.parquet"
        parquet_file_part1 = project_dir / f"{sanitized_run_id}_part_1.parquet"

        table_part0 = pa.Table.from_pandas(test_data_part0, schema=model.SCHEMA)
        table_part1 = pa.Table.from_pandas(test_data_part1, schema=model.SCHEMA)
        pq.write_table(table_part0, parquet_file_part0)
        pq.write_table(table_part1, parquet_file_part1)

        reader = ParquetReader(temp_path)
        parts = list(reader.read_run_data(project_dir, sanitized_run_id))

        assert len(parts) == 2
        assert len(parts[0]) == 1
        assert len(parts[1]) == 1
        assert parts[0]["attribute_path"][0].as_py() == "test/param1"
        assert parts[1]["attribute_path"][0].as_py() == "test/param2"


def test_read_run_metadata():
    """Test reading run metadata."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / "test-project"
        project_dir.mkdir()

        # Create test data with metadata attributes
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project"] * 5,
                "run_id": ["RUN-123"] * 5,
                "attribute_path": [
                    "sys/custom_run_id",
                    "sys/name",
                    "sys/forking/parent",
                    "sys/forking/step",
                    "test/param",
                ],
                "attribute_type": ["string", "string", "string", "float", "string"],
                "step": [None, None, None, None, None],
                "timestamp": [None, None, None, None, None],
                "int_value": [None, None, None, None, None],
                "float_value": [None, None, None, 10.5, None],
                "string_value": [
                    "custom-run-123",
                    "my-experiment",
                    "RUN-456",
                    None,
                    "test_value",
                ],
                "bool_value": [None, None, None, None, None],
                "datetime_value": [None, None, None, None, None],
                "string_set_value": [None, None, None, None, None],
                "file_value": [None, None, None, None, None],
                "histogram_value": [None, None, None, None, None],
            }
        )

        parquet_file = project_dir / "RUN-123_part_0.parquet"
        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)
        pq.write_table(table, parquet_file)

        reader = ParquetReader(temp_path)
        metadata = reader.read_run_metadata(project_dir, "RUN-123")

        assert metadata is not None
        assert metadata.project_id == "test-project"
        assert metadata.run_id == "RUN-123"
        assert metadata.custom_run_id == "custom-run-123"
        assert metadata.experiment_name == "my-experiment"
        assert metadata.parent_source_run_id == "RUN-456"
        assert metadata.fork_step == 10.5
