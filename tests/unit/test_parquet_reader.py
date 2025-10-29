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
from pathlib import Path
from tempfile import TemporaryDirectory
from decimal import Decimal

from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter import model


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


def test_list_parquet_files():
    """Test listing parquet files for a project."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / "test-project"
        project_dir.mkdir()

        # Create multiple parquet files
        for i in range(3):
            parquet_file = project_dir / f"part_{i}.parquet"
            parquet_file.touch()

        reader = ParquetReader(temp_path)
        files = reader._list_parquet_files_in_project(project_dir)

        assert len(files) == 3
        assert all(
            f.name.startswith("part_") and f.name.endswith(".parquet") for f in files
        )


def test_read_project_data():
    """Test reading project data with filters."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project", "test-project"],
                "run_id": ["RUN-123", "RUN-456"],
                "attribute_path": ["test/param1", "test/param2"],
                "attribute_type": ["string", "float"],
                "step": [None, None],
                "timestamp": [None, None],
                "int_value": [None, None],
                "float_value": [None, 3.14],
                "string_value": ["test_value", None],
                "bool_value": [None, None],
                "datetime_value": [None, None],
                "string_set_value": [None, None],
                "file_value": [None, None],
                "histogram_value": [None, None],
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

        # Read all data
        all_data = list(reader.read_project_data(project_dir))
        assert len(all_data) == 1
        assert len(all_data[0]) == 2

        # Read filtered data
        filtered_data = list(reader.read_project_data(project_dir, runs=["RUN-123"]))
        assert len(filtered_data) == 1
        assert len(filtered_data[0]) == 1
        assert filtered_data[0]["run_id"][0].as_py() == "RUN-123"


def test_get_unique_run_ids():
    """Test getting unique run IDs from parquet files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data with multiple runs
        test_data = pd.DataFrame(
            {
                "project_id": ["test-project", "test-project", "test-project"],
                "run_id": ["RUN-123", "RUN-456", "RUN-123"],  # RUN-123 appears twice
                "attribute_path": ["test/param1", "test/param2", "test/param3"],
                "attribute_type": ["string", "float", "int"],
                "step": [None, None, None],
                "timestamp": [None, None, None],
                "int_value": [None, None, 42],
                "float_value": [None, 3.14, None],
                "string_value": ["test_value", None, None],
                "bool_value": [None, None, None],
                "datetime_value": [None, None, None],
                "string_set_value": [None, None, None],
                "file_value": [None, None, None],
                "histogram_value": [None, None, None],
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

        # Test with data
        run_ids = reader.get_unique_run_ids(project_dir)
        assert run_ids == {"RUN-123", "RUN-456"}

        # Test with empty directory
        empty_dir = temp_path / "empty-project"
        empty_dir.mkdir()
        run_ids = reader.get_unique_run_ids(empty_dir)
        assert run_ids == set()

        # Test with non-existent directory
        non_existent_dir = temp_path / "non-existent"
        run_ids = reader.get_unique_run_ids(non_existent_dir)
        assert run_ids == set()


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

        # Create project directory and parquet file
        project_dir = temp_path / "test-project"
        project_dir.mkdir()
        parquet_file = project_dir / "part_0.parquet"

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
