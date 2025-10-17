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

from pathlib import Path
import pyarrow as pa
from unittest.mock import Mock

from neptune_exporter.summary_manager import SummaryManager
from neptune_exporter.storage.parquet_reader import ParquetReader


def test_summary_manager_init():
    """Test SummaryManager initialization."""
    mock_reader = Mock(spec=ParquetReader)
    manager = SummaryManager(parquet_reader=mock_reader)

    assert manager._parquet_reader == mock_reader
    assert manager._logger is not None


def test_get_data_summary_empty():
    """Test get_data_summary when no projects exist."""
    mock_reader = Mock(spec=ParquetReader)
    mock_reader.list_project_directories.return_value = []

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_data_summary()

    assert result == {"total_projects": 0, "projects": {}}
    mock_reader.list_project_directories.assert_called_once()


def test_get_data_summary_with_projects():
    """Test get_data_summary with multiple projects."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir1 = Path("/data/project1")
    project_dir2 = Path("/data/project2")
    mock_reader.list_project_directories.return_value = [project_dir1, project_dir2]

    manager = SummaryManager(parquet_reader=mock_reader)

    # Mock get_project_summary to return different data for each project
    def mock_get_project_summary(project_dir):
        if project_dir == project_dir1:
            return {
                "project_id": "project-1",
                "total_runs": 2,
                "attribute_types": ["float", "string"],
                "runs": ["run1", "run2"],
            }
        else:
            return {
                "project_id": "project-2",
                "total_runs": 1,
                "attribute_types": ["float"],
                "runs": ["run3"],
            }

    manager.get_project_summary = mock_get_project_summary

    result = manager.get_data_summary()

    assert result["total_projects"] == 2
    assert len(result["projects"]) == 2
    assert project_dir1 in result["projects"]
    assert project_dir2 in result["projects"]
    assert result["projects"][project_dir1]["project_id"] == "project-1"
    assert result["projects"][project_dir2]["project_id"] == "project-2"


def test_get_project_summary_success():
    """Test get_project_summary with successful data reading."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/project1")

    # Create mock PyArrow table data
    mock_table = pa.table(
        {
            "project_id": ["project-1", "project-1", "project-1"],
            "run_id": ["run1", "run1", "run2"],
            "attribute_type": ["float", "string", "float"],
            "attribute_path": ["loss", "name", "accuracy"],
        }
    )

    # Mock the generator that read_project_data returns
    mock_generator = iter([mock_table])
    mock_reader.read_project_data.return_value = mock_generator

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result["project_id"] == "project-1"
    assert result["total_runs"] == 2
    assert set(result["attribute_types"]) == {"float", "string"}
    assert set(result["runs"]) == {"run1", "run2"}

    mock_reader.read_project_data.assert_called_once_with(project_dir)


def test_get_project_summary_empty_data():
    """Test get_project_summary when no data is found."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/empty_project")

    # Mock empty generator
    mock_generator = iter([])
    mock_reader.read_project_data.return_value = mock_generator

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result == {
        "project_id": None,
        "total_runs": 0,
        "attribute_types": [],
        "runs": [],
    }


def test_get_project_summary_exception_handling():
    """Test get_project_summary when an exception occurs."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/error_project")
    mock_reader.read_project_data.side_effect = Exception("File not found")

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result is None


def test_get_project_summary_with_complex_data():
    """Test get_project_summary with more complex data structure."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/complex_project")

    # Create mock PyArrow table with more data
    mock_table = pa.table(
        {
            "project_id": ["complex-project"] * 6,
            "run_id": ["run1", "run1", "run1", "run2", "run2", "run3"],
            "attribute_type": ["float", "string", "int", "float", "string", "float"],
            "attribute_path": [
                "loss",
                "name",
                "epoch",
                "accuracy",
                "status",
                "f1_score",
            ],
        }
    )

    mock_generator = iter([mock_table])
    mock_reader.read_project_data.return_value = mock_generator

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result["project_id"] == "complex-project"
    assert result["total_runs"] == 3
    assert set(result["attribute_types"]) == {"float", "string", "int"}
    assert set(result["runs"]) == {"run1", "run2", "run3"}


def test_get_project_summary_multiple_tables():
    """Test get_project_summary when read_project_data returns multiple tables."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/multi_table_project")

    # Create multiple mock tables
    table1 = pa.table(
        {
            "project_id": ["multi-project", "multi-project"],
            "run_id": ["run1", "run1"],
            "attribute_type": ["float", "string"],
            "attribute_path": ["loss", "name"],
        }
    )

    table2 = pa.table(
        {
            "project_id": ["multi-project", "multi-project"],
            "run_id": ["run2", "run2"],
            "attribute_type": ["int", "float"],
            "attribute_path": ["epoch", "accuracy"],
        }
    )

    mock_generator = iter([table1, table2])
    mock_reader.read_project_data.return_value = mock_generator

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result["project_id"] == "multi-project"
    assert result["total_runs"] == 2
    assert set(result["attribute_types"]) == {"float", "string", "int"}
    assert set(result["runs"]) == {"run1", "run2"}


def test_get_data_summary_with_project_summary_exception():
    """Test get_data_summary when get_project_summary raises an exception."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/error_project")
    mock_reader.list_project_directories.return_value = [project_dir]

    manager = SummaryManager(parquet_reader=mock_reader)
    manager.get_project_summary = lambda x: None

    result = manager.get_data_summary()

    assert result["total_projects"] == 1
    assert project_dir in result["projects"]
    assert result["projects"][project_dir] is None


def test_get_project_summary_with_nonexistent_directory():
    """Test get_project_summary with a directory that doesn't exist."""
    mock_reader = Mock(spec=ParquetReader)
    project_dir = Path("/data/nonexistent")
    mock_reader.read_project_data.side_effect = FileNotFoundError("Directory not found")

    manager = SummaryManager(parquet_reader=mock_reader)
    result = manager.get_project_summary(project_dir)

    assert result is None
