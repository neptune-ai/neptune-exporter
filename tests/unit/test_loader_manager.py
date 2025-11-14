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

import datetime
from pathlib import Path
from unittest.mock import Mock
import pyarrow as pa

from neptune_exporter.loader_manager import LoaderManager
from neptune_exporter.storage.parquet_reader import ParquetReader, RunMetadata
from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.utils import sanitize_path_part


def test_loader_manager_init():
    """Test LoaderManager initialization."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    assert manager._parquet_reader == mock_reader
    assert manager._data_loader == mock_loader
    assert manager._files_directory == files_dir
    assert manager._step_multiplier == 100


def test_topological_sort_parent_before_child():
    """Test that parent runs are processed before child runs."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: parent run and child run
    parent_run_id = "PARENT-123"
    child_run_id = "CHILD-456"

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    child_metadata = RunMetadata(
        project_id="project1",
        run_id=child_run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_run_id,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader to return both runs
    mock_reader.list_run_files.return_value = [child_run_id, parent_run_id]
    mock_reader.read_run_metadata.side_effect = [child_metadata, parent_metadata]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1, 2, 3]})])

    # Mock loader to track creation order
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: parent should be created before child
    assert len(created_runs) == 2
    assert created_runs[0] == parent_run_id
    assert created_runs[1] == child_run_id


def test_topological_sort_multiple_levels():
    """Test topological sorting with multiple levels of parent-child relationships."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: grandparent -> parent -> child
    grandparent_id = "GRANDPARENT-1"
    parent_id = "PARENT-2"
    child_id = "CHILD-3"

    grandparent_metadata = RunMetadata(
        project_id="project1",
        run_id=grandparent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=grandparent_id,
        fork_step=None,
        creation_time=None,
    )

    child_metadata = RunMetadata(
        project_id="project1",
        run_id=child_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader - runs listed in wrong order (child, parent, grandparent)
    mock_reader.list_run_files.return_value = [child_id, parent_id, grandparent_id]
    mock_reader.read_run_metadata.side_effect = [
        child_metadata,
        parent_metadata,
        grandparent_metadata,
    ]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation order
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: should be processed in topological order
    assert len(created_runs) == 3
    assert created_runs[0] == grandparent_id
    assert created_runs[1] == parent_id
    assert created_runs[2] == child_id


def test_topological_sort_creation_time_order():
    """Test that runs are processed in creation time order."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: two runs with different creation times
    run1_id = "RUN-1"
    run2_id = "RUN-2"
    run3_id = "RUN-3"
    creation_time1 = datetime.datetime(2025, 1, 1, 12, 0, 0)
    creation_time2 = None
    creation_time3 = datetime.datetime(2025, 1, 1, 12, 0, 2)

    run1_metadata = RunMetadata(
        project_id="project1",
        run_id=run1_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=creation_time1,
    )

    run2_metadata = RunMetadata(
        project_id="project1",
        run_id=run2_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=creation_time2,
    )

    run3_metadata = RunMetadata(
        project_id="project1",
        run_id=run3_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=creation_time3,
    )
    # Mock reader
    mock_reader.list_run_files.return_value = [run3_id, run2_id, run1_id]
    mock_reader.read_run_metadata.side_effect = [
        run3_metadata,
        run2_metadata,
        run1_metadata,
    ]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation order
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: runs should be processed in creation time order
    assert len(created_runs) == 3
    assert created_runs[0] == run1_id
    assert created_runs[1] == run3_id
    assert created_runs[2] == run2_id


def test_topological_sort_multiple_children():
    """Test topological sorting with one parent and multiple children."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: one parent with two children
    parent_id = "PARENT-1"
    child1_id = "CHILD-2"
    child2_id = "CHILD-3"

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    child1_metadata = RunMetadata(
        project_id="project1",
        run_id=child1_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    child2_metadata = RunMetadata(
        project_id="project1",
        run_id=child2_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader
    mock_reader.list_run_files.return_value = [child1_id, child2_id, parent_id]
    mock_reader.read_run_metadata.side_effect = [
        child1_metadata,
        child2_metadata,
        parent_metadata,
    ]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation order
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: parent should be first, children can be in any order after
    assert len(created_runs) == 3
    assert created_runs[0] == parent_id
    assert child1_id in created_runs[1:]
    assert child2_id in created_runs[1:]


def test_topological_sort_orphaned_run():
    """Test that orphaned runs (parent not in dataset) are processed."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: orphaned run (parent not in dataset)
    orphan_id = "ORPHAN-1"
    missing_parent_id = "MISSING-PARENT"

    orphan_metadata = RunMetadata(
        project_id="project1",
        run_id=orphan_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=missing_parent_id,  # Parent not in dataset
        fork_step=None,
        creation_time=None,
    )

    # Mock reader
    mock_reader.list_run_files.return_value = [orphan_id]
    mock_reader.read_run_metadata.return_value = orphan_metadata
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: orphaned run should still be processed (as root node)
    assert len(created_runs) == 1
    assert created_runs[0] == orphan_id
    # Verify parent_run_id is None (parent not found)
    create_run_calls = mock_loader.create_run.call_args_list
    assert create_run_calls[0][1]["parent_run_id"] is None


def test_topological_sort_mixed_orphaned_and_normal():
    """Test topological sorting with both orphaned and normal runs."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: one normal parent-child pair and one orphaned run
    parent_id = "PARENT-1"
    child_id = "CHILD-2"
    orphan_id = "ORPHAN-3"
    missing_parent_id = "MISSING-PARENT"

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    child_metadata = RunMetadata(
        project_id="project1",
        run_id=child_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    orphan_metadata = RunMetadata(
        project_id="project1",
        run_id=orphan_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=missing_parent_id,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader
    mock_reader.list_run_files.return_value = [orphan_id, child_id, parent_id]
    mock_reader.read_run_metadata.side_effect = [
        orphan_metadata,
        child_metadata,
        parent_metadata,
    ]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation order
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: parent should be before child, orphan can be anywhere (treated as root)
    assert len(created_runs) == 3
    assert created_runs.index(parent_id) < created_runs.index(child_id)
    # Orphan should be processed (as root node)
    assert orphan_id in created_runs


def test_topological_sort_no_runs():
    """Test loading when no runs are found."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Mock reader to return no runs
    mock_reader.list_run_files.return_value = []

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: no runs should be created
    mock_loader.create_run.assert_not_called()


def test_topological_sort_missing_metadata():
    """Test handling of runs with missing metadata."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Setup: one valid run and one with missing metadata
    valid_run_id = "VALID-1"
    invalid_run_id = "INVALID-2"

    valid_metadata = RunMetadata(
        project_id="project1",
        run_id=valid_run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader
    mock_reader.list_run_files.return_value = [valid_run_id, invalid_run_id]
    mock_reader.read_run_metadata.side_effect = [
        valid_metadata,
        None,
    ]  # Second returns None
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Track creation
    created_runs = []

    def track_create_run(*args, **kwargs):
        run_name = kwargs.get("run_name") or args[1]
        created_runs.append(run_name)
        return f"target-{run_name}"

    mock_loader.create_run.side_effect = track_create_run
    mock_loader.create_experiment.return_value = "exp-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Execute
    manager._load_project(project_dir, runs=None)

    # Verify: only valid run should be processed
    assert len(created_runs) == 1
    assert created_runs[0] == valid_run_id


def test_process_run_custom_run_id():
    """Test that custom_run_id is used when provided, otherwise falls back to run_id."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Test with custom_run_id
    run_id = "RUN-123"
    custom_run_id = "my-custom-name"

    metadata_with_custom = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=custom_run_id,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata_with_custom
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_run.return_value = "target-run-1"
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    # Mock find_run to return None (runs don't exist yet)
    mock_loader.find_run.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify custom_run_id was used
    mock_loader.create_run.assert_called_once()
    call_kwargs = mock_loader.create_run.call_args[1]
    assert call_kwargs["run_name"] == custom_run_id

    # Test without custom_run_id (should use run_id)
    metadata_without_custom = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_loader.reset_mock()
    mock_reader.read_run_metadata.return_value = metadata_without_custom

    manager._load_project(project_dir, runs=None)

    # Verify run_id was used as fallback
    call_kwargs = mock_loader.create_run.call_args[1]
    assert call_kwargs["run_name"] == run_id


def test_process_run_experiment_creation():
    """Test that experiments are created when experiment_name is provided."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    run_id = "RUN-123"
    experiment_name = "my-experiment"

    # Test with experiment_name
    metadata_with_experiment = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=experiment_name,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata_with_experiment
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_experiment.return_value = "exp-123"
    mock_loader.create_run.return_value = "target-run-1"
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify experiment was created
    mock_loader.create_experiment.assert_called_once_with(
        project_id="project1", experiment_name=experiment_name
    )
    # Verify experiment_id was passed to create_run
    call_kwargs = mock_loader.create_run.call_args[1]
    assert call_kwargs["experiment_id"] == "exp-123"

    # Test without experiment_name
    metadata_without_experiment = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_loader.reset_mock()
    mock_reader.read_run_metadata.return_value = metadata_without_experiment

    manager._load_project(project_dir, runs=None)

    # Verify experiment was not created
    mock_loader.create_experiment.assert_not_called()
    # Verify experiment_id is None
    call_kwargs = mock_loader.create_run.call_args[1]
    assert call_kwargs["experiment_id"] is None


def test_process_run_parent_lookup():
    """Test that parent run ID is correctly looked up and passed to create_run."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    parent_id = "PARENT-1"
    child_id = "CHILD-2"

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    child_metadata = RunMetadata(
        project_id="project1",
        run_id=child_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [parent_id, child_id]
    mock_reader.read_run_metadata.side_effect = [parent_metadata, child_metadata]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_run.side_effect = ["target-parent-1", "target-child-2"]
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify parent was created without parent_run_id
    parent_call = mock_loader.create_run.call_args_list[0]
    assert parent_call[1]["parent_run_id"] is None

    # Verify child was created with parent_run_id
    child_call = mock_loader.create_run.call_args_list[1]
    assert child_call[1]["parent_run_id"] == "target-parent-1"


def test_process_run_fork_step_and_multiplier():
    """Test that fork_step and step_multiplier are passed correctly."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=200,  # Custom multiplier
    )

    run_id = "RUN-123"
    fork_step = 42.5

    metadata = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=fork_step,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_run.return_value = "target-run-1"
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    # Mock find_run to return None (runs don't exist yet)
    mock_loader.find_run.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify fork_step and step_multiplier were passed
    call_kwargs = mock_loader.create_run.call_args[1]
    assert call_kwargs["fork_step"] == fork_step
    assert call_kwargs["step_multiplier"] == 200

    # Verify step_multiplier was also passed to upload_run_data
    upload_call_kwargs = mock_loader.upload_run_data.call_args[1]
    assert upload_call_kwargs["step_multiplier"] == 200


def test_process_run_files_directory():
    """Test that files directory is constructed correctly with sanitized project_id."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    run_id = "RUN-123"
    project_id = "my/project:id"  # Contains special characters

    metadata = RunMetadata(
        project_id=project_id,
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_run.return_value = "target-run-1"
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify files_directory was constructed with sanitized project_id
    upload_call_kwargs = mock_loader.upload_run_data.call_args[1]
    expected_files_dir = files_dir / sanitize_path_part(project_id)
    assert upload_call_kwargs["files_directory"] == expected_files_dir


def test_load_multiple_projects():
    """Test that load() processes multiple projects correctly."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    project_dir1 = Path("/tmp/data/project1")
    project_dir2 = Path("/tmp/data/project2")

    run_id1 = "RUN-1"
    run_id2 = "RUN-2"

    metadata1 = RunMetadata(
        project_id="project1",
        run_id=run_id1,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    metadata2 = RunMetadata(
        project_id="project2",
        run_id=run_id2,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader to return two projects
    mock_reader.list_project_directories.return_value = [project_dir1, project_dir2]
    mock_reader.list_run_files.side_effect = [[run_id1], [run_id2]]
    mock_reader.read_run_metadata.side_effect = [metadata1, metadata2]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    mock_loader.create_run.side_effect = ["target-run-1", "target-run-2"]
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    manager.load(project_ids=None, runs=None)

    # Verify both projects were processed
    assert mock_reader.list_run_files.call_count == 2
    assert mock_loader.create_run.call_count == 2


def test_load_error_handling_project_failure():
    """Test that load() continues processing other projects when one fails."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    project_dir1 = Path("/tmp/data/project1")
    project_dir2 = Path("/tmp/data/project2")

    run_id2 = "RUN-2"
    metadata2 = RunMetadata(
        project_id="project2",
        run_id=run_id2,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    # Mock reader: project1 fails, project2 succeeds
    mock_reader.list_project_directories.return_value = [project_dir1, project_dir2]
    mock_reader.list_run_files.side_effect = [
        Exception("Project 1 error"),  # First project fails
        [run_id2],  # Second project succeeds
    ]
    mock_reader.read_run_metadata.return_value = metadata2
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])
    mock_loader.find_run.return_value = None

    mock_loader.create_run.return_value = "target-run-2"
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None

    # Should not raise, should continue processing
    manager.load(project_ids=None, runs=None)

    # Verify second project was still processed
    assert mock_loader.create_run.call_count == 1


def test_load_error_handling_run_failure():
    """Test that _load_project continues processing other runs when one fails."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    run_id1 = "RUN-1"
    run_id2 = "RUN-2"

    metadata1 = RunMetadata(
        project_id="project1",
        run_id=run_id1,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    metadata2 = RunMetadata(
        project_id="project1",
        run_id=run_id2,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id1, run_id2]
    mock_reader.read_run_metadata.side_effect = [metadata1, metadata2]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # First run fails, second succeeds
    mock_loader.create_run.side_effect = [
        Exception("Run 1 error"),  # First run fails
        "target-run-2",  # Second run succeeds
    ]
    mock_loader.create_experiment.return_value = None
    mock_loader.upload_run_data.return_value = None
    mock_loader.find_run.return_value = None

    # Should not raise, should continue processing
    manager._load_project(project_dir, runs=None)

    # Verify second run was still processed
    assert mock_loader.create_run.call_count == 2
    assert mock_loader.upload_run_data.call_count == 1  # Only second run uploaded


def test_load_no_projects():
    """Test that load() handles no projects found gracefully."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    # Mock reader to return no projects
    mock_reader.list_project_directories.return_value = []

    # Should not raise
    manager.load(project_ids=None, runs=None)

    # Verify no runs were created
    mock_loader.create_run.assert_not_called()


def test_process_run_finds_existing_run():
    """Test that find_run is called and existing runs are reused."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    run_id = "RUN-123"
    existing_target_run_id = "existing-target-run-123"

    metadata = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Mock find_run to return existing run
    mock_loader.find_run.return_value = existing_target_run_id
    mock_loader.create_experiment.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify find_run was called
    mock_loader.find_run.assert_called_once_with(
        project_id="project1",
        run_name=run_id,
        experiment_id=None,
    )

    # Verify create_run was NOT called (run already exists)
    mock_loader.create_run.assert_not_called()

    # Verify upload_run_data was NOT called (run already exists)
    mock_loader.upload_run_data.assert_not_called()


def test_process_run_creates_new_run_when_not_found():
    """Test that new runs are created when find_run returns None."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    run_id = "RUN-123"
    new_target_run_id = "new-target-run-123"

    metadata = RunMetadata(
        project_id="project1",
        run_id=run_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [run_id]
    mock_reader.read_run_metadata.return_value = metadata
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Mock find_run to return None (run doesn't exist)
    mock_loader.find_run.return_value = None
    mock_loader.create_experiment.return_value = None
    mock_loader.create_run.return_value = new_target_run_id
    mock_loader.upload_run_data.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify find_run was called
    mock_loader.find_run.assert_called_once_with(
        project_id="project1",
        run_name=run_id,
        experiment_id=None,
    )

    # Verify create_run was called
    mock_loader.create_run.assert_called_once()

    # Verify upload_run_data was called
    mock_loader.upload_run_data.assert_called_once()


def test_process_run_uses_existing_run_for_parent_relationships():
    """Test that existing runs are used for parent/child relationships."""
    mock_reader = Mock(spec=ParquetReader)
    mock_loader = Mock(spec=DataLoader)
    files_dir = Path("/tmp/files")
    project_dir = Path("/tmp/data/project1")

    manager = LoaderManager(
        parquet_reader=mock_reader,
        data_loader=mock_loader,
        files_directory=files_dir,
        step_multiplier=100,
    )

    parent_id = "PARENT-1"
    child_id = "CHILD-2"
    existing_parent_target_id = "existing-parent-123"
    existing_child_target_id = "existing-child-456"

    parent_metadata = RunMetadata(
        project_id="project1",
        run_id=parent_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=None,
        fork_step=None,
        creation_time=None,
    )

    child_metadata = RunMetadata(
        project_id="project1",
        run_id=child_id,
        custom_run_id=None,
        experiment_name=None,
        parent_source_run_id=parent_id,
        fork_step=None,
        creation_time=None,
    )

    mock_reader.list_run_files.return_value = [parent_id, child_id]
    mock_reader.read_run_metadata.side_effect = [parent_metadata, child_metadata]
    mock_reader.read_run_data.return_value = iter([pa.table({"col": [1]})])

    # Mock find_run to return existing runs for both
    def find_run_side_effect(project_id, run_name, experiment_id):
        if run_name == parent_id:
            return existing_parent_target_id
        elif run_name == child_id:
            return existing_child_target_id
        return None

    mock_loader.find_run.side_effect = find_run_side_effect
    mock_loader.create_experiment.return_value = None

    manager._load_project(project_dir, runs=None)

    # Verify find_run was called for both runs
    assert mock_loader.find_run.call_count == 2

    # Verify create_run was NOT called (both runs exist)
    mock_loader.create_run.assert_not_called()

    # Verify upload_run_data was NOT called (both runs exist)
    mock_loader.upload_run_data.assert_not_called()

    # Verify parent/child mapping was stored correctly
    # (This is internal, but we can verify by checking that find_run was called with correct names)
    parent_call = [
        call_args
        for call_args in mock_loader.find_run.call_args_list
        if call_args[1]["run_name"] == parent_id
    ][0]
    child_call = [
        call_args
        for call_args in mock_loader.find_run.call_args_list
        if call_args[1]["run_name"] == child_id
    ][0]

    assert parent_call[1]["project_id"] == "project1"
    assert child_call[1]["project_id"] == "project1"
