import pyarrow as pa
from pathlib import Path
from tempfile import TemporaryDirectory
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.utils import sanitize_path_part


def test_parquet_storage_init():
    """Test ParquetWriter initialization."""
    base_path = Path("./test_output")
    storage = ParquetWriter(base_path)
    assert storage.base_path == base_path
    assert base_path.exists()


def test_parquet_storage_save(temp_dir):
    """Test saving data to Parquet file."""
    base_path = temp_dir
    storage = ParquetWriter(base_path)

    # Create test data as RecordBatch
    data = pa.record_batch(
        {
            "project_id": ["test-project"],
            "run_id": ["test-run"],
            "attribute_path": ["test/attribute"],
            "attribute_type": ["string"],
            "string_value": ["test-value"],
        }
    )

    # Save data using the new API
    storage.save("test-project", data)
    storage.close_all()

    # Check if file was created with new naming scheme
    expected_file = base_path / "test-project" / "part_0.parquet"
    assert expected_file.exists()


def test_parquet_storage_context_manager(temp_dir):
    """Test using ParquetWriter with context manager."""
    base_path = temp_dir
    storage = ParquetWriter(base_path)

    # Create test data as RecordBatch
    data = pa.record_batch(
        {
            "project_id": ["test-project"],
            "run_id": ["test-run"],
            "attribute_path": ["test/attribute"],
            "attribute_type": ["string"],
            "string_value": ["test-value"],
        }
    )

    # Use context manager
    with storage.project_writer("test-project") as writer:
        writer.save(data)

    # Check if file was created
    expected_file = base_path / "test-project" / "part_0.parquet"
    assert expected_file.exists()

    # Clean up
    expected_file.unlink()
    (base_path / "test-project").rmdir()
    base_path.rmdir()


def test_parquet_storage_part_splitting(temp_dir):
    """Test that data is split into multiple parts when size limit is reached."""
    base_path = temp_dir
    # Use a very small size limit to force part splitting
    storage = ParquetWriter(base_path, target_part_size_bytes=1024)  # 1KB limit

    # Create test data that will exceed the size limit
    large_data = pa.record_batch(
        {
            "project_id": ["test-project"] * 100,  # Repeat to make it larger
            "run_id": ["test-run"] * 100,
            "attribute_path": ["test/attribute"] * 100,
            "attribute_type": ["string"] * 100,
            "string_value": [
                "test-value-with-some-additional-content-to-make-it-larger"
            ]
            * 100,
        }
    )

    # Save multiple batches to trigger part splitting
    for i in range(5):
        storage.save("test-project", large_data)

    storage.close_all()

    # Check that multiple parts were created
    project_dir = base_path / "test-project"
    assert project_dir.exists()

    # List all parquet files in the project directory
    parquet_files = list(project_dir.glob("part_*.parquet"))
    assert len(parquet_files) > 1, (
        f"Expected multiple parts, but found {len(parquet_files)} files: {parquet_files}"
    )

    # Verify part numbering starts from 0 and is sequential
    part_numbers = []
    for file_path in parquet_files:
        # Extract part number from filename like "part_0.parquet"
        part_num = int(file_path.stem.split("_")[1])
        part_numbers.append(part_num)

    part_numbers.sort()
    expected_numbers = list(range(len(parquet_files)))
    assert part_numbers == expected_numbers, (
        f"Expected part numbers {expected_numbers}, but got {part_numbers}"
    )

    # Verify all files exist and are not empty
    for file_path in parquet_files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0, f"Part file {file_path} is empty"


def test_parquet_storage_sanitizes_project_id(temp_dir):
    """Test that ParquetWriter sanitizes project IDs with special characters."""
    base_path = temp_dir
    storage = ParquetWriter(base_path)

    # Create test data
    data = pa.record_batch(
        {
            "project_id": ["test-project"],
            "run_id": ["test-run"],
            "attribute_path": ["test/attribute"],
            "attribute_type": ["string"],
            "string_value": ["test-value"],
        }
    )

    # Use a project ID with special characters that need sanitization
    project_id_with_slashes = "org/project"
    storage.save(project_id_with_slashes, data)
    storage.close_all()

    # The file should be created with sanitized project ID
    expected_file = base_path / "org_project" / "part_0.parquet"
    assert expected_file.exists(), (
        f"Expected file at {expected_file}, but it doesn't exist"
    )

    # Verify the original project ID directory was not created
    original_path = base_path / "org" / "project" / "part_0.parquet"
    assert not original_path.exists(), f"Original path {original_path} should not exist"


def test_delete_project_data():
    """Test deleting project directory."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        storage = ParquetWriter(base_path=temp_path)

        # Create a project directory with some files
        project_id = "test-org/test-project"
        sanitized_project_id = sanitize_path_part(project_id)
        project_dir = temp_path / sanitized_project_id
        project_dir.mkdir(parents=True)

        # Create some dummy files
        (project_dir / "part_0.parquet").touch()
        (project_dir / "part_1.parquet").touch()
        (project_dir / "other_file.txt").touch()

        # Verify directory exists
        assert project_dir.exists()
        assert len(list(project_dir.iterdir())) == 3

        # Delete project data
        storage.delete_project_data(project_id)

        # Verify directory is deleted
        assert not project_dir.exists()

        # Test deleting non-existent project (should not raise error)
        storage.delete_project_data("non-existent-project")
