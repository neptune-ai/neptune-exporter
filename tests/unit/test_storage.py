import pyarrow as pa
from pathlib import Path
from neptune_exporter.storage.parquet import ParquetStorage


def test_parquet_storage_init():
    """Test ParquetStorage initialization."""
    base_path = Path("./test_output")
    storage = ParquetStorage(base_path)
    assert storage.base_path == base_path
    assert base_path.exists()


def test_parquet_storage_save(temp_dir):
    """Test saving data to Parquet file."""
    base_path = temp_dir
    storage = ParquetStorage(base_path)

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
    """Test using ParquetStorage with context manager."""
    base_path = temp_dir
    storage = ParquetStorage(base_path)

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
    storage = ParquetStorage(base_path, target_part_size_bytes=1024)  # 1KB limit

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
