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
    expected_file = base_path / "test-project" / "part_1.parquet"
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
    expected_file = base_path / "test-project" / "part_1.parquet"
    assert expected_file.exists()

    # Clean up
    expected_file.unlink()
    (base_path / "test-project").rmdir()
    base_path.rmdir()
