import pyarrow as pa
from pathlib import Path
from neptune_exporter.storage.parquet import ParquetStorage


def test_parquet_storage_init():
    """Test ParquetStorage initialization."""
    base_path = Path("./test_output")
    storage = ParquetStorage(base_path)
    assert storage.base_path == base_path
    assert base_path.exists()


def test_parquet_storage_save():
    """Test saving data to Parquet file."""
    base_path = Path("./test_output")
    storage = ParquetStorage(base_path)

    # Create test data
    data = pa.table(
        {
            "project_id": ["test-project"],
            "run_id": ["test-run"],
            "attribute_path": ["test/attribute"],
            "attribute_type": ["string"],
            "string_value": ["test-value"],
        }
    )

    # Save data
    storage.save("test-project", data)

    # Check if file was created
    expected_file = base_path / "test-project.parquet"
    assert expected_file.exists()

    # Clean up
    expected_file.unlink()
    base_path.rmdir()
