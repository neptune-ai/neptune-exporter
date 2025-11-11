import pyarrow as pa
from pathlib import Path
from neptune_exporter.storage.parquet_writer import ParquetWriter


def test_parquet_writer_init():
    """Test ParquetWriter initialization."""
    base_path = Path("./test_output")
    storage = ParquetWriter(base_path)
    assert storage.base_path == base_path
    assert base_path.exists()


def test_parquet_writer_save(temp_dir):
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
    with storage.run_writer("test-project", "test-run") as writer:
        writer.save(data)
        # Context manager will call finish_run() on exit

    # Check if file was created with new naming scheme
    expected_file = base_path / "test-project" / "test-run_part_0.parquet"
    assert expected_file.exists()


def test_parquet_writer_context_manager(temp_dir):
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
    with storage.run_writer("test-project", "test-run") as writer:
        writer.save(data)
        # Context manager will call finish_run() on exit

    # Check if file was created
    expected_file = base_path / "test-project" / "test-run_part_0.parquet"
    assert expected_file.exists()


def test_parquet_writer_part_splitting(temp_dir):
    """Test that a run is split into multiple parts when size limit is reached."""
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

    # Save multiple batches for a single run, which will create multiple parts
    with storage.run_writer("test-project", "test-run") as writer:
        for i in range(5):
            writer.save(large_data)
        # Context manager will call finish_run() on exit

    storage.close_all()

    # Check that multiple parts were created
    project_dir = base_path / "test-project"
    assert project_dir.exists()

    # List all parquet files for this run
    parquet_files = list(project_dir.glob("test-run_part_*.parquet"))
    assert len(parquet_files) > 1, (
        f"Expected multiple parts, but found {len(parquet_files)} files: {parquet_files}"
    )

    # Verify part numbering starts from 0 and is sequential
    part_numbers = []
    for file_path in parquet_files:
        # Extract part number from filename like "test-run_part_0.parquet"
        stem = file_path.stem  # "test-run_part_0"
        parts = stem.rsplit("_part_", 1)
        if len(parts) == 2:
            part_numbers.append(int(parts[1]))

    part_numbers.sort()
    expected_numbers = list(range(len(parquet_files)))
    assert part_numbers == expected_numbers, (
        f"Expected part numbers {expected_numbers}, but got {part_numbers}"
    )

    # Verify all files exist and are not empty
    for file_path in parquet_files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0, f"Part file {file_path} is empty"

    # Verify part_0 exists (run is complete)
    part_0_file = project_dir / "test-run_part_0.parquet"
    assert part_0_file.exists(), "part_0 should exist for complete run"


def test_parquet_writer_runs_dont_split_across_parts(temp_dir):
    """Test that a single run's data can span multiple parts when size limit is exceeded."""
    base_path = temp_dir
    # Use a very small size limit
    storage = ParquetWriter(base_path, target_part_size_bytes=1024)  # 1KB limit

    # Create test data that will exceed the size limit
    large_data = pa.record_batch(
        {
            "project_id": ["test-project"] * 200,  # Large batch
            "run_id": ["run-A"] * 200,  # All data for run A
            "attribute_path": ["test/attribute"] * 200,
            "attribute_type": ["string"] * 200,
            "string_value": [
                "test-value-with-some-additional-content-to-make-it-larger"
            ]
            * 200,
        }
    )

    # Write a run that exceeds size limit - should create multiple parts
    with storage.run_writer("test-project", "run-A") as writer:
        # Write multiple batches for the same run
        for i in range(3):
            writer.save(large_data)
        # Context manager will call finish_run() on exit

    storage.close_all()

    # Check that parts were created for run-A
    project_dir = base_path / "test-project"
    assert project_dir.exists()

    parquet_files = list(project_dir.glob("run-A_part_*.parquet"))
    # Should have at least 1 part (may have multiple if size limit exceeded)
    assert len(parquet_files) >= 1, (
        f"Expected at least one part for run-A, but found {len(parquet_files)} files: {parquet_files}"
    )

    # Verify part_0 exists (run is complete)
    part_0_file = project_dir / "run-A_part_0.parquet"
    assert part_0_file.exists(), "part_0 should exist for complete run"

    # Verify the file is not empty
    assert part_0_file.stat().st_size > 0


def test_parquet_writer_sanitizes_project_id(temp_dir):
    """Test that ParquetWriter sanitizes project IDs and run IDs with special characters."""
    base_path = temp_dir
    storage = ParquetWriter(base_path)

    # Create test data
    data = pa.record_batch(
        {
            "project_id": ["org/project"],
            "run_id": ["run/with/slashes"],
            "attribute_path": ["test/attribute"],
            "attribute_type": ["string"],
            "string_value": ["test-value"],
        }
    )

    # Use project ID and run ID with special characters that need sanitization
    project_id_with_slashes = "org/project"
    run_id_with_slashes = "run/with/slashes"

    with storage.run_writer(project_id_with_slashes, run_id_with_slashes) as writer:
        writer.save(data)
        # Context manager will call finish_run() on exit

    # The file should be created with sanitized project ID and run ID
    expected_file = base_path / "org_project" / "run_with_slashes_part_0.parquet"
    assert expected_file.exists(), (
        f"Expected file at {expected_file}, but it doesn't exist"
    )

    # Verify the original project ID directory was not created
    original_path = (
        base_path / "org" / "project" / "run" / "with" / "slashes_part_0.parquet"
    )
    assert not original_path.exists(), f"Original path {original_path} should not exist"


def test_parquet_writer_renaming_moves_part_0_last(temp_dir):
    """Test that part_0 is moved last when renaming .tmp files."""
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

    # Create a run with multiple parts
    with storage.run_writer("test-project", "test-run") as writer:
        # Write data that will create multiple parts
        for i in range(3):
            writer.save(data)
        # Context manager will call finish_run() on exit

    # Verify part_0 exists (moved last)
    project_dir = base_path / "test-project"
    part_0_file = project_dir / "test-run_part_0.parquet"
    assert part_0_file.exists(), "part_0 should exist after renaming"

    # Verify no .tmp files remain
    tmp_files = list(project_dir.glob("*.tmp"))
    assert len(tmp_files) == 0, f"No .tmp files should remain, found: {tmp_files}"


def test_parquet_writer_incomplete_run_not_considered(temp_dir):
    """Test that incomplete runs (no part_0) are not considered."""
    from neptune_exporter.storage.parquet_reader import ParquetReader

    base_path = temp_dir
    storage = ParquetWriter(base_path)
    reader = ParquetReader(base_path)

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

    # Create a run but don't finish it (simulate incomplete run)
    # We'll manually create a part_1 file without part_0
    project_dir = base_path / "test-project"
    project_dir.mkdir(parents=True)

    # Manually create part_1 without part_0 (incomplete run)
    import pyarrow.parquet as pq

    incomplete_file = project_dir / "test-run_part_1.parquet"
    pq.write_table(pa.Table.from_batches([data]), incomplete_file)

    # check_run_exists should return False for incomplete run (no part_0)
    assert not reader.check_run_exists("test-project", "test-run"), (
        "Incomplete run should not exist"
    )

    # Now create a complete run (with part_0)
    with storage.run_writer("test-project", "complete-run") as writer:
        writer.save(data)
        # Context manager will call finish_run() on exit

    # check_run_exists should return True for complete run
    assert reader.check_run_exists("test-project", "complete-run"), (
        "Complete run should exist"
    )
    assert not reader.check_run_exists("test-project", "test-run"), (
        "Incomplete run should still not exist"
    )


def test_parquet_writer_cleanup_incomplete_run(temp_dir):
    """Test that leftover .tmp files are cleaned up when starting a new write."""
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

    project_dir = base_path / "test-project"
    project_dir.mkdir(parents=True)

    # Manually create some leftover .tmp files (simulating interrupted write)
    import pyarrow.parquet as pq

    leftover_file_1 = project_dir / "test-run_part_0.parquet.tmp"
    leftover_file_2 = project_dir / "test-run_part_1.parquet.tmp"
    pq.write_table(pa.Table.from_batches([data]), leftover_file_1)
    pq.write_table(pa.Table.from_batches([data]), leftover_file_2)

    # Verify .tmp files exist
    assert leftover_file_1.exists()
    assert leftover_file_2.exists()

    # Start a new write - should clean up the leftover .tmp files
    with storage.run_writer("test-project", "test-run") as writer:
        writer.save(data)
        # Context manager will call finish_run() on exit

    # Verify leftover .tmp files were deleted
    assert not leftover_file_1.exists(), "Leftover .tmp file should be deleted"
    assert not leftover_file_2.exists(), "Leftover .tmp file should be deleted"

    # Verify new file was created
    final_file = project_dir / "test-run_part_0.parquet"
    assert final_file.exists(), "New file should be created"
