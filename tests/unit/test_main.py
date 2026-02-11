from datetime import datetime
from pathlib import Path

from click.testing import CliRunner
from neptune_exporter.main import cli


_TEST_LOG_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _test_log_file() -> str:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"neptune_exporter_{_TEST_LOG_STAMP}.log")


def test_main_rejects_empty_project_ids():
    """Test that export command rejects empty project IDs."""
    runner = CliRunner()

    # Test with empty string
    result = runner.invoke(cli, ["export", "-p", "", "--log-file", _test_log_file()])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    # Test with whitespace-only string
    result = runner.invoke(cli, ["export", "-p", "   ", "--log-file", _test_log_file()])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    # Test with multiple project IDs where one is empty
    result = runner.invoke(
        cli,
        ["export", "-p", "valid-project", "-p", "", "--log-file", _test_log_file()],
    )
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output


def test_main_accepts_valid_project_ids():
    """Test that export command accepts valid project IDs."""
    runner = CliRunner()

    # Test with valid project ID (this will fail later due to missing API token, but not due to validation)
    result = runner.invoke(
        cli, ["export", "-p", "valid-project", "--log-file", _test_log_file()]
    )
    # Should not fail due to empty project ID validation
    assert "Project ID cannot be empty" not in result.output


def test_export_models_rejects_empty_project_ids():
    """Test that export-models command rejects empty project IDs."""
    runner = CliRunner()

    result = runner.invoke(
        cli, ["export-models", "-p", "", "--log-file", _test_log_file()]
    )
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output


def test_export_models_rejects_classes_and_exclude_combination():
    """Test that export-models rejects using --classes and --exclude together."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export-models",
            "-p",
            "valid-project",
            "-c",
            "parameters",
            "--exclude",
            "files",
            "--log-file",
            _test_log_file(),
        ],
    )
    assert result.exit_code != 0
    assert "Cannot specify both --classes and --exclude" in result.output


def test_summary_accepts_model_data_path(tmp_path):
    """Test that summary command accepts --model-data-path option."""
    runner = CliRunner()
    data_dir = tmp_path / "data"
    model_data_dir = tmp_path / "model_data"
    data_dir.mkdir()
    model_data_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "summary",
            "--data-path",
            str(data_dir),
            "--model-data-path",
            str(model_data_dir),
            "--log-file",
            _test_log_file(),
        ],
    )
    assert result.exit_code == 0


def test_summary_allows_missing_run_data_path_for_model_only_export(tmp_path):
    """Test summary works when run data path is missing but model data path exists."""
    runner = CliRunner()
    missing_data_dir = tmp_path / "missing_runs_data"
    model_data_dir = tmp_path / "model_data"
    model_data_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "summary",
            "--data-path",
            str(missing_data_dir),
            "--model-data-path",
            str(model_data_dir),
            "--log-file",
            _test_log_file(),
        ],
    )

    assert result.exit_code == 0
