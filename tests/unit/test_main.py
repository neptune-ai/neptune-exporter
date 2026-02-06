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
