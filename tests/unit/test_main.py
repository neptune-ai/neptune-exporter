from click.testing import CliRunner
from neptune_exporter.main import cli


def test_main_rejects_empty_project_ids():
    """Test that export command rejects empty project IDs."""
    runner = CliRunner()

    # Test with empty string
    result = runner.invoke(cli, ["export", "-p", ""])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    # Test with whitespace-only string
    result = runner.invoke(cli, ["export", "-p", "   "])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    # Test with multiple project IDs where one is empty
    result = runner.invoke(cli, ["export", "-p", "valid-project", "-p", ""])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output


def test_main_accepts_valid_project_ids():
    """Test that export command accepts valid project IDs."""
    runner = CliRunner()

    # Test with valid project ID (this will fail later due to missing API token, but not due to validation)
    result = runner.invoke(cli, ["export", "-p", "valid-project"])
    # Should not fail due to empty project ID validation
    assert "Project ID cannot be empty" not in result.output
