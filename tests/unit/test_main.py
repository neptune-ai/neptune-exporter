from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner
import neptune_exporter.main as main
from neptune_exporter.main import cli


_TEST_LOG_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _test_log_file() -> str:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"neptune_exporter_{_TEST_LOG_STAMP}.log")


def test_main_rejects_empty_project_ids():
    """Test that export command rejects empty project IDs."""
    runner = CliRunner()

    result = runner.invoke(cli, ["export", "-p", "", "--log-file", _test_log_file()])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    result = runner.invoke(cli, ["export", "-p", "   ", "--log-file", _test_log_file()])
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output

    result = runner.invoke(
        cli,
        ["export", "-p", "valid-project", "-p", "", "--log-file", _test_log_file()],
    )
    assert result.exit_code != 0
    assert "Project ID cannot be empty" in result.output


def test_main_accepts_valid_project_ids():
    """Test that export command accepts valid project IDs."""
    runner = CliRunner()

    result = runner.invoke(
        cli, ["export", "-p", "valid-project", "--log-file", _test_log_file()]
    )
    assert "Project ID cannot be empty" not in result.output


def test_main_rejects_mixed_explicit_and_workspace_modes():
    """Test that explicit project IDs cannot be mixed with workspace discovery."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export",
            "-p",
            "my-org/my-project",
            "--workspace",
            "my-org",
            "--log-file",
            _test_log_file(),
        ],
    )

    assert result.exit_code != 0
    assert "Cannot use --project-ids/-p together with --workspace" in result.output


def test_main_rejects_project_pattern_without_workspace():
    """Test that project regex filters require workspace mode."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export",
            "--project-pattern",
            ".*prod.*",
            "--log-file",
            _test_log_file(),
        ],
    )

    assert result.exit_code != 0
    assert "require --workspace" in result.output


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


def test_export_models_rejects_mixed_explicit_and_workspace_modes():
    """Test that export-models rejects mixed explicit and discovery flags."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export-models",
            "-p",
            "my-org/my-project",
            "--workspace",
            "my-org",
            "--log-file",
            _test_log_file(),
        ],
    )

    assert result.exit_code != 0
    assert "Cannot use --project-ids/-p together with --workspace" in result.output


def test_export_models_rejects_project_pattern_without_workspace():
    """Test that export-models project regex filters require workspace mode."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "export-models",
            "--project-pattern",
            ".*prod.*",
            "--log-file",
            _test_log_file(),
        ],
    )

    assert result.exit_code != 0
    assert "require --workspace" in result.output


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


def test_log_selected_project_ids_shows_preview_and_remainder(monkeypatch):
    """Test selected project logging previews first 10 IDs and reports remainder."""
    messages: list[str] = []
    monkeypatch.setattr(
        main, "info_always", lambda _logger, message: messages.append(message)
    )

    logger = Mock()
    project_ids = [f"ws/proj-{index}" for index in range(12)]
    main._log_selected_project_ids(logger, project_ids)

    assert messages[0] == "  Project IDs resolved: 12"
    assert messages[1] == "    - ws/proj-0"
    assert messages[10] == "    - ws/proj-9"
    assert messages[11] == "    ... and 2 more"
