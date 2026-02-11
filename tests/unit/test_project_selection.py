from unittest.mock import Mock

import click
import pytest

from neptune_exporter.project_selection import resolve_export_project_ids


def test_resolve_explicit_project_ids():
    """Test explicit project ID mode."""
    discover_workspace_projects = Mock(return_value=["ws/ignored"])

    result = resolve_export_project_ids(
        project_ids=("ws/proj-1", "ws/proj-2"),
        workspace=None,
        project_patterns=(),
        project_exclude_patterns=(),
        env_project=None,
        discover_workspace_projects=discover_workspace_projects,
    )

    assert result == ["ws/proj-1", "ws/proj-2"]
    discover_workspace_projects.assert_not_called()


def test_resolve_from_env_project():
    """Test fallback to NEPTUNE_PROJECT in explicit mode."""
    discover_workspace_projects = Mock(return_value=["ws/ignored"])

    result = resolve_export_project_ids(
        project_ids=(),
        workspace=None,
        project_patterns=(),
        project_exclude_patterns=(),
        env_project="env/proj",
        discover_workspace_projects=discover_workspace_projects,
    )

    assert result == ["env/proj"]
    discover_workspace_projects.assert_not_called()


def test_rejects_mixing_explicit_and_workspace_flags():
    """Test that explicit and discovery modes cannot be mixed."""
    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=("ws/proj",),
            workspace="ws",
            project_patterns=(),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=lambda _: [],
        )

    assert "Cannot use --project-ids/-p together with --workspace" in str(exc.value)


def test_rejects_patterns_without_workspace():
    """Test that project patterns require workspace mode."""
    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=(),
            workspace=None,
            project_patterns=("prod",),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=lambda _: [],
        )

    assert "require --workspace" in str(exc.value)


def test_workspace_include_patterns_or_semantics():
    """Test include patterns are combined with OR semantics."""
    discover_workspace_projects = Mock(
        return_value=["ws/prod-a", "ws/dev-b", "ws/stage-c"]
    )

    result = resolve_export_project_ids(
        project_ids=(),
        workspace="ws",
        project_patterns=("prod", "stage"),
        project_exclude_patterns=(),
        env_project=None,
        discover_workspace_projects=discover_workspace_projects,
    )

    assert result == ["ws/prod-a", "ws/stage-c"]


def test_workspace_patterns_match_project_name_only():
    """Test include patterns do not match workspace prefix."""
    discover_workspace_projects = Mock(return_value=["team/prod-a", "team/dev-b"])

    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=(),
            workspace="team",
            project_patterns=("team",),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=discover_workspace_projects,
        )

    assert "No projects found in workspace 'team'" in str(exc.value)


def test_workspace_exclude_patterns_or_semantics():
    """Test exclude patterns are combined with OR semantics."""
    discover_workspace_projects = Mock(
        return_value=["ws/a", "ws/b-archive", "ws/c-old", "ws/d"]
    )

    result = resolve_export_project_ids(
        project_ids=(),
        workspace="ws",
        project_patterns=(),
        project_exclude_patterns=("archive", "old"),
        env_project=None,
        discover_workspace_projects=discover_workspace_projects,
    )

    assert result == ["ws/a", "ws/d"]


def test_workspace_exclude_patterns_match_project_name_only():
    """Test exclude patterns do not exclude by workspace prefix."""
    discover_workspace_projects = Mock(
        return_value=["team/team-project", "team/prod-project"]
    )

    result = resolve_export_project_ids(
        project_ids=(),
        workspace="team",
        project_patterns=(),
        project_exclude_patterns=("team",),
        env_project=None,
        discover_workspace_projects=discover_workspace_projects,
    )

    assert result == ["team/prod-project"]


def test_invalid_include_regex_fails():
    """Test invalid include regex fails with actionable message."""
    discover_workspace_projects = Mock(return_value=["ws/proj"])

    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=(),
            workspace="ws",
            project_patterns=("[",),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=discover_workspace_projects,
        )

    assert "Invalid regex for --project-pattern" in str(exc.value)
    discover_workspace_projects.assert_not_called()


def test_invalid_exclude_regex_fails():
    """Test invalid exclude regex fails with actionable message."""
    discover_workspace_projects = Mock(return_value=["ws/proj"])

    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=(),
            workspace="ws",
            project_patterns=(),
            project_exclude_patterns=("[",),
            env_project=None,
            discover_workspace_projects=discover_workspace_projects,
        )

    assert "Invalid regex for --project-exclude-pattern" in str(exc.value)
    discover_workspace_projects.assert_not_called()


def test_empty_result_after_filters_fails():
    """Test empty discovery result after filtering is a validation error."""
    discover_workspace_projects = Mock(return_value=["ws/dev-project"])

    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=(),
            workspace="ws",
            project_patterns=("prod",),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=discover_workspace_projects,
        )

    assert "No projects found in workspace 'ws'" in str(exc.value)


def test_rejects_empty_explicit_project_id():
    """Test explicit project IDs cannot be empty."""
    with pytest.raises(click.BadParameter) as exc:
        resolve_export_project_ids(
            project_ids=("   ",),
            workspace=None,
            project_patterns=(),
            project_exclude_patterns=(),
            env_project=None,
            discover_workspace_projects=lambda _: [],
        )

    assert "Project ID cannot be empty" in str(exc.value)
