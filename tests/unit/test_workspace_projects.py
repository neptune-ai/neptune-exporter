from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from neptune.exceptions import NeptuneClientUpgradeRequiredError

from neptune_exporter.exporters.exceptions import NeptuneExporterAuthError
import neptune_exporter.workspace_projects as workspace_projects


def _mock_page(
    entries: list[SimpleNamespace], matching_item_count: int
) -> SimpleNamespace:
    return SimpleNamespace(entries=entries, matchingItemCount=matching_item_count)


def _mock_operation_result(page: SimpleNamespace) -> Mock:
    operation = Mock()
    operation.response.return_value = SimpleNamespace(result=page)
    return operation


def test_list_workspace_projects_paginates_all_pages(monkeypatch):
    """Test that list_workspace_projects fetches all pages."""
    entry_1 = SimpleNamespace(organizationName="ws", name="proj-1")
    entry_2 = SimpleNamespace(organizationName="ws", name="proj-2")
    entry_3 = SimpleNamespace(organizationName="ws", name="proj-3")
    page_1 = _mock_page([entry_1, entry_2], matching_item_count=3)
    page_2 = _mock_page([entry_3], matching_item_count=3)

    backend_client = Mock()
    backend_client.api.listProjects.side_effect = [
        _mock_operation_result(page_1),
        _mock_operation_result(page_2),
    ]

    monkeypatch.setattr(
        workspace_projects.Credentials, "from_token", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        workspace_projects,
        "create_http_client_with_auth",
        Mock(return_value=(Mock(), Mock())),
    )
    monkeypatch.setattr(
        workspace_projects, "create_backend_client", Mock(return_value=backend_client)
    )
    monkeypatch.setattr(workspace_projects, "ssl_verify", Mock(return_value=True))

    result = workspace_projects.list_workspace_projects(
        workspace="ws", api_token="token", page_size=2
    )

    assert result == ["ws/proj-1", "ws/proj-2", "ws/proj-3"]
    assert backend_client.api.listProjects.call_count == 2

    first_call_kwargs = backend_client.api.listProjects.call_args_list[0].kwargs
    second_call_kwargs = backend_client.api.listProjects.call_args_list[1].kwargs
    assert first_call_kwargs["organizationIdentifier"] == "ws"
    assert first_call_kwargs["offset"] == 0
    assert first_call_kwargs["limit"] == 2
    assert second_call_kwargs["offset"] == 2
    assert second_call_kwargs["limit"] == 2


def test_list_workspace_projects_returns_empty_when_no_entries(monkeypatch):
    """Test that list_workspace_projects returns empty list when workspace has no projects."""
    page_1 = _mock_page([], matching_item_count=0)
    backend_client = Mock()
    backend_client.api.listProjects.return_value = _mock_operation_result(page_1)

    monkeypatch.setattr(
        workspace_projects.Credentials, "from_token", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        workspace_projects,
        "create_http_client_with_auth",
        Mock(return_value=(Mock(), Mock())),
    )
    monkeypatch.setattr(
        workspace_projects, "create_backend_client", Mock(return_value=backend_client)
    )
    monkeypatch.setattr(workspace_projects, "ssl_verify", Mock(return_value=True))

    result = workspace_projects.list_workspace_projects(workspace="ws", api_token="t")

    assert result == []
    backend_client.api.listProjects.assert_called_once()


def test_list_workspace_projects_maps_entries_to_workspace_project_ids(monkeypatch):
    """Test that workspace and project names are mapped to full project IDs."""
    entry_1 = SimpleNamespace(organizationName="workspace-a", name="project-x")
    entry_2 = SimpleNamespace(organizationName="workspace-a", name="project-y")
    page_1 = _mock_page([entry_1, entry_2], matching_item_count=2)
    backend_client = Mock()
    backend_client.api.listProjects.return_value = _mock_operation_result(page_1)

    monkeypatch.setattr(
        workspace_projects.Credentials, "from_token", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        workspace_projects,
        "create_http_client_with_auth",
        Mock(return_value=(Mock(), Mock())),
    )
    monkeypatch.setattr(
        workspace_projects, "create_backend_client", Mock(return_value=backend_client)
    )
    monkeypatch.setattr(workspace_projects, "ssl_verify", Mock(return_value=True))

    result = workspace_projects.list_workspace_projects(
        workspace="workspace-a", api_token="t"
    )

    assert result == ["workspace-a/project-x", "workspace-a/project-y"]


def test_list_workspace_projects_raises_auth_error_on_token_issue(monkeypatch):
    """Test that token errors are normalized to NeptuneExporterAuthError."""
    backend_client = Mock()
    backend_client.api.listProjects.side_effect = Exception(
        "Parameter 'X-Neptune-Api-Token' missing or invalid"
    )

    monkeypatch.setattr(
        workspace_projects.Credentials, "from_token", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        workspace_projects,
        "create_http_client_with_auth",
        Mock(return_value=(Mock(), Mock())),
    )
    monkeypatch.setattr(
        workspace_projects, "create_backend_client", Mock(return_value=backend_client)
    )
    monkeypatch.setattr(workspace_projects, "ssl_verify", Mock(return_value=True))

    with pytest.raises(NeptuneExporterAuthError):
        workspace_projects.list_workspace_projects(
            workspace="ws", api_token="bad-token"
        )


def test_list_workspace_projects_falls_back_when_version_check_fails(monkeypatch):
    """Test that discovery falls back to auth client creation without version check."""
    credentials = Mock()
    entry = SimpleNamespace(organizationName="ws", name="proj-1")
    page_1 = _mock_page([entry], matching_item_count=1)
    backend_client = Mock()
    backend_client.api.listProjects.return_value = _mock_operation_result(page_1)

    fallback_http_client = Mock()
    fallback_client_config = Mock()
    fallback_creator = Mock(return_value=(fallback_http_client, fallback_client_config))

    monkeypatch.setattr(
        workspace_projects.Credentials, "from_token", Mock(return_value=credentials)
    )
    monkeypatch.setattr(
        workspace_projects,
        "create_http_client_with_auth",
        Mock(
            side_effect=NeptuneClientUpgradeRequiredError(
                version="1.14.0",
                min_version="2.0.0",
            )
        ),
    )
    monkeypatch.setattr(
        workspace_projects,
        "_create_http_client_with_auth_without_version_check",
        fallback_creator,
    )
    monkeypatch.setattr(
        workspace_projects, "create_backend_client", Mock(return_value=backend_client)
    )
    monkeypatch.setattr(workspace_projects, "ssl_verify", Mock(return_value=True))

    result = workspace_projects.list_workspace_projects(
        workspace="ws", api_token="token"
    )

    assert result == ["ws/proj-1"]
    fallback_creator.assert_called_once_with(
        credentials=credentials,
        ssl_verify_value=True,
        proxies={},
    )
