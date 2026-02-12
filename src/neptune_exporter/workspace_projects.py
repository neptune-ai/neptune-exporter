#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workspace-scoped Neptune project discovery."""

from neptune.common.oauth import NeptuneAuthenticator
from neptune.exceptions import NeptuneClientUpgradeRequiredError
from neptune.internal.backends.hosted_client import (
    BACKEND_SWAGGER_PATH,
    DEFAULT_REQUEST_KWARGS,
    _get_token_client,
    create_backend_client,
    create_http_client,
    create_http_client_with_auth,
    get_client_config,
)
from neptune.internal.backends.utils import build_operation_url, ssl_verify
from neptune.internal.credentials import Credentials

from neptune_exporter.exporters.exceptions import raise_if_neptune_api_token_error


def _create_http_client_with_auth_without_version_check(
    credentials: Credentials,
    ssl_verify_value: bool,
    proxies: dict[str, str],
):
    client_config = get_client_config(
        credentials=credentials,
        ssl_verify=ssl_verify_value,
        proxies=proxies,
    )

    config_api_url = credentials.api_url_opt or credentials.token_origin_address
    endpoint_url = None
    if config_api_url != client_config.api_url:
        endpoint_url = build_operation_url(client_config.api_url, BACKEND_SWAGGER_PATH)

    http_client = create_http_client(ssl_verify=ssl_verify_value, proxies=proxies)
    http_client.authenticator = NeptuneAuthenticator(
        credentials.api_token,
        _get_token_client(
            credentials=credentials,
            ssl_verify=ssl_verify_value,
            proxies=proxies,
            endpoint_url=endpoint_url,
        ),
        ssl_verify_value,
        proxies,
    )

    return http_client, client_config


def list_workspace_projects(
    workspace: str,
    api_token: str | None,
    page_size: int = 200,
) -> list[str]:
    """List all projects in a workspace using paginated backend API calls."""
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    try:
        credentials = Credentials.from_token(api_token=api_token)
        ssl_verify_value = ssl_verify()
        proxies: dict[str, str] = {}

        try:
            http_client, client_config = create_http_client_with_auth(
                credentials=credentials,
                ssl_verify=ssl_verify_value,
                proxies=proxies,
            )
        except NeptuneClientUpgradeRequiredError:
            http_client, client_config = (
                _create_http_client_with_auth_without_version_check(
                    credentials=credentials,
                    ssl_verify_value=ssl_verify_value,
                    proxies=proxies,
                )
            )

        backend_client = create_backend_client(
            client_config=client_config, http_client=http_client
        )

        projects: list[str] = []
        offset = 0

        while True:
            response = (
                backend_client.api.listProjects(
                    organizationIdentifier=workspace,
                    offset=offset,
                    limit=page_size,
                    **DEFAULT_REQUEST_KWARGS,
                )
                .response()
                .result
            )
            entries = response.entries
            projects.extend(
                f"{entry.organizationName}/{entry.name}" for entry in entries
            )

            offset += len(entries)
            if offset >= response.matchingItemCount or not entries:
                break

        return projects
    except Exception as e:
        raise_if_neptune_api_token_error(e)
        raise
