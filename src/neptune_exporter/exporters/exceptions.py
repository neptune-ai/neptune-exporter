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

"""Exceptions and helpers for Neptune exporters."""


def _is_neptune_api_token_error(exc: BaseException) -> bool:
    """Return True if this exception (or its cause chain) indicates X-Neptune-Api-Token 400."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).lower()
        if "x-neptune-api-token" in msg and ("missing" in msg or "invalid" in msg):
            return True
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return False


class NeptuneExporterAuthError(Exception):
    """Raised when the Neptune API returns 400 due to X-Neptune-Api-Token missing or invalid.

    This usually means the user is using the wrong exporter for their backend,
    or an API token that does not match the Neptune version (e.g. Neptune 2 token with neptune3).
    """

    def __init__(self, original: Exception | None = None) -> None:
        super().__init__(
            "The Neptune server rejected the request: missing or invalid `X-Neptune-Api-Token`.\n\n"
            "How to fix this:\n"
            "1. Verify you are using the correct exporter:\n"
            "   - `neptune2` for Neptune 2\n"
            "   - `neptune3` for Neptune 3\n"
            "2. Make sure the API token matches the exporter version\n"
            "   (e.g. a Neptune 3 token when using the `neptune3` exporter).\n\n"
            "Technical details:\n"
            "- HTTP status: 400\n"
            "- Server message: \"Parameter 'X-Neptune-Api-Token' missing or invalid\""
        )
        if original is not None:
            self.__cause__ = original


def raise_if_neptune_api_token_error(exc: Exception) -> None:
    """If the exception indicates X-Neptune-Api-Token 400, raise NeptuneExporterAuthError."""
    if _is_neptune_api_token_error(exc):
        raise NeptuneExporterAuthError(exc) from exc
