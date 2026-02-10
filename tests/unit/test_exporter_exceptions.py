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

import pytest

from neptune_exporter.exporters.exceptions import (
    NeptuneExporterAuthError,
    raise_if_neptune_api_token_error,
)


def test_neptune_exporter_auth_error_message():
    """NeptuneExporterAuthError message instructs user to check exporter and token."""
    err = NeptuneExporterAuthError()
    assert "neptune2" in str(err)
    assert "neptune3" in str(err)
    assert "API token" in str(err)
    assert "Neptune 3" in str(err)


def test_raise_if_neptune_api_token_error_raises_on_token_message():
    """raise_if_neptune_api_token_error raises when message contains X-Neptune-Api-Token missing/invalid."""
    exc = Exception(
        'Response content: {"code":400,"message":"Parameter \'X-Neptune-Api-Token\' missing or invalid"}'
    )
    with pytest.raises(NeptuneExporterAuthError) as excinfo:
        raise_if_neptune_api_token_error(exc)
    assert excinfo.value.__cause__ is exc


def test_raise_if_neptune_api_token_error_raises_on_cause_chain():
    """raise_if_neptune_api_token_error detects token error in __cause__ chain."""
    inner = Exception("X-Neptune-Api-Token missing or invalid")
    outer = Exception("Failed to fetch config")
    outer.__cause__ = inner
    with pytest.raises(NeptuneExporterAuthError):
        raise_if_neptune_api_token_error(outer)


def test_raise_if_neptune_api_token_error_does_not_raise_on_other_errors():
    """raise_if_neptune_api_token_error does not raise for unrelated exceptions."""
    exc = Exception("Network timeout")
    raise_if_neptune_api_token_error(exc)  # does not raise
