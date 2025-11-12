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


import hashlib


def sanitize_path_part(part: str) -> str:
    """Sanitize a string to be safe for use in file paths.

    Replaces any character that is not alphanumeric, underscore, or dash with an underscore.
    This ensures the string can be safely used as part of a file path.

    Args:
        part: The string to sanitize

    Returns:
        A sanitized string safe for use in file paths

    Examples:
        >>> sanitize_path_part("org/project")
        "org_project-12345678"
        >>> sanitize_path_part("my-project_123")
        "my-project_123-12345678"
        >>> sanitize_path_part("test@#$%")
        "test____-12345678"
    """
    return "".join(
        c if c.isalnum() or c in ("_", "-") else "_" for c in part
    ) + _digest_suffix(part)


def _digest_suffix(string: str) -> str:
    digest = hashlib.blake2b(string.encode("utf-8"), digest_size=8).hexdigest()
    return "-" + digest
