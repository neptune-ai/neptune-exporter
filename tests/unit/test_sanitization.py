import hashlib
import pytest
from neptune_exporter.utils import sanitize_path_part


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("org/project", "org_project"),
        ("my-project_123", "my-project_123"),
        ("test@#$%", "test____"),
        ("simple", "simple"),
        ("with spaces", "with_spaces"),
        ("multiple/slashes/here", "multiple_slashes_here"),
        ("", ""),
        ("special!@#$%^&*()chars", "special__________chars"),
        ("already_safe", "already_safe"),
        ("123-numbers", "123-numbers"),
        ("my-org/my-project", "my-org_my-project"),
    ],
)
def test_sanitize_path_part(input_str: str, expected: str):
    """Test the sanitize_path_part function with various inputs."""
    result = sanitize_path_part(input_str)
    expected_with_digest = (
        expected
        + "-"
        + hashlib.blake2b(input_str.encode("utf-8"), digest_size=8).hexdigest()
    )
    assert result == expected_with_digest, (
        f"Expected '{expected_with_digest}', got '{result}' for input '{input_str}'"
    )
