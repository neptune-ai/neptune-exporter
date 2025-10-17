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

from pathlib import Path
from unittest.mock import patch

from neptune_exporter.validation import ReportFormatter


def test_format_data_summary():
    """Test formatting data summary."""
    summary = {
        "total_projects": 2,
        "projects": {
            Path("/test/exports/project1"): {
                "project_id": "project1",
                "total_runs": 3,
                "attribute_types": ["float_series", "string"],
                "runs": ["RUN-1", "RUN-2", "RUN-3"],
            },
            Path("/test/exports/project2"): {
                "project_id": "project2",
                "total_runs": 1,
                "attribute_types": ["file"],
                "runs": ["RUN-4"],
            },
        },
    }

    input_path = Path("/test/exports")
    formatted = ReportFormatter.format_data_summary(summary, input_path)

    assert "Data Summary from /test/exports" in formatted
    assert "Total projects: 2" in formatted
    assert "Project: project1" in formatted
    assert "Directory: /test/exports/project1" in formatted
    assert "Runs: 3" in formatted
    assert "Attribute types: float_series, string" in formatted
    assert "Run IDs: RUN-1, RUN-2, RUN-3" in formatted
    assert "Project: project2" in formatted
    assert "Directory: /test/exports/project2" in formatted
    assert "Runs: 1" in formatted
    assert "Attribute types: file" in formatted
    assert "Run IDs: RUN-4" in formatted


def test_format_data_summary_with_error():
    """Test formatting data summary with error."""
    summary = {"total_projects": 1, "projects": {Path("/test/exports/project1"): None}}

    input_path = Path("/test/exports")
    formatted = ReportFormatter.format_data_summary(summary, input_path)

    assert (
        "Project directory: /test/exports/project1 (ERROR: Failed to read data)"
        in formatted
    )


@patch("neptune_exporter.validation.report_formatter.click.echo")
def test_print_data_summary(mock_echo):
    """Test printing data summary."""
    summary = {
        "total_projects": 1,
        "projects": {
            Path("/test/exports/project1"): {
                "project_id": "project1",
                "total_runs": 1,
                "attribute_types": ["string"],
                "runs": ["RUN-1"],
            }
        },
    }

    input_path = Path("/test/exports")
    ReportFormatter.print_data_summary(summary, input_path)

    mock_echo.assert_called_once()
    call_args = mock_echo.call_args[0][0]
    assert "Data Summary from /test/exports" in call_args
