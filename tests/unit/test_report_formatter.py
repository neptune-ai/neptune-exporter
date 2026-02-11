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
import click

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
                "total_records": 100,
                "attribute_breakdown": {"float_series": 5, "string": 2},
                "run_breakdown": {"RUN-1": 50, "RUN-2": 30, "RUN-3": 20},
                "file_info": {
                    "total_files": 2,
                    "total_size_bytes": 1024000,
                    "records_per_file": [60, 40],
                },
                "step_statistics": {
                    "total_steps": 80,
                    "min_step": 0,
                    "max_step": 100,
                    "unique_steps": 50,
                },
            },
            Path("/test/exports/project2"): {
                "project_id": "project2",
                "total_runs": 1,
                "attribute_types": ["file"],
                "runs": ["RUN-4"],
                "total_records": 10,
                "attribute_breakdown": {"file": 1},
                "run_breakdown": {"RUN-4": 10},
                "file_info": {
                    "total_files": 1,
                    "total_size_bytes": 512000,
                    "records_per_file": [10],
                },
                "step_statistics": {
                    "total_steps": 0,
                    "min_step": None,
                    "max_step": None,
                    "unique_steps": 0,
                },
            },
        },
    }

    input_path = Path("/test/exports")
    formatted = ReportFormatter.format_data_summary(summary, input_path)

    assert "📊 Data Summary from /test/exports" in formatted
    assert "📈 Total projects: 2" in formatted
    assert "🏗️  Project: project1" in formatted
    assert "📁 Directory: /test/exports/project1" in formatted
    assert "🏃 Runs: 3" in formatted
    assert "📝 Total records: 100" in formatted
    assert "📄 Files: 2 (0.98 MB)" in formatted
    assert "🏷️  Attribute types: float_series, string" in formatted
    assert "📊 Attribute breakdown:" in formatted
    assert "float_series: 5 unique attributes" in formatted
    assert "string: 2 unique attributes" in formatted
    assert "📈 Step statistics:" in formatted
    assert "Total steps: 80" in formatted
    assert "Range: 0 - 100" in formatted
    assert "🏃 Run breakdown (top 5 by record count):" in formatted
    assert "RUN-1: 50 records" in formatted
    assert "🏗️  Project: project2" in formatted
    assert "📁 Directory: /test/exports/project2" in formatted
    assert "🏃 Runs: 1" in formatted
    assert "📝 Total records: 10" in formatted
    assert "📄 Files: 1 (0.49 MB)" in formatted
    assert "🏷️  Attribute types: file" in formatted
    assert "file: 1 unique attributes" in formatted


def test_format_data_summary_with_error():
    """Test formatting data summary with error."""
    summary = {"total_projects": 1, "projects": {Path("/test/exports/project1"): None}}

    input_path = Path("/test/exports")
    formatted = ReportFormatter.format_data_summary(summary, input_path)

    assert (
        "❌ Project directory: /test/exports/project1 (ERROR: Failed to read data)"
        in formatted
    )


@patch("neptune_exporter.validation.report_formatter.click.echo", spec=click.echo)
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
                "total_records": 5,
                "attribute_breakdown": {"string": 1},
                "run_breakdown": {"RUN-1": 5},
                "file_info": {
                    "total_files": 1,
                    "total_size_bytes": 1024,
                    "records_per_file": [5],
                },
                "step_statistics": {
                    "total_steps": 0,
                    "min_step": None,
                    "max_step": None,
                    "unique_steps": 0,
                },
            }
        },
    }

    input_path = Path("/test/exports")
    ReportFormatter.print_data_summary(summary, input_path)

    mock_echo.assert_called_once()
    call_args = mock_echo.call_args[0][0]
    assert "📊 Data Summary from /test/exports" in call_args


def test_format_model_registry_summary():
    """Test formatting model registry summary."""
    summary = {
        "total_projects": 1,
        "total_models": 2,
        "total_model_versions": 3,
        "projects": {
            Path("/test/model_exports/project1"): {
                "project_id": "project1",
                "models": {
                    "total_entities": 2,
                    "total_records": 12,
                    "attribute_types": ["string", "float"],
                },
                "model_versions": {
                    "total_entities": 3,
                    "total_records": 30,
                    "attribute_types": ["string", "float_series"],
                },
            }
        },
    }

    input_path = Path("/test/model_exports")
    formatted = ReportFormatter.format_model_registry_summary(summary, input_path)

    assert "🧠 Model Registry Summary from /test/model_exports" in formatted
    assert "📈 Total projects: 1" in formatted
    assert "📦 Total models: 2" in formatted
    assert "🏷️  Total model versions: 3" in formatted
    assert "🏗️  Project: project1" in formatted
    assert "📦 Models: 2 (12 records)" in formatted
    assert "🏷️  Model versions: 3 (30 records)" in formatted


def test_format_model_registry_summary_empty():
    """Test formatting model registry summary when no model data exists."""
    summary = {
        "total_projects": 0,
        "total_models": 0,
        "total_model_versions": 0,
        "projects": {},
    }

    input_path = Path("/test/model_exports")
    formatted = ReportFormatter.format_model_registry_summary(summary, input_path)

    assert "ℹ️  No exported model registry data found." in formatted
