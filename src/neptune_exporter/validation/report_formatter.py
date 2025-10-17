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
from typing import Dict, Any
import click


class ReportFormatter:
    """Formats and displays reports for exported Neptune data."""

    @staticmethod
    def format_data_summary(summary: Dict[str, Any], input_path: Path) -> str:
        """Format a data summary report."""
        lines = []
        lines.append(f"Data Summary from {input_path.absolute()}")
        lines.append(f"Total projects: {summary['total_projects']}")
        lines.append("")

        for project_directory, project_info in summary["projects"].items():
            if project_info is None:
                lines.append(
                    f"Project directory: {project_directory} (ERROR: Failed to read data)"
                )
            else:
                project_id = project_info.get("project_id", "Unknown")
                if project_id is None:
                    project_id = "Unknown"

                lines.append(f"Project: {project_id}")
                lines.append(f"  Directory: {project_directory}")
                lines.append(f"  Runs: {project_info['total_runs']}")
                lines.append(
                    f"  Attribute types: {', '.join(project_info['attribute_types'])}"
                )
                if project_info["runs"]:
                    lines.append(f"  Run IDs: {', '.join(project_info['runs'])}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def print_data_summary(summary: Dict[str, Any], input_path: Path) -> None:
        """Print a formatted data summary to the console."""
        report = ReportFormatter.format_data_summary(summary, input_path)
        click.echo(report)
