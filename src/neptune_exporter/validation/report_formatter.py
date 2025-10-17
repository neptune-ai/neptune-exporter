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
        """Format a detailed data summary report."""
        lines = []
        lines.append(f"📊 Data Summary from {input_path.absolute()}")
        lines.append(f"📈 Total projects: {summary['total_projects']}")
        lines.append("")

        for project_directory, project_info in summary["projects"].items():
            if project_info is None:
                lines.append(
                    f"❌ Project directory: {project_directory} (ERROR: Failed to read data)"
                )
            else:
                project_id = project_info.get("project_id", "Unknown")
                if project_id is None:
                    project_id = "Unknown"

                lines.append(f"🏗️  Project: {project_id}")
                lines.append(f"   📁 Directory: {project_directory}")
                lines.append(f"   🏃 Runs: {project_info['total_runs']}")
                lines.append(
                    f"   📝 Total records: {project_info.get('total_records', 0):,}"
                )

                # File information
                file_info = project_info.get("file_info", {})
                if file_info:
                    total_size_mb = file_info.get("total_size_bytes", 0) / 1024 / 1024
                    lines.append(
                        f"   📄 Files: {file_info.get('total_files', 0)} ({total_size_mb:.2f} MB)"
                    )

                # Attribute types and breakdown
                attribute_types = project_info.get("attribute_types", [])
                if attribute_types:
                    lines.append(f"   🏷️  Attribute types: {', '.join(attribute_types)}")

                    # Attribute breakdown
                    attribute_breakdown = project_info.get("attribute_breakdown", {})
                    if attribute_breakdown:
                        lines.append("   📊 Attribute breakdown:")
                        for attr_type, count in sorted(attribute_breakdown.items()):
                            lines.append(
                                f"      {attr_type}: {count} unique attributes"
                            )

                # Step statistics
                step_stats = project_info.get("step_statistics", {})
                if step_stats and step_stats.get("total_steps", 0) > 0:
                    lines.append("   📈 Step statistics:")
                    lines.append(
                        f"      Total steps: {step_stats.get('total_steps', 0):,}"
                    )
                    lines.append(
                        f"      Unique steps: {step_stats.get('unique_steps', 0):,}"
                    )
                    if step_stats.get("min_step") is not None:
                        lines.append(
                            f"      Range: {step_stats.get('min_step')} - {step_stats.get('max_step')}"
                        )

                # Run breakdown (show top 5 runs by record count)
                run_breakdown = project_info.get("run_breakdown", {})
                if run_breakdown:
                    sorted_runs = sorted(
                        run_breakdown.items(), key=lambda x: x[1], reverse=True
                    )
                    lines.append("   🏃 Run breakdown (top 5 by record count):")
                    for run_id, count in sorted_runs[:5]:
                        lines.append(f"      {run_id}: {count:,} records")
                    if len(sorted_runs) > 5:
                        lines.append(f"      ... and {len(sorted_runs) - 5} more runs")

                # Show all run IDs if there are few
                runs = project_info.get("runs", [])
                if runs and len(runs) <= 10:
                    lines.append(f"   🆔 Run IDs: {', '.join(runs)}")
                elif runs:
                    lines.append(
                        f"   🆔 Run IDs: {', '.join(runs[:5])} ... and {len(runs) - 5} more"
                    )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def print_data_summary(summary: Dict[str, Any], input_path: Path) -> None:
        """Print a formatted data summary to the console."""
        report = ReportFormatter.format_data_summary(summary, input_path)
        click.echo(report)
