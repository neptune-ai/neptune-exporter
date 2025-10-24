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

import os
import click
from pathlib import Path

from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter
from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from neptune_exporter.export_manager import ExportManager
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.loaders.mlflow import MLflowLoader
from neptune_exporter.loader_manager import LoaderManager
from neptune_exporter.summary_manager import SummaryManager
from neptune_exporter.validation import ReportFormatter


@click.group()
def cli():
    """Neptune Exporter - Export and migrate Neptune experiment data."""
    pass


@cli.command()
@click.option(
    "--project-ids",
    "-p",
    multiple=True,
    help="Neptune project IDs to export. Can be specified multiple times. If not provided, reads from NEPTUNE_PROJECT environment variable.",
)
@click.option(
    "--runs",
    "-r",
    help="Filter runs by pattern (e.g., 'RUN-*' or specific run ID).",
)
@click.option(
    "--attributes",
    "-a",
    multiple=True,
    help="Filter attributes by name. Can be specified multiple times. "
    "If a single string is provided, it's treated as a regex pattern. "
    "If multiple strings are provided, they're treated as exact attribute names to match.",
)
@click.option(
    "--export-classes",
    "-e",
    type=click.Choice(
        ["parameters", "metrics", "series", "files"], case_sensitive=False
    ),
    multiple=True,
    default=["parameters", "metrics", "series", "files"],
    help="Types of data to export. Default: all types.",
)
@click.option(
    "--exporter",
    type=click.Choice(["neptune2", "neptune3"], case_sensitive=False),
    default="neptune3",
    help="Neptune exporter to use. Default: neptune3.",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default="./exports",
    help="Base path for exported data. Default: ./exports",
)
@click.option(
    "--api-token",
    help="Neptune API token. If not provided, will use environment variable NEPTUNE_API_TOKEN.",
)
def export(
    project_ids: tuple[str, ...],
    runs: str | None,
    attributes: tuple[str, ...],
    export_classes: tuple[str, ...],
    exporter: str,
    output_path: Path,
    api_token: str | None,
) -> None:
    """Export Neptune experiment data to parquet files.

    This tool exports data from Neptune projects including parameters, metrics,
    series data, and files to parquet format for further analysis.

    Examples:

    \b
    # Export all data from a project
    neptune-exporter -p "my-org/my-project"

    \b
    # Export only parameters and metrics from specific runs
    neptune-exporter -p "my-org/my-project" -r "RUN-*" -e parameters -e metrics

    \b
    # Export specific attributes only (exact match)
    neptune-exporter -p "my-org/my-project" -a "learning_rate" -a "batch_size"

    \b
    # Export attributes matching a pattern (regex)
    neptune-exporter -p "my-org/my-project" -a "config/.*"

    \b
    # Use Neptune 2.x exporter
    neptune-exporter -p "my-org/my-project" --exporter neptune2

    \b
    # Use environment variable for project ID
    NEPTUNE_PROJECT="my-org/my-project" neptune-exporter
    """
    # Convert tuples to lists and handle None values
    project_ids_list = list(project_ids)

    # If no project IDs provided, try to read from environment variable
    if not project_ids_list:
        env_project = os.getenv("NEPTUNE_PROJECT")
        if env_project:
            project_ids_list = [env_project]
        else:
            raise click.BadParameter(
                "No project IDs provided. Either use --project-ids/-p option or set NEPTUNE_PROJECT environment variable."
            )

    attributes_list = list(attributes) if attributes else None
    export_classes_list = list(export_classes)

    # Validate project IDs are not empty
    for project_id in project_ids_list:
        if not project_id.strip():
            raise click.BadParameter(
                "Project ID cannot be empty. Please provide a valid project ID."
            )

    # Validate export classes
    valid_export_classes = {"parameters", "metrics", "series", "files"}
    export_classes_set = set(export_classes_list)
    if not export_classes_set.issubset(valid_export_classes):
        invalid = export_classes_set - valid_export_classes
        raise click.BadParameter(f"Invalid export classes: {', '.join(invalid)}")

    # Create exporter instance
    if exporter == "neptune2":
        exporter_instance: NeptuneExporter = Neptune2Exporter(api_token=api_token)
    elif exporter == "neptune3":
        exporter_instance = Neptune3Exporter(api_token=api_token)
    else:
        raise click.BadParameter(f"Unknown exporter: {exporter}")

    # Create storage instance
    storage = ParquetWriter(base_path=output_path)

    # Create and run export manager
    export_manager = ExportManager(
        exporter=exporter_instance,
        storage=storage,
        files_destination=output_path / "files",
    )

    click.echo(f"Starting export of {len(project_ids_list)} project(s)...")
    click.echo(f"Export classes: {', '.join(export_classes_list)}")
    click.echo(f"Output path: {output_path.absolute()}")

    try:
        export_manager.run(
            project_ids=project_ids_list,
            runs=runs,
            attributes=attributes_list,
            export_classes=export_classes_set,  # type: ignore
        )
        click.echo("Export completed successfully!")
    except Exception as e:
        click.echo(f"Export failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(path_type=Path),
    default="./exports",
    help="Base path for exported parquet data. Default: ./exports",
)
@click.option(
    "--project-ids",
    "-p",
    multiple=True,
    help="Project IDs to load. If not specified, loads all available projects.",
)
@click.option(
    "--runs",
    "-r",
    multiple=True,
    help="Run IDs to filter by. Can be specified multiple times.",
)
@click.option(
    "--mlflow-tracking-uri",
    help="MLflow tracking URI. If not provided, uses default MLflow tracking URI.",
)
@click.option(
    "--name-prefix",
    help="Optional prefix for MLflow experiment and run names (to handle org/project structure).",
)
def load(
    input_path: Path,
    project_ids: tuple[str, ...],
    runs: tuple[str, ...],
    mlflow_tracking_uri: str | None,
    name_prefix: str | None,
) -> None:
    """Load exported Neptune data from parquet files to MLflow.

    This tool loads previously exported Neptune data from parquet files
    and uploads it to MLflow for further analysis and tracking.

    Examples:

    \b
    # Load all data from exported parquet files
    neptune-exporter load

    \b
    # Load specific projects
    neptune-exporter load -p "my-org/my-project1" -p "my-org/my-project2"

    \b
    # Load specific runs
    neptune-exporter load -r "RUN-123" -r "RUN-456"

    \b
    # Load only parameters and metrics
    neptune-exporter load -t parameters -t float_series

    \b
    # Load to specific MLflow tracking URI
    neptune-exporter load --mlflow-tracking-uri "http://localhost:5000"
    """
    # Convert tuples to lists and handle None values
    project_ids_list = list(project_ids) if project_ids else None
    runs_list = list(runs) if runs else None

    # Validate input path exists
    if not input_path.exists():
        raise click.BadParameter(f"Input path does not exist: {input_path}")

    # Create parquet reader
    parquet_reader = ParquetReader(base_path=input_path)

    # Create MLflow loader
    data_loader = MLflowLoader(
        tracking_uri=mlflow_tracking_uri,
        name_prefix=name_prefix,
    )

    # Create loader manager
    loader_manager = LoaderManager(
        parquet_reader=parquet_reader,
        data_loader=data_loader,
        files_directory=input_path / "files",
    )

    click.echo(f"Starting MLflow loading from {input_path.absolute()}")
    if project_ids_list:
        click.echo(f"Project IDs: {', '.join(project_ids_list)}")
    if runs_list:
        click.echo(f"Run IDs: {', '.join(runs_list)}")

    try:
        loader_manager.load(
            project_ids=project_ids_list,
            runs=runs_list,
        )
        click.echo("MLflow loading completed successfully!")
    except Exception as e:
        click.echo(f"MLflow loading failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(path_type=Path),
    default="./exports",
    help="Base path for exported parquet data. Default: ./exports",
)
def summary(input_path: Path) -> None:
    """Show summary of exported Neptune data.

    This command shows a summary of available data in the exported parquet files,
    including project counts, run counts, and attribute types.
    """
    # Validate input path exists
    if not input_path.exists():
        raise click.BadParameter(f"Input path does not exist: {input_path}")

    # Create parquet reader and summary manager
    parquet_reader = ParquetReader(base_path=input_path)
    summary_manager = SummaryManager(parquet_reader=parquet_reader)

    try:
        # Show general data summary
        summary_data = summary_manager.get_data_summary()
        ReportFormatter.print_data_summary(summary_data, input_path)

    except Exception as e:
        click.echo(f"Failed to generate summary: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
