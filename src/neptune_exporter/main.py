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
    "--classes",
    "-c",
    type=click.Choice(
        ["parameters", "metrics", "series", "files"], case_sensitive=False
    ),
    multiple=True,
    help="Types of data to include in export. Can be specified multiple times.",
)
@click.option(
    "--exclude",
    type=click.Choice(
        ["parameters", "metrics", "series", "files"], case_sensitive=False
    ),
    multiple=True,
    help="Types of data to exclude from export. Can be specified multiple times.",
)
@click.option(
    "--exporter",
    type=click.Choice(["neptune2", "neptune3"], case_sensitive=False),
    default="neptune3",
    help="Neptune exporter to use. Default: neptune3.",
)
@click.option(
    "--data-path",
    "-d",
    type=click.Path(path_type=Path),
    default="./exports/data",
    help="Path for exported parquet data. Default: ./exports/data",
)
@click.option(
    "--files-path",
    "-f",
    type=click.Path(path_type=Path),
    default="./exports/files",
    help="Path for downloaded files. Default: ./exports/files",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing exported runs. By default, already-exported runs are skipped.",
)
@click.option(
    "--api-token",
    help="Neptune API token. If not provided, will use environment variable NEPTUNE_API_TOKEN.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging including Neptune internal logs.",
)
def export(
    project_ids: tuple[str, ...],
    runs: str | None,
    attributes: tuple[str, ...],
    classes: tuple[str, ...],
    exclude: tuple[str, ...],
    exporter: str,
    data_path: Path,
    files_path: Path,
    overwrite: bool,
    api_token: str | None,
    verbose: bool,
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
    neptune-exporter -p "my-org/my-project" -r "RUN-*" -c parameters -c metrics

    \b
    # Export everything except files
    neptune-exporter -p "my-org/my-project" --exclude files

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

    # Handle attributes: single string = regex, multiple strings = exact matches
    if not attributes:
        attributes_list: list[str] | str | None = None
    elif len(attributes) == 1:
        # Single string - treat as regex pattern
        attributes_list = attributes[0]
    else:
        # Multiple strings - treat as exact attribute names
        attributes_list = list(attributes)

    # Determine export classes based on include/exclude logic
    all_classes = {"parameters", "metrics", "series", "files"}
    classes_set = set(classes) if classes else set()
    exclude_set = set(exclude) if exclude else set()

    # Validate that both classes and exclude are not specified
    if classes_set and exclude_set:
        raise click.BadParameter(
            "Cannot specify both --classes and --exclude. Use --classes to include specific types or --exclude to exclude specific types."
        )

    # Determine export classes based on include/exclude logic
    if classes_set:
        export_classes_list = list(classes_set)
    elif exclude_set:
        export_classes_list = list(all_classes - exclude_set)
    else:
        # Default to all classes if neither is specified
        export_classes_list = list(all_classes)

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
        exporter_instance: NeptuneExporter = Neptune2Exporter(
            api_token=api_token, verbose=verbose
        )
    elif exporter == "neptune3":
        exporter_instance = Neptune3Exporter(api_token=api_token, verbose=verbose)
    else:
        raise click.BadParameter(f"Unknown exporter: {exporter}")

    # Create storage and reader instances
    writer = ParquetWriter(base_path=data_path)
    reader = ParquetReader(base_path=data_path)

    # Check for existing data and notify user about resume/overwrite behavior
    existing_projects = reader.list_project_directories()
    if existing_projects:
        if overwrite:
            click.echo(f"⚠️  Found existing data at {data_path.absolute()}")
            click.echo(
                "   Will delete and overwrite existing project data (--overwrite flag set)"
            )
        else:
            click.echo(f"ℹ️  Found existing data at {data_path.absolute()}")
            click.echo("   Will resume export, skipping already-exported runs")
            click.echo("   Use --overwrite to delete and re-export all data")

    # Create export manager with resume support
    export_manager = ExportManager(
        exporter=exporter_instance,
        reader=reader,
        writer=writer,
        files_destination=files_path,
        resume=not overwrite,
    )

    click.echo(f"Starting export of {len(project_ids_list)} project(s)...")
    click.echo(f"Export classes: {', '.join(export_classes_list)}")
    click.echo(f"Data path: {data_path.absolute()}")
    click.echo(f"Files path: {files_path.absolute()}")

    try:
        runs_exported = export_manager.run(
            project_ids=project_ids_list,
            runs=runs,
            attributes=attributes_list,
            export_classes=export_classes_set,  # type: ignore
        )

        if runs_exported == 0:
            click.echo("ℹ️  No runs found matching the specified criteria.")
            if runs:
                click.echo(f"   Filter: {runs}")
            click.echo(
                "   Try adjusting your run filter or check if the project contains any runs."
            )
        else:
            click.echo("Export completed successfully!")
    except Exception as e:
        click.echo(f"Export failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--data-path",
    "-d",
    type=click.Path(path_type=Path),
    default="./exports/data",
    help="Path for exported parquet data. Default: ./exports/data",
)
@click.option(
    "--files-path",
    "-f",
    type=click.Path(path_type=Path),
    default="./exports/files",
    help="Path for downloaded files. Default: ./exports/files",
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging including Neptune internal logs.",
)
def load(
    data_path: Path,
    files_path: Path,
    project_ids: tuple[str, ...],
    runs: tuple[str, ...],
    mlflow_tracking_uri: str | None,
    name_prefix: str | None,
    verbose: bool,
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

    # Validate data path exists
    if not data_path.exists():
        raise click.BadParameter(f"Data path does not exist: {data_path}")

    # Create parquet reader
    parquet_reader = ParquetReader(base_path=data_path)

    # Create MLflow loader
    data_loader = MLflowLoader(
        tracking_uri=mlflow_tracking_uri,
        name_prefix=name_prefix,
        verbose=verbose,
    )

    # Create loader manager
    loader_manager = LoaderManager(
        parquet_reader=parquet_reader,
        data_loader=data_loader,
        files_directory=files_path,
    )

    click.echo(f"Starting MLflow loading from {data_path.absolute()}")
    click.echo(f"Files directory: {files_path.absolute()}")
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
    "--data-path",
    "-d",
    type=click.Path(path_type=Path),
    default="./exports/data",
    help="Path for exported parquet data. Default: ./exports/data",
)
def summary(data_path: Path) -> None:
    """Show summary of exported Neptune data.

    This command shows a summary of available data in the exported parquet files,
    including project counts, run counts, and attribute types.
    """
    # Validate data path exists
    if not data_path.exists():
        raise click.BadParameter(f"Data path does not exist: {data_path}")

    # Create parquet reader and summary manager
    parquet_reader = ParquetReader(base_path=data_path)
    summary_manager = SummaryManager(parquet_reader=parquet_reader)

    try:
        # Show general data summary
        summary_data = summary_manager.get_data_summary()
        ReportFormatter.print_data_summary(summary_data, data_path)

    except Exception as e:
        click.echo(f"Failed to generate summary: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
