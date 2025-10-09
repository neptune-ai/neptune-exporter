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

import click
from pathlib import Path

from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter
from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from neptune_exporter.manager import ExportManager
from neptune_exporter.storage.parquet import ParquetStorage


@click.command()
@click.option(
    "--project-ids",
    "-p",
    multiple=True,
    required=True,
    help="Neptune project IDs to export. Can be specified multiple times.",
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
    help="Filter attributes by name. Can be specified multiple times.",
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
def main(
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
    # Export specific attributes only
    neptune-exporter -p "my-org/my-project" -a "learning_rate" -a "batch_size"

    \b
    # Use Neptune 2.x exporter
    neptune-exporter -p "my-org/my-project" --exporter neptune2
    """
    # Convert tuples to lists and handle None values
    project_ids_list = list(project_ids)
    attributes_list = list(attributes) if attributes else None
    export_classes_list = list(export_classes)

    # Validate export classes
    valid_export_classes = {"parameters", "metrics", "series", "files"}
    export_classes_set = set(export_classes_list)
    if not export_classes_set.issubset(valid_export_classes):
        invalid = export_classes_set - valid_export_classes
        raise click.BadParameter(f"Invalid export classes: {', '.join(invalid)}")

    # Create exporter instance
    if exporter == "neptune2":
        exporter_instance: NeptuneExporter = Neptune2Exporter(api_token=api_token)  # type: ignore
    elif exporter == "neptune3":
        exporter_instance = Neptune3Exporter(api_token=api_token)
    else:
        raise click.BadParameter(f"Unknown exporter: {exporter}")

    # Create storage instance
    storage = ParquetStorage(base_path=output_path)

    # Create and run export manager
    export_manager = ExportManager(
        exporter=exporter_instance,
        storage=storage,
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


if __name__ == "__main__":
    main()
