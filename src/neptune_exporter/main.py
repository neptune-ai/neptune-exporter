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

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from neptune_exporter.export_manager import ExportManager
from neptune_exporter.model_registry_export_manager import (
    ModelRegistryExportManager,
)
from neptune_exporter.model_registry_summary_manager import (
    ModelRegistrySummaryManager,
)
from neptune_exporter.project_selection import resolve_export_project_ids
from neptune_exporter.progress.listeners import (
    Neptune2ProgressListenerFactory,
    Neptune3ProgressListenerFactory,
    NoopProgressListenerFactory,
    ProgressListenerFactory,
)
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.exceptions import NeptuneExporterAuthError
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter
from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from neptune_exporter.loader_manager import LoaderManager
from neptune_exporter.loaders.loader import DataLoader
from neptune_exporter.logging_utils import create_console_handler, info_always
from neptune_exporter.storage.parquet_reader import ParquetReader
from neptune_exporter.storage.parquet_writer import ParquetWriter
from neptune_exporter.summary_manager import SummaryManager
from neptune_exporter.types import ProjectId, SourceRunId
from neptune_exporter.validation import ReportFormatter
from neptune_exporter.workspace_projects import list_workspace_projects


@click.group()
def cli():
    """Neptune Exporter - Export and migrate Neptune experiment data."""
    pass


def _log_selected_project_ids(
    logger: logging.Logger, project_ids: list[str], max_to_show: int = 10
) -> None:
    info_always(logger, f"  Project IDs resolved: {len(project_ids)}")
    shown_ids = project_ids[:max_to_show]
    for project_id in shown_ids:
        info_always(logger, f"    - {project_id}")

    remaining = len(project_ids) - len(shown_ids)
    if remaining > 0:
        info_always(logger, f"    ... and {remaining} more")


@cli.command()
@click.option(
    "--project-ids",
    "-p",
    multiple=True,
    help="Neptune project IDs to export (explicit mode). Can be specified multiple times. If not provided, reads from NEPTUNE_PROJECT environment variable.",
)
@click.option(
    "--workspace",
    help="Workspace name for discovery mode. Mutually exclusive with --project-ids/-p.",
)
@click.option(
    "--project-pattern",
    multiple=True,
    help="Include filter regex on discovered project names. Requires --workspace. Can be specified multiple times.",
)
@click.option(
    "--project-exclude-pattern",
    multiple=True,
    help="Exclude filter regex on discovered project names. Requires --workspace. Can be specified multiple times.",
)
@click.option(
    "--runs",
    "-r",
    help="Filter runs by pattern (e.g., 'RUN-*' or specific run ID).",
)
@click.option(
    "--runs-query",
    "-q",
    help="Filter runs by a custom nql query. Only supported by Neptune 2 exporter. See https://docs-legacy.neptune.ai/usage/nql for more details.",
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
    "--include-archived-runs",
    is_flag=True,
    help="Include archived or trashed runs in export.",
)
@click.option(
    "--include-metric-previews",
    is_flag=True,
    help="Include metric previews in export. Only supported by Neptune 3 exporter. See https://docs.neptune.ai/metric_previews",
)
@click.option(
    "--exporter",
    type=click.Choice(["neptune2", "neptune3"], case_sensitive=False),
    help="Neptune exporter to use.",
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
    "--api-token",
    help="Neptune API token. If not provided, will use environment variable NEPTUNE_API_TOKEN.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging including Neptune internal logs.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter.log",
    help="Path for logging file. A timestamp suffix (YYYYMMDD_HHMMSS) will be automatically added to the filename. Default: ./neptune_exporter.log",
)
@click.option(
    "--error-report-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter_errors.jsonl",
    help="Path for error report file. Default: ./neptune_exporter_errors.jsonl",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar.",
)
def export(
    project_ids: tuple[str, ...],
    workspace: str | None,
    project_pattern: tuple[str, ...],
    project_exclude_pattern: tuple[str, ...],
    runs: str | None,
    runs_query: str | None,
    attributes: tuple[str, ...],
    classes: tuple[str, ...],
    exclude: tuple[str, ...],
    include_archived_runs: bool,
    include_metric_previews: bool,
    exporter: str,
    data_path: Path,
    files_path: Path,
    api_token: str | None,
    verbose: bool,
    log_file: Path,
    error_report_file: Path,
    no_progress: bool,
) -> None:
    """Export Neptune experiment data to parquet files.

    This tool exports data from Neptune projects including parameters, metrics,
    series data, and files to parquet format for further analysis.

    The log file specified with --log-file will have a timestamp suffix
    automatically added (e.g., neptune_exporter_20250115_143022.log) to ensure
    unique log files for each export run.

    Examples:

    \b
    # Export all data from a project
    neptune-exporter export --exporter neptune3 -p "my-org/my-project"

    \b
    # Export only parameters and metrics from specific runs
    neptune-exporter export --exporter neptune3 -p "my-org/my-project" -r "RUN-.*" -c parameters -c metrics

    \b
    # Export everything except files
    neptune-exporter export --exporter neptune3 -p "my-org/my-project" --exclude files

    \b
    # Export specific attributes only (exact match)
    neptune-exporter export --exporter neptune3 -p "my-org/my-project" -a "learning_rate" -a "batch_size"

    \b
    # Export attributes matching a pattern (regex)
    neptune-exporter export --exporter neptune3 -p "my-org/my-project" -a "config/.*"

    \b
    # Use Neptune 2.x exporter
    neptune-exporter export -p "my-org/my-project" --exporter neptune2

    \b
    # Use environment variable for project ID
    NEPTUNE_PROJECT="my-org/my-project" neptune-exporter export --exporter neptune3
    """
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

    # Validate export classes
    valid_export_classes = {"parameters", "metrics", "series", "files"}
    export_classes_set = set(export_classes_list)
    if not export_classes_set.issubset(valid_export_classes):
        invalid = export_classes_set - valid_export_classes
        raise click.BadParameter(f"Invalid export classes: {', '.join(invalid)}")

    # Configure logging
    log_file_path = configure_logging(
        stderr_level=logging.INFO if verbose else logging.ERROR,
        log_file=log_file if log_file else None,
    )

    logger = logging.getLogger(__name__)

    try:
        project_ids_list = resolve_export_project_ids(
            project_ids=project_ids,
            workspace=workspace,
            project_patterns=project_pattern,
            project_exclude_patterns=project_exclude_pattern,
            env_project=os.getenv("NEPTUNE_PROJECT"),
            discover_workspace_projects=lambda workspace_name: list_workspace_projects(
                workspace=workspace_name,
                api_token=api_token,
            ),
        )
    except NeptuneExporterAuthError as e:
        logger.error("%s", e)
        raise click.Abort()

    selection_mode = "workspace discovery" if workspace else "explicit"
    info_always(logger, f"Exporting from {exporter} exporter using arguments:")
    info_always(logger, f"  Project selection mode: {selection_mode}")
    if workspace:
        info_always(logger, f"  Workspace: {workspace}")
        info_always(
            logger,
            f"  Project include patterns: {', '.join(project_pattern) if project_pattern else 'none'}",
        )
        info_always(
            logger,
            f"  Project exclude patterns: {', '.join(project_exclude_pattern) if project_exclude_pattern else 'none'}",
        )
    _log_selected_project_ids(logger, project_ids_list)
    info_always(logger, f"  Runs: {runs}")
    info_always(logger, f"  Runs query: {runs_query}")
    info_always(
        logger,
        f"  Attributes: {', '.join(attributes_list) if isinstance(attributes_list, list) else attributes_list}",
    )
    info_always(logger, f"  Export classes: {', '.join(export_classes_list)}")
    info_always(logger, f"  Exclude: {', '.join(exclude_set)}")
    info_always(logger, f"  Include archived runs: {include_archived_runs}")

    # Create error reporter instance
    error_reporter = ErrorReporter(path=error_report_file)

    # Create exporter instance
    progress_listener_factory: ProgressListenerFactory
    if exporter == "neptune2":
        exporter_instance: NeptuneExporter = Neptune2Exporter(
            error_reporter=error_reporter,
            api_token=api_token,
            include_trashed_runs=include_archived_runs,
        )
        progress_listener_factory = Neptune2ProgressListenerFactory()
    elif exporter == "neptune3":
        exporter_instance = Neptune3Exporter(
            error_reporter=error_reporter,
            api_token=api_token,
            include_archived_runs=include_archived_runs,
            include_metric_previews=include_metric_previews,
        )
        progress_listener_factory = Neptune3ProgressListenerFactory()
    else:
        raise click.BadParameter(f"Unknown exporter: {exporter}")

    if no_progress:
        progress_listener_factory = NoopProgressListenerFactory()

    # Create storage and reader instances
    writer = ParquetWriter(base_path=data_path)
    reader = ParquetReader(base_path=data_path)

    # Create export manager
    export_manager = ExportManager(
        exporter=exporter_instance,
        reader=reader,
        writer=writer,
        error_reporter=error_reporter,
        files_destination=files_path,
        progress_listener_factory=progress_listener_factory,
    )

    info_always(logger, f"Starting export of {len(project_ids_list)} project(s)...")
    info_always(logger, f"Export classes: {', '.join(export_classes_list)}")
    info_always(logger, f"Data path: {data_path.absolute()}")
    info_always(logger, f"Files path: {files_path.absolute()}")

    runs_matched: int | None = None
    export_result = None
    export_failed = False
    try:
        export_result = export_manager.run(
            project_ids=[ProjectId(project_id) for project_id in project_ids_list],
            runs=runs,
            runs_query=runs_query,
            attributes=attributes_list,
            export_classes=export_classes_set,  # type: ignore
        )
        runs_matched = export_result.total_runs

        if export_result.total_runs == 0:
            info_always(logger, "No runs found matching the specified criteria.")
            if runs:
                info_always(logger, f"   Filter: {runs}")
            if runs_query:
                info_always(logger, f"   Runs query: {runs_query}")
            info_always(
                logger,
                "   Try adjusting your run filter or check if the project contains any runs.",
            )
    except NeptuneExporterAuthError as e:
        export_failed = True
        logger.error("%s", e)
        raise click.Abort()
    except Exception:
        export_failed = True
        logger.error("Export failed", exc_info=True)
        raise click.Abort()
    finally:
        exporter_instance.close()
        writer.close_all()
        exception_summary = error_reporter.get_summary()
        skipped_runs = export_result.skipped_runs if export_result is not None else 0
        summary_lines = ["Export summary:"]
        if export_failed:
            summary_lines.append("  Status: failed.")
        else:
            summary_lines.append("  Status: finished.")
        if runs_matched is not None:
            summary_lines.append(f"  Runs matched: {runs_matched}.")
        if skipped_runs > 0:
            summary_lines.append(f"  Runs skipped: {skipped_runs}.")
        if exception_summary.exception_count > 0:
            summary_lines.append(
                f"  Errors recorded: {exception_summary.exception_count}."
            )
            summary_lines.append(f"  Error report: {error_report_file.absolute()}.")
        if log_file_path:
            summary_lines.append(f"  Log file: {log_file_path.absolute()}.")
        for line in summary_lines:
            info_always(logger, line)


@cli.command(name="export-models")
@click.option(
    "--project-ids",
    "-p",
    multiple=True,
    help="Neptune project IDs to export models from (explicit mode). Can be specified multiple times. If not provided, reads from NEPTUNE_PROJECT environment variable.",
)
@click.option(
    "--workspace",
    help="Workspace name for discovery mode. Mutually exclusive with --project-ids/-p.",
)
@click.option(
    "--project-pattern",
    multiple=True,
    help="Include filter regex on discovered project names. Requires --workspace. Can be specified multiple times.",
)
@click.option(
    "--project-exclude-pattern",
    multiple=True,
    help="Exclude filter regex on discovered project names. Requires --workspace. Can be specified multiple times.",
)
@click.option(
    "--models-query",
    "-q",
    help="Filter models by a custom nql query. Applies to model selection in Neptune 2 exporter.",
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
    "--include-archived-models",
    is_flag=True,
    help="Include archived/trashed models in export.",
)
@click.option(
    "--data-path",
    "-d",
    type=click.Path(path_type=Path),
    default="./exports/model_data",
    help="Path for exported model parquet data. Default: ./exports/model_data",
)
@click.option(
    "--files-path",
    "-f",
    type=click.Path(path_type=Path),
    default="./exports/model_files",
    help="Path for downloaded model files. Default: ./exports/model_files",
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
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter.log",
    help="Path for logging file. A timestamp suffix (YYYYMMDD_HHMMSS) will be automatically added to the filename. Default: ./neptune_exporter.log",
)
@click.option(
    "--error-report-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter_errors.jsonl",
    help="Path for error report file. Default: ./neptune_exporter_errors.jsonl",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar.",
)
def export_models(
    project_ids: tuple[str, ...],
    workspace: str | None,
    project_pattern: tuple[str, ...],
    project_exclude_pattern: tuple[str, ...],
    models_query: str | None,
    attributes: tuple[str, ...],
    classes: tuple[str, ...],
    exclude: tuple[str, ...],
    include_archived_models: bool,
    data_path: Path,
    files_path: Path,
    api_token: str | None,
    verbose: bool,
    log_file: Path,
    error_report_file: Path,
    no_progress: bool,
) -> None:
    """Export Neptune2 model registry data to parquet files."""
    if not attributes:
        attributes_list: list[str] | str | None = None
    elif len(attributes) == 1:
        attributes_list = attributes[0]
    else:
        attributes_list = list(attributes)

    all_classes = {"parameters", "metrics", "series", "files"}
    classes_set = set(classes) if classes else set()
    exclude_set = set(exclude) if exclude else set()

    if classes_set and exclude_set:
        raise click.BadParameter(
            "Cannot specify both --classes and --exclude. Use --classes to include specific types or --exclude to exclude specific types."
        )

    if classes_set:
        export_classes_list = list(classes_set)
    elif exclude_set:
        export_classes_list = list(all_classes - exclude_set)
    else:
        export_classes_list = list(all_classes)

    valid_export_classes = {"parameters", "metrics", "series", "files"}
    export_classes_set = set(export_classes_list)
    if not export_classes_set.issubset(valid_export_classes):
        invalid = export_classes_set - valid_export_classes
        raise click.BadParameter(f"Invalid export classes: {', '.join(invalid)}")

    log_file_path = configure_logging(
        stderr_level=logging.INFO if verbose else logging.ERROR,
        log_file=log_file if log_file else None,
    )

    logger = logging.getLogger(__name__)

    try:
        project_ids_list = resolve_export_project_ids(
            project_ids=project_ids,
            workspace=workspace,
            project_patterns=project_pattern,
            project_exclude_patterns=project_exclude_pattern,
            env_project=os.getenv("NEPTUNE_PROJECT"),
            discover_workspace_projects=lambda workspace_name: list_workspace_projects(
                workspace=workspace_name,
                api_token=api_token,
            ),
        )
    except NeptuneExporterAuthError as e:
        logger.error("%s", e)
        raise click.Abort()

    selection_mode = "workspace discovery" if workspace else "explicit"
    info_always(logger, "Exporting model registry data using arguments:")
    info_always(logger, f"  Project selection mode: {selection_mode}")
    if workspace:
        info_always(logger, f"  Workspace: {workspace}")
        info_always(
            logger,
            f"  Project include patterns: {', '.join(project_pattern) if project_pattern else 'none'}",
        )
        info_always(
            logger,
            f"  Project exclude patterns: {', '.join(project_exclude_pattern) if project_exclude_pattern else 'none'}",
        )
    _log_selected_project_ids(logger, project_ids_list)
    info_always(logger, f"  Models query: {models_query}")
    info_always(
        logger,
        f"  Attributes: {', '.join(attributes_list) if isinstance(attributes_list, list) else attributes_list}",
    )
    info_always(logger, f"  Export classes: {', '.join(export_classes_list)}")
    info_always(logger, f"  Exclude: {', '.join(exclude_set)}")
    info_always(logger, f"  Include archived models: {include_archived_models}")

    error_reporter = ErrorReporter(path=error_report_file)
    exporter_instance = Neptune2Exporter(
        error_reporter=error_reporter,
        api_token=api_token,
    )
    progress_listener_factory: ProgressListenerFactory = (
        Neptune2ProgressListenerFactory()
    )

    if no_progress:
        progress_listener_factory = NoopProgressListenerFactory()

    writer = ParquetWriter(base_path=data_path)
    reader = ParquetReader(base_path=data_path)

    export_manager = ModelRegistryExportManager(
        exporter=exporter_instance,
        reader=reader,
        writer=writer,
        error_reporter=error_reporter,
        files_destination=files_path,
        progress_listener_factory=progress_listener_factory,
    )

    info_always(
        logger,
        f"Starting model registry export of {len(project_ids_list)} project(s)...",
    )
    info_always(logger, f"Export classes: {', '.join(export_classes_list)}")
    info_always(logger, f"Data path: {data_path.absolute()}")
    info_always(logger, f"Files path: {files_path.absolute()}")

    model_export_result = None
    export_failed = False
    try:
        model_export_result = export_manager.run(
            project_ids=[ProjectId(project_id) for project_id in project_ids_list],
            models_query=models_query,
            attributes=attributes_list,
            include_archived_models=include_archived_models,
            export_classes=export_classes_set,  # type: ignore[arg-type]
        )
        if (
            model_export_result.total_models == 0
            and model_export_result.total_model_versions == 0
        ):
            info_always(logger, "No models found matching the specified criteria.")
            if models_query:
                info_always(logger, f"   Models query: {models_query}")
            info_always(
                logger,
                "   Try adjusting your model query or check if the project contains any models.",
            )
    except NeptuneExporterAuthError as e:
        export_failed = True
        logger.error("%s", e)
        raise click.Abort()
    except Exception:
        export_failed = True
        logger.error("Model registry export failed", exc_info=True)
        raise click.Abort()
    finally:
        exporter_instance.close()
        writer.close_all()
        exception_summary = error_reporter.get_summary()
        summary_lines = ["Model registry export summary:"]
        if export_failed:
            summary_lines.append("  Status: failed.")
        else:
            summary_lines.append("  Status: finished.")
        if model_export_result is not None:
            summary_lines.append(
                f"  Models matched: {model_export_result.total_models}."
            )
            summary_lines.append(
                f"  Model versions matched: {model_export_result.total_model_versions}."
            )
            if model_export_result.skipped_models > 0:
                summary_lines.append(
                    f"  Models skipped: {model_export_result.skipped_models}."
                )
            if model_export_result.skipped_model_versions > 0:
                summary_lines.append(
                    "  Model versions skipped: "
                    f"{model_export_result.skipped_model_versions}."
                )
        if exception_summary.exception_count > 0:
            summary_lines.append(
                f"  Errors recorded: {exception_summary.exception_count}."
            )
            summary_lines.append(f"  Error report: {error_report_file.absolute()}.")
        if log_file_path:
            summary_lines.append(f"  Log file: {log_file_path.absolute()}.")
        for line in summary_lines:
            info_always(logger, line)


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
    "--step-multiplier",
    type=int,
    help="Step multiplier for converting decimal steps to integers. Default: 1.",
    default=1,
)
@click.option(
    "--loader",
    type=click.Choice(
        [
            "mlflow",
            "wandb",
            "litlogger",
            "zenml",
            "comet",
            "minfx",
            "pluto",
            "goodseed",
        ],
        case_sensitive=False,
    ),
    help="Target platform loader to use.",
)
@click.option(
    "--mlflow-tracking-uri",
    help="MLflow tracking URI. Only used with --loader mlflow.",
)
@click.option(
    "--wandb-entity",
    help="W&B entity (organization/username). Only used with --loader wandb.",
)
@click.option(
    "--wandb-api-key",
    help="W&B API key for authentication. Only used with --loader wandb.",
)
@click.option(
    "--comet-workspace",
    help="Comet workspace. Only used with --loader comet.",
)
@click.option(
    "--comet-api-key",
    help="Comet API key for authentication. Only used with --loader comet.",
)
@click.option(
    "--litlogger-owner",
    help="Lightning.ai user or organization name. Only used with --loader litlogger.",
)
@click.option(
    "--litlogger-api-key",
    help="Lightning.ai API key for authentication. Only used with --loader litlogger.",
)
@click.option(
    "--litlogger-user-id",
    help="Lightning.ai user ID for authentication. Only used with --loader litlogger.",
)
@click.option(
    "--minfx-project",
    help="Minfx project. Only used with --loader minfx.",
)
@click.option(
    "--minfx-api-token",
    help="Minfx API token. Only used with --loader minfx.",
)
@click.option(
    "--pluto-api-key",
    help="Pluto API key for authentication. Only used with --loader pluto.",
)
@click.option(
    "--goodseed-home",
    help="GoodSeed data directory. Only used with --loader goodseed. Default: ~/.goodseed.",
)
@click.option(
    "--goodseed-project",
    help="Override GoodSeed project name for all imported runs. Only used with --loader goodseed. If not set, uses the Neptune project ID.",
)
@click.option(
    "--name-prefix",
    help="Optional prefix for experiment/project and run names.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to the console.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter.log",
    help="Path for logging file. A timestamp suffix (YYYYMMDD_HHMMSS) will be automatically added to the filename. Default: ./neptune_exporter.log",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar.",
)
def load(
    data_path: Path,
    files_path: Path,
    project_ids: tuple[str, ...],
    runs: tuple[str, ...],
    step_multiplier: int,
    loader: str,
    mlflow_tracking_uri: str | None,
    wandb_entity: str | None,
    wandb_api_key: str | None,
    litlogger_owner: str | None,
    litlogger_api_key: str | None,
    litlogger_user_id: str | None,
    minfx_project: str | None,
    minfx_api_token: str | None,
    pluto_api_key: str | None,
    goodseed_home: str | None,
    goodseed_project: str | None,
    name_prefix: str | None,
    verbose: bool,
    log_file: Path,
    no_progress: bool,
    comet_workspace: str | None,
    comet_api_key: str | None,
) -> None:
    """Load exported Neptune data from parquet files to target platforms.

    This tool loads previously exported Neptune data from parquet files
    and uploads it to MLflow, Weights & Biases, Comet, LitLogger, or Minfx
    for further analysis and tracking.

    The log file specified with --log-file will have a timestamp suffix
    automatically added (e.g., neptune_exporter_20250115_143022.log) to ensure
    unique log files for each load run.

    Examples:

    \b
    # Load all data from exported parquet files to MLflow
    neptune-exporter load --loader mlflow

    \b
    # Load to Weights & Biases
    neptune-exporter load --loader wandb --wandb-entity my-org

    \b
    # Load to Comet
    neptune-exporter load --loader comet --comet-workspace "my-workspace"

    \b
    # Load specific projects
    neptune-exporter load -p "my-org/my-project1" -p "my-org/my-project2"

    \b
    # Load specific runs
    neptune-exporter load -r "RUN-123" -r "RUN-456"

    \b
    # Load to specific MLflow tracking URI
    neptune-exporter load --mlflow-tracking-uri "http://localhost:5000"

    \b
    # Load to W&B with API key
    neptune-exporter load --loader wandb --wandb-entity my-org --wandb-api-key xxx

    \b
    # Load to Comet with API key
    neptune-exporter load --loader comet --comet-workspace "my-workspace" --comet-api-key "MY-COMET-API-KEY"

    \b
    # Load to LitLogger (Lightning.ai)
    neptune-exporter load --loader litlogger --litlogger-owner my-user

    \b
    # Load to LitLogger (Lightning.ai) with ID/Key authentication
    neptune-exporter load --loader litlogger --litlogger-owner my-org --litlogger-api-key xxx --litlogger-user-id YYY

    \b
    # Load to LitLogger (Lightning.ai) with prior login
    lightning login && neptune-exporter load --loader litlogger

    \b
    # Load to Minfx
    neptune-exporter load --loader minfx --minfx-project "target-org/target-project" --minfx-api-token xxx

    \b
    # Load to GoodSeed (local)
    neptune-exporter load --loader goodseed

    \b
    # Load to GoodSeed with custom data directory and project
    neptune-exporter load --loader goodseed --goodseed-home /path/to/.goodseed --goodseed-project my-project
    """
    # Convert tuples to lists and handle None values
    project_ids_list = list(project_ids) if project_ids else None
    runs_list = list(runs) if runs else None

    # Validate data path exists
    if not data_path.exists():
        raise click.BadParameter(f"Data path does not exist: {data_path}")

    # Create parquet reader
    parquet_reader = ParquetReader(base_path=data_path)

    # Configure logging
    log_file_path = configure_logging(
        stderr_level=logging.INFO if verbose else logging.ERROR,
        log_file=log_file if log_file else None,
    )

    logger = logging.getLogger(__name__)

    info_always(
        logger,
        f"Starting {loader} loading from {data_path.absolute()} using arguments:",
    )
    info_always(
        logger,
        f"  Project IDs: {', '.join(project_ids_list) if project_ids_list else 'all'}",
    )
    info_always(logger, f"  Runs: {', '.join(runs_list) if runs_list else 'all'}")
    info_always(logger, f"  Step multiplier: {step_multiplier}")
    info_always(logger, f"  Files directory: {files_path.absolute()}")
    if mlflow_tracking_uri:
        info_always(logger, f"  MLflow tracking URI: {mlflow_tracking_uri}")
    if wandb_entity:
        info_always(logger, f"  W&B entity: {wandb_entity}")
    if comet_workspace:
        info_always(logger, f"  Comet workspace: {comet_workspace}")
    if litlogger_owner:
        info_always(logger, f"  LitLogger owner: {litlogger_owner}")
    if name_prefix:
        info_always(logger, f"  Name prefix: {name_prefix}")

    # Create appropriate loader based on --loader flag
    data_loader: DataLoader
    if loader == "mlflow":
        from neptune_exporter.loaders.mlflow_loader import MLflowLoader
        from neptune_exporter.loaders import MLFLOW_AVAILABLE

        if not MLFLOW_AVAILABLE:
            raise click.BadParameter(
                "MLflow loader selected but mlflow is not installed. "
                "Install with `uv sync --extra mlflow`."
            )

        if not mlflow_tracking_uri:
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not mlflow_tracking_uri:
                raise click.BadParameter(
                    "MLflow tracking URI is required when using --loader mlflow. You can set it as an environment variable MLFLOW_TRACKING_URI or provide it with --mlflow-tracking-uri."
                )
        data_loader = MLflowLoader(
            tracking_uri=mlflow_tracking_uri,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "MLflow"
    elif loader == "wandb":
        from neptune_exporter.loaders.wandb_loader import WandBLoader
        from neptune_exporter.loaders import WANDB_AVAILABLE

        if not WANDB_AVAILABLE:
            raise click.BadParameter(
                "W&B loader selected but wandb is not installed. "
                "Install with `uv sync --extra wandb`."
            )

        if not wandb_entity:
            wandb_entity = os.getenv("WANDB_ENTITY")
            if not wandb_entity:
                raise click.BadParameter(
                    "W&B entity is required when using --loader wandb. You can set it as an environment variable WANDB_ENTITY or provide it with --wandb-entity."
                )
        if not wandb_api_key:
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if not wandb_api_key:
                raise click.BadParameter(
                    "W&B API key is required when using --loader wandb. You can set it as an environment variable WANDB_API_KEY or provide it with --wandb-api-key."
                )
        data_loader = WandBLoader(
            entity=wandb_entity,
            api_key=wandb_api_key,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "W&B"
    elif loader == "zenml":
        from neptune_exporter.loaders.zenml_loader import (
            ZenMLLoader,
            ZENML_AVAILABLE,
        )

        if not ZENML_AVAILABLE:
            raise click.BadParameter(
                "ZenML loader selected but zenml is not installed. "
                "Install with `uv sync --extra zenml` and "
                "ensure you are logged into a ZenML server (e.g., via `zenml login`)."
            )

        data_loader = ZenMLLoader(
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "ZenML"
    elif loader == "comet":
        from neptune_exporter.loaders.comet_loader import CometLoader
        from neptune_exporter.loaders import COMET_AVAILABLE

        if not COMET_AVAILABLE:
            raise click.BadParameter(
                "Comet loader selected but comet-ml is not installed. "
                "Install with `uv sync --extra cometml`."
            )

        import comet_ml

        if not comet_workspace:
            comet_workspace = comet_ml.config.get_config("comet.workspace")
            if not comet_workspace:
                raise click.BadParameter(
                    "Comet workspace is required when using --loader comet. You can set it as an environment variable COMET_WORKSPACE, provide it with --comet-workspace, or in a ~/.comet.config file."
                )
        if not comet_api_key:
            comet_api_key = comet_ml.config.get_config("comet.api_key")
            if not comet_api_key:
                raise click.BadParameter(
                    "Comet API key is required when using --loader comet. You can set it as an environment variable COMET_API_KEY, provide it with --comet-api-key, or in a ~/.comet.config file"
                )

        data_loader = CometLoader(
            workspace=comet_workspace,
            api_key=comet_api_key,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "Comet"
    elif loader == "litlogger":
        from neptune_exporter.loaders.litlogger_loader import LitLoggerLoader
        from neptune_exporter.loaders import LITLOGGER_AVAILABLE

        if not LITLOGGER_AVAILABLE:
            raise click.BadParameter(
                "LitLogger loader selected but litlogger is not installed. "
                "Install with `uv sync --extra litlogger`."
            )

        # Teamspace is optional for litlogger - uses default if not provided
        if not litlogger_owner:
            litlogger_owner = os.getenv("LITLOGGER_OWNER")
        # API key is optional - uses default authentication if not provided
        if not litlogger_api_key:
            litlogger_api_key = os.getenv("LITLOGGER_API_KEY")
        # User ID is optional
        if not litlogger_user_id:
            litlogger_user_id = os.getenv("LITLOGGER_USER_ID")
        data_loader = LitLoggerLoader(
            owner=litlogger_owner,
            api_key=litlogger_api_key,
            user_id=litlogger_user_id,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "LitLogger"
    elif loader == "pluto":
        from neptune_exporter.loaders import PlutoLoader, PLUTO_AVAILABLE

        if not PLUTO_AVAILABLE:
            raise click.BadParameter(
                "Pluto loader selected but pluto SDK is not available. "
                "Install with `pip install pluto-ml` or `pluto-ml-nightly` and try again."
            )

        # Resolve Pluto API key: CLI option -> environment variable
        if not pluto_api_key:
            pluto_api_key = os.getenv("PLUTO_API_KEY")

        data_loader = PlutoLoader(
            api_key=pluto_api_key,
            host=None,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "Pluto"
    elif loader == "minfx":
        from neptune_exporter.loaders.minfx_loader import (
            MinfxLoader,
        )
        from neptune_exporter.loaders import MINFX_AVAILABLE

        if not MINFX_AVAILABLE:
            raise click.BadParameter(
                "minfx loader selected but minfx is not installed. "
                "Install with `uv sync --extra minfx`."
            )

        if not minfx_project:
            minfx_project = os.getenv("MINFX_PROJECT")
            if not minfx_project:
                raise click.BadParameter(
                    "Minfx project is required when using --loader minfx. "
                    "You can set it as an environment variable MINFX_PROJECT or "
                    "provide it with --minfx-project."
                )
        if not minfx_api_token:
            minfx_api_token = os.getenv("MINFX_API_TOKEN")
            # API token is optional - minfx client can use other auth methods

        data_loader = MinfxLoader(
            project=minfx_project,
            api_token=minfx_api_token,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "Minfx"
        info_always(logger, f"  Minfx project: {minfx_project}")
    elif loader == "goodseed":
        from neptune_exporter.loaders.goodseed_loader import GoodseedLoader
        from neptune_exporter.loaders import GOODSEED_AVAILABLE

        if not GOODSEED_AVAILABLE:
            raise click.BadParameter(
                "GoodSeed loader selected but goodseed is not installed. "
                "Install with `uv sync --extra goodseed`."
            )

        if not goodseed_home:
            goodseed_home = os.getenv("GOODSEED_HOME")
        if not goodseed_project:
            goodseed_project = os.getenv("GOODSEED_PROJECT")

        data_loader = GoodseedLoader(
            goodseed_home=goodseed_home,
            goodseed_project=goodseed_project,
            name_prefix=name_prefix,
            show_client_logs=verbose,
        )
        loader_name = "GoodSeed"
        if goodseed_home:
            info_always(logger, f"  GoodSeed home: {goodseed_home}")
        if goodseed_project:
            info_always(logger, f"  GoodSeed project: {goodseed_project}")
    else:
        raise click.BadParameter(f"Unknown loader: {loader}")

    # Create loader manager
    loader_manager = LoaderManager(
        parquet_reader=parquet_reader,
        data_loader=data_loader,
        files_directory=files_path,
        step_multiplier=step_multiplier,
        progress_bar=not no_progress,
    )

    load_failed = False
    try:
        loader_manager.load(
            project_ids=(
                [ProjectId(project_id) for project_id in project_ids_list]
                if project_ids_list
                else None
            ),
            runs=[SourceRunId(run_id) for run_id in runs_list] if runs_list else None,
        )
    except Exception:
        load_failed = True
        logger.error(f"{loader_name} loading failed", exc_info=True)
        raise click.Abort()
    finally:
        summary_lines = [f"{loader_name} load summary:"]
        if load_failed:
            summary_lines.append("  Status: failed.")
        else:
            summary_lines.append("  Status: finished.")
        if log_file_path:
            summary_lines.append(f"  Log file: {log_file_path.absolute()}.")
        for line in summary_lines:
            info_always(logger, line)


@cli.command()
@click.option(
    "--data-path",
    "-d",
    type=click.Path(path_type=Path),
    default="./exports/data",
    help="Path for exported parquet data. Default: ./exports/data",
)
@click.option(
    "--model-data-path",
    type=click.Path(path_type=Path),
    default="./exports/model_data",
    help="Path for exported model registry parquet data. Default: ./exports/model_data",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging to the console.",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default="./neptune_exporter.log",
    help="Path for logging file. A timestamp suffix (YYYYMMDD_HHMMSS) will be automatically added to the filename. Default: ./neptune_exporter.log",
)
def summary(
    data_path: Path,
    model_data_path: Path,
    verbose: bool,
    log_file: Path,
) -> None:
    """Show summary of exported Neptune data.

    This command shows a summary of available data in the exported parquet files,
    including project counts, run counts, and attribute types.

    The log file specified with --log-file will have a timestamp suffix
    automatically added (e.g., neptune_exporter_20250115_143022.log) to ensure
    unique log files for each summary run.
    """
    # Configure logging
    configure_logging(
        stderr_level=logging.INFO if verbose else logging.ERROR,
        log_file=log_file if log_file else None,
    )

    logger = logging.getLogger(__name__)

    parquet_reader = ParquetReader(base_path=data_path)
    summary_manager = SummaryManager(parquet_reader=parquet_reader)
    model_parquet_reader = ParquetReader(base_path=model_data_path)
    model_summary_manager = ModelRegistrySummaryManager(
        parquet_reader=model_parquet_reader
    )

    try:
        # Show general data summary
        summary_data = summary_manager.get_data_summary()
        ReportFormatter.print_data_summary(summary_data, data_path)
        click.echo("")
        model_summary_data = model_summary_manager.get_data_summary()
        ReportFormatter.print_model_registry_summary(
            model_summary_data, model_data_path
        )

    except Exception:
        logger.error("Failed to generate summary", exc_info=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


def configure_logging(
    stderr_level: Optional[int],
    log_file: Optional[Path],
) -> Optional[Path]:
    """Configure logging with optional file handler.

    If a log_file path is provided, a timestamp suffix is automatically added
    to the filename (before the extension) to ensure unique log files for each run.
    For example, './neptune_exporter.log' becomes './neptune_exporter_20250115_143022.log'.

    Args:
        stderr_level: Logging level for stderr stream handler (None to disable).
        log_file: Path for log file. Timestamp suffix will be added automatically.

    Returns:
        Path to the log file with timestamp applied, or None if file logging is
        disabled.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    package_logger = logging.getLogger("neptune_exporter")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = True
    package_logger.handlers.clear()
    FORMAT = "%(asctime)s %(name)s:%(levelname)s: %(message)s"

    # Reset handlers for deterministic CLI output.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if stderr_level:
        stream_handler = create_console_handler(stderr_level)
        stream_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root_logger.addHandler(stream_handler)

    log_file_with_timestamp: Optional[Path] = None
    if log_file:
        # Add timestamp suffix to log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = Path(log_file)
        if log_file_path.suffix:
            # Has extension: insert timestamp before extension
            log_file_with_timestamp = log_file_path.with_stem(
                f"{log_file_path.stem}_{timestamp}"
            )
        else:
            # No extension: append timestamp
            log_file_with_timestamp = (
                log_file_path.parent / f"{log_file_path.name}_{timestamp}"
            )

        file_handler = logging.FileHandler(log_file_with_timestamp, mode="w")
        file_handler.setFormatter(logging.Formatter(FORMAT))
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

    return log_file_with_timestamp
