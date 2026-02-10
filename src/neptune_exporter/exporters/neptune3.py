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

import dataclasses
from decimal import Decimal
from neptune_query.exceptions import NeptuneError
import pyarrow as pa
import pandas as pd
from typing import Generator, Literal, Optional, Sequence
from pathlib import Path
import neptune_query as nq
from neptune_query import filters, runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging

from neptune_exporter import model
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.exceptions import (
    raise_if_neptune_api_token_error,
)
from neptune_exporter.progress.listeners import (
    NoopProgressListener,
    ProgressListener,
)
from neptune_exporter.types import ProjectId, SourceRunId


_PARAMETER_TYPES: Sequence[str] = (
    "float",
    "int",
    "string",
    "bool",
    "datetime",
    "string_set",
)
_METRIC_TYPES: Sequence[str] = ("float_series",)
_SERIES_TYPES: Sequence[str] = (
    "string_series",
    "histogram_series",
)
_FILE_TYPES: Sequence[str] = ("file",)
_FILE_SERIES_TYPES: Sequence[str] = ("file_series",)


class Neptune3Exporter(NeptuneExporter):
    def __init__(
        self,
        error_reporter: ErrorReporter,
        api_token: Optional[str] = None,
        series_attribute_batch_size: int = 128,
        file_attribute_batch_size: int = 16,
        file_series_attribute_batch_size: int = 8,
        max_workers: int = 8,
        include_archived_runs: bool = False,
        show_client_logs: bool = False,
        include_metric_previews: bool = False,
    ):
        self._quantize_base = Decimal("1.000000")
        self._include_archived_runs = include_archived_runs
        self._include_metric_previews = include_metric_previews
        self._series_attribute_batch_size = series_attribute_batch_size
        self._file_attribute_batch_size = file_attribute_batch_size
        self._file_series_attribute_batch_size = file_series_attribute_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._logger = logging.getLogger(__name__)
        self._initialize_client(api_token=api_token, show_client_logs=show_client_logs)
        self._error_reporter = error_reporter

    def _initialize_client(
        self, api_token: Optional[str], show_client_logs: bool
    ) -> None:
        if api_token is not None:
            nq.set_api_token(api_token)

        if show_client_logs:
            logging.getLogger("neptune").setLevel(logging.INFO)
        else:
            logging.getLogger("neptune").setLevel(logging.ERROR)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def list_projects(self) -> list[ProjectId]:
        raise NotImplementedError(
            "Listing projects is not implemented in neptune 3 client, list projects manually"
        )

    def list_runs(
        self,
        project_id: ProjectId,
        runs: Optional[str] = None,
        query: Optional[str] = None,
    ) -> list[SourceRunId]:
        """
        List Neptune runs.
        The runs parameter is a regex pattern that the sys/custom_run_id must match.
        """
        if query is not None:
            raise NotImplementedError(
                "Query parameter is not implemented in neptune 3 client"
            )
        if runs is not None:
            runs_filter = filters.Filter.matches(
                filters.Attribute("sys/custom_run_id", type="string"), runs
            )
        else:
            runs_filter = filters.Filter.matches(
                filters.Attribute("sys/custom_run_id", type="string"), ".+"
            )
        if not self._include_archived_runs:
            runs_filter = runs_filter & filters.Filter.ne(
                filters.Attribute("sys/archived", type="bool"), True
            )

        try:
            return nq_runs.list_runs(project=project_id, runs=runs_filter)
        except Exception as e:
            raise_if_neptune_api_token_error(e)
            raise

    def download_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        try:
            parameters_df = (
                nq_runs.fetch_runs_table(  # index="run", cols="attribute" (=path)
                    project=project_id,
                    runs=run_ids,
                    attributes=AttributeFilter(name=attributes, type=_PARAMETER_TYPES),
                    sort_by=Attribute(name="sys/id", type="string"),
                    sort_direction="asc",
                    type_suffix_in_column_names=True,
                )
            )
            if parameters_df.empty:
                yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)
                return

            converted_df = self._convert_parameters_to_schema(parameters_df, project_id)
            yield self._record_batch_from_pandas(converted_df)
        except Exception as e:
            self._handle_runs_exception(
                project_id=project_id,
                run_ids=run_ids,
                batch_type="parameters",
                exception=e,
            )
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def _convert_parameters_to_schema(
        self, parameters_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert wide parameters DataFrame to long format matching model.SCHEMA."""
        parameters_df = parameters_df.reset_index()

        # Melt the DataFrame to convert from wide to long format
        melted_df = parameters_df.melt(
            id_vars=["run"],
            var_name="attribute_path_type",
            value_name="value",
        )

        # Split attribute_path_type into path and type
        melted_df[["attribute_path", "attribute_type"]] = melted_df[
            "attribute_path_type"
        ].str.rsplit(":", n=1, expand=True)

        # Create the schema-compliant DataFrame
        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": melted_df["run"],
                "attribute_path": melted_df["attribute_path"],
                "attribute_type": melted_df["attribute_type"],
                "step": None,
                "timestamp": None,
                "value": melted_df["value"],
                "int_value": None,
                "float_value": None,
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

        # Fill in the appropriate value column based on attribute_type
        # Use vectorized operations for better performance
        for attr_type in result_df["attribute_type"].unique():
            mask = result_df["attribute_type"] == attr_type

            if attr_type == "int":
                result_df.loc[mask, "int_value"] = result_df.loc[mask, "value"]
            elif attr_type == "float":
                result_df.loc[mask, "float_value"] = result_df.loc[mask, "value"]
            elif attr_type == "string":
                result_df.loc[mask, "string_value"] = result_df.loc[mask, "value"]
            elif attr_type == "bool":
                result_df.loc[mask, "bool_value"] = result_df.loc[mask, "value"]
            elif attr_type == "datetime":
                result_df.loc[mask, "datetime_value"] = result_df.loc[mask, "value"]
            elif attr_type == "string_set":
                result_df.loc[mask, "string_set_value"] = result_df.loc[mask, "value"]
            else:
                raise ValueError(f"Unsupported parameter type: {attr_type}")

        result_df = result_df.drop(columns=["value"])

        return result_df

    @staticmethod
    def _record_batch_from_pandas(data: pd.DataFrame) -> pa.RecordBatch:
        table = pa.Table.from_pandas(
            data, schema=model.SCHEMA, preserve_index=False
        ).combine_chunks()
        if table.num_rows == 0:
            return pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in model.SCHEMA],
                schema=model.SCHEMA,
            )
        return table.to_batches()[0]

    def download_metrics(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        listener = progress or NoopProgressListener()
        try:
            attributes_list = nq_runs.list_attributes(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(name=attributes, type=_METRIC_TYPES),
            )
            listener.on_batch_total("metrics", run_ids, len(attributes_list))

            def fetch_and_convert_batch(
                attribute_batch: list[str],
            ) -> Optional[pa.RecordBatch]:
                try:
                    metrics_df = nq_runs.fetch_metrics(
                        project=project_id,
                        runs=run_ids,
                        attributes=attribute_batch,
                        include_time="absolute",
                        include_point_previews=self._include_metric_previews,
                        lineage_to_the_root=False,
                        type_suffix_in_column_names=False,  # assume the type is always "float_series"
                    )

                    if not metrics_df.empty:
                        converted_df = self._convert_metrics_to_schema(
                            metrics_df, project_id
                        )
                        return self._record_batch_from_pandas(converted_df)
                    return None
                except Exception as e:
                    self._handle_batch_exception(
                        project_id, run_ids, attribute_batch, "metrics", e
                    )
                    return None

            # Create batches of attributes
            attribute_batches = [
                attributes_list[i : i + self._series_attribute_batch_size]
                for i in range(
                    0, len(attributes_list), self._series_attribute_batch_size
                )
            ]

            # Submit all batches to the executor
            futures: dict[Future[Optional[pa.RecordBatch]], int] = {
                self._executor.submit(fetch_and_convert_batch, attribute_batch): len(
                    attribute_batch
                )
                for attribute_batch in attribute_batches
            }

            # Yield results as they complete
            for future in as_completed(list(futures)):
                try:
                    result: Optional[pa.RecordBatch] = future.result()
                    if result is not None:
                        yield result
                except Exception as e:
                    self._handle_runs_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        batch_type="metrics",
                        exception=e,
                    )
                finally:
                    listener.on_batch_advance("metrics", run_ids, futures[future])
        except Exception as e:
            self._handle_runs_exception(
                project_id=project_id,
                run_ids=run_ids,
                batch_type="metrics",
                exception=e,
            )
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def _convert_metrics_to_schema(
        self, series_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert metrics (which are float series) DataFrame with multiindex to long format matching model.SCHEMA."""
        stacked_df = series_df.stack(
            [0], future_stack=True
        )  # index = [run, step, attribute_path_type], columns = [value_type(absolute_time, value)]
        stacked_df.index.names = ["run", "step", "attribute_path"]
        stacked_df = stacked_df.reset_index().dropna(
            subset=["absolute_time", "value"], how="all"
        )

        return pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": stacked_df["run"],
                "attribute_path": stacked_df["attribute_path"],
                "attribute_type": "float_series",
                "step": stacked_df["step"].map(
                    lambda x: Decimal(x).quantize(self._quantize_base)
                    if pd.notna(x)
                    else None
                ),
                "timestamp": stacked_df["absolute_time"],
                "int_value": None,
                "float_value": stacked_df["value"],
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        listener = progress or NoopProgressListener()
        try:
            attributes_df = nq_runs.fetch_runs_table(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(name=attributes, type=_SERIES_TYPES),
                sort_by=Attribute(name="sys/id", type="string"),
                sort_direction="asc",
                type_suffix_in_column_names=True,
            )
            attribute_path_types = [
                attr.rsplit(":", maxsplit=1) for attr in attributes_df.columns
            ]
            listener.on_batch_total("series", run_ids, len(attribute_path_types))

            def fetch_and_convert_batch(
                attributes_batch: list[tuple[str, str]],
            ) -> Optional[pa.RecordBatch]:
                try:
                    series_df = nq_runs.fetch_series(
                        project=project_id,
                        runs=run_ids,
                        attributes=[name for name, _ in attributes_batch],
                        include_time="absolute",
                        lineage_to_the_root=False,
                    )

                    if not series_df.empty:
                        converted_df = self._convert_series_to_schema(
                            series_df=series_df,
                            project_id=project_id,
                            attribute_path_types=attributes_batch,
                        )
                        return self._record_batch_from_pandas(converted_df)
                    return None
                except Exception as e:
                    self._handle_batch_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        attribute_batch=[name for name, _ in attributes_batch],
                        batch_type="series",
                        exception=e,
                    )
                    return None

            # Create batches of attributes
            attribute_batches = [
                attribute_path_types[i : i + self._series_attribute_batch_size]
                for i in range(
                    0, len(attribute_path_types), self._series_attribute_batch_size
                )
            ]

            # Submit all batches to the executor
            futures: dict[Future[Optional[pa.RecordBatch]], int] = {
                self._executor.submit(fetch_and_convert_batch, attributes_batch): len(
                    attributes_batch
                )
                for attributes_batch in attribute_batches
            }

            # Yield results as they complete
            for future in as_completed(list(futures)):
                try:
                    result: Optional[pa.RecordBatch] = future.result()
                    if result is not None:
                        yield result
                except Exception as e:
                    self._handle_runs_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        batch_type="series",
                        exception=e,
                    )
                finally:
                    listener.on_batch_advance("series", run_ids, futures[future])
        except Exception as e:
            self._handle_runs_exception(
                project_id=project_id, run_ids=run_ids, batch_type="series", exception=e
            )
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def _convert_series_to_schema(
        self,
        series_df: pd.DataFrame,
        project_id: str,
        attribute_path_types: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """Convert series DataFrame with multiindex to long format matching model.SCHEMA."""
        stacked_df = series_df.stack(
            [0], future_stack=True
        )  # index = [run, step, attribute_path], columns = [value_type(absolute_time, value)]
        stacked_df.index.names = ["run", "step", "attribute_path"]
        stacked_df = stacked_df.reset_index().dropna(
            subset=["absolute_time", "value"], how="all"
        )

        attribute_type_map = {name: typ for name, typ in attribute_path_types}
        stacked_df["attribute_type"] = stacked_df["attribute_path"].map(
            attribute_type_map
        )

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": stacked_df["run"],
                "attribute_path": stacked_df["attribute_path"],
                "attribute_type": stacked_df["attribute_type"],
                "step": stacked_df["step"].map(
                    lambda x: Decimal(x).quantize(self._quantize_base)
                    if pd.notna(x)
                    else None
                ),
                "timestamp": stacked_df["absolute_time"],
                "value": stacked_df["value"],
                "int_value": None,
                "float_value": None,
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

        for attr_type in result_df["attribute_type"].unique():
            mask = result_df["attribute_type"] == attr_type
            if attr_type == "string_series":
                result_df.loc[mask, "string_value"] = result_df.loc[mask, "value"]
            elif attr_type == "histogram_series":
                result_df.loc[mask, "histogram_value"] = result_df.loc[
                    mask, "value"
                ].map(lambda x: dataclasses.asdict(x) if pd.notna(x) else None)
            else:
                raise ValueError(f"Unsupported series type: {attr_type}")

        result_df = result_df.drop(columns=["value"])

        return result_df

    def download_files(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        destination: Path,
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        listener = progress or NoopProgressListener()
        try:
            destination = destination.resolve()

            # Get list of file attributes to batch
            file_attributes = nq_runs.list_attributes(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(name=attributes, type=_FILE_TYPES),
            )

            file_series_attributes = nq_runs.list_attributes(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(name=attributes, type=_FILE_SERIES_TYPES),
            )
            listener.on_batch_total(
                "files", run_ids, len(file_attributes) + len(file_series_attributes)
            )

            def fetch_and_convert_file_batch(
                attribute_batch: list[str],
            ) -> Optional[pa.RecordBatch]:
                try:
                    if not attribute_batch:
                        return None

                    # Fetch files table for this batch
                    files_df = nq_runs.fetch_runs_table(
                        project=project_id,
                        runs=run_ids,
                        attributes=AttributeFilter(
                            name=attribute_batch, type=_FILE_TYPES
                        ),
                        sort_by=Attribute(name="sys/id", type="string"),
                        sort_direction="asc",
                        type_suffix_in_column_names=True,
                    )

                    if files_df.empty:
                        return None

                    # Download files for this batch
                    file_paths_df = nq_runs.download_files(
                        files=files_df,
                        destination=destination,
                    )

                    # Convert to schema
                    converted_df = self._convert_files_to_schema(
                        downloaded_files_df=file_paths_df,
                        project_id=project_id,
                        attribute_type="file",
                        file_series_df=None,
                        destination=destination,
                    )

                    return self._record_batch_from_pandas(converted_df)
                except Exception as e:
                    self._handle_batch_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        attribute_batch=attribute_batch,
                        batch_type="files",
                        exception=e,
                    )
                    return None

            def fetch_and_convert_file_series_batch(
                attribute_batch: list[str],
            ) -> Optional[pa.RecordBatch]:
                try:
                    if not attribute_batch:
                        return None

                    # Fetch file series for this batch
                    files_series_df = nq_runs.fetch_series(
                        project=project_id,
                        runs=run_ids,
                        attributes=AttributeFilter(
                            name=attribute_batch, type=_FILE_SERIES_TYPES
                        ),
                        include_time="absolute",
                        lineage_to_the_root=False,
                    )

                    if files_series_df.empty:
                        return None

                    # Download file series for this batch
                    file_series_paths_df = nq_runs.download_files(
                        files=files_series_df,
                        destination=destination,
                    )

                    # Convert to schema
                    converted_df = self._convert_files_to_schema(
                        downloaded_files_df=file_series_paths_df,
                        project_id=project_id,
                        attribute_type="file_series",
                        file_series_df=files_series_df,
                        destination=destination,
                    )

                    return self._record_batch_from_pandas(converted_df)
                except Exception as e:
                    self._handle_batch_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        attribute_batch=attribute_batch,
                        batch_type="files",
                        exception=e,
                    )
                    return None

            # Create batches of attributes
            file_attribute_batches = [
                file_attributes[i : i + self._file_attribute_batch_size]
                for i in range(0, len(file_attributes), self._file_attribute_batch_size)
            ]

            file_series_attribute_batches = [
                file_series_attributes[i : i + self._file_series_attribute_batch_size]
                for i in range(
                    0,
                    len(file_series_attributes),
                    self._file_series_attribute_batch_size,
                )
            ]

            # Submit all batches to the executor
            futures: dict[Future[Optional[pa.RecordBatch]], int] = {}

            # Submit file batches
            for attribute_batch in file_attribute_batches:
                future = self._executor.submit(
                    fetch_and_convert_file_batch, attribute_batch
                )
                futures[future] = len(attribute_batch)

            # Submit file series batches
            for attribute_batch in file_series_attribute_batches:
                future = self._executor.submit(
                    fetch_and_convert_file_series_batch, attribute_batch
                )
                futures[future] = len(attribute_batch)

            # Yield results as they complete
            for future in as_completed(list(futures)):
                try:
                    result: Optional[pa.RecordBatch] = future.result()
                    if result is not None:
                        yield result
                except Exception as e:
                    self._handle_runs_exception(
                        project_id=project_id,
                        run_ids=run_ids,
                        batch_type="files",
                        exception=e,
                    )
                finally:
                    listener.on_batch_advance("files", run_ids, futures[future])
        except Exception as e:
            self._handle_runs_exception(
                project_id=project_id, run_ids=run_ids, batch_type="files", exception=e
            )
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def _convert_files_to_schema(
        self,
        downloaded_files_df: pd.DataFrame,
        project_id: str,
        attribute_type: Literal["file", "file_series"],
        file_series_df: Optional[pd.DataFrame],
        destination: Path,
    ) -> pd.DataFrame:
        """Convert downloaded files DataFrame to long format matching model.SCHEMA."""
        # Reset index to make 'run' a column
        downloaded_files_df = downloaded_files_df.reset_index()

        # Melt the DataFrame to convert from wide to long format
        melted_df = downloaded_files_df.melt(
            id_vars=["run", "step"],
            var_name="attribute_path",
            value_name="file_path",
        )

        # For file_series, we need to add timestamp information
        if file_series_df is not None:
            # Reset index and melt the file_series_df to get step and timestamp info
            series_stacked_df = file_series_df.stack([0], future_stack=True)
            series_stacked_df.index.names = ["run", "step", "attribute_path"]
            series_stacked_df = series_stacked_df.reset_index().dropna(
                subset=["absolute_time", "value"], how="all"
            )

            # Merge with melted_df to get timestamp
            melted_df = melted_df.merge(
                series_stacked_df[["run", "attribute_path", "step", "absolute_time"]],
                on=["run", "attribute_path", "step"],
                how="left",
            )
        else:
            # For regular files, no timestamp
            melted_df["absolute_time"] = None

        return pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": melted_df["run"],
                "attribute_path": melted_df["attribute_path"],
                "attribute_type": attribute_type,
                "step": melted_df["step"].map(
                    lambda x: Decimal(x).quantize(self._quantize_base)
                    if pd.notna(x)
                    else None
                ),
                "timestamp": melted_df["absolute_time"],
                "int_value": None,
                "float_value": None,
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": melted_df["file_path"].map(
                    lambda x: {"path": str(Path(x).relative_to(destination))}
                    if pd.notna(x)
                    else None
                ),
                "histogram_value": None,
            }
        )

    def _handle_batch_exception(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attribute_batch: list[str],
        batch_type: Literal["parameters", "metrics", "series", "files"],
        exception: Exception,
    ) -> None:
        """Handle exceptions that occur during batch processing."""
        raise_if_neptune_api_token_error(exception)
        if isinstance(exception, NeptuneError):
            # Neptune-related errors
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, attributes {attribute_batch}, batch type {batch_type} because of neptune-query client error.",
                exc_info=True,
            )
        elif isinstance(exception, OSError):
            # Other I/O errors - could be temporary or permanent
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, attributes {attribute_batch}, batch type {batch_type} because of I/O error.",
                exc_info=True,
            )
        else:
            # Unexpected errors - definitely need investigation
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, attributes {attribute_batch}, batch type {batch_type} because of unexpected error.",
                exc_info=True,
            )

        self._error_reporter.record_batch_exception(
            project_id=project_id,
            run_ids=run_ids,
            attribute_paths=attribute_batch,
            exception=exception,
        )

    def _handle_runs_exception(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        batch_type: Literal["parameters", "metrics", "series", "files"],
        exception: Exception,
    ) -> None:
        """Handle exceptions that occur during batch processing."""
        raise_if_neptune_api_token_error(exception)
        if isinstance(exception, NeptuneError):
            # Neptune-related errors
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, batch type {batch_type} because of neptune-query client error.",
                exc_info=True,
            )
        elif isinstance(exception, OSError):
            # Other I/O errors - could be temporary or permanent
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, batch type {batch_type} because of I/O error.",
                exc_info=True,
            )
        else:
            # Unexpected errors - definitely need investigation
            self._logger.warning(
                f"Skipping project {project_id}, run {run_ids}, batch type {batch_type} because of unexpected error.",
                exc_info=True,
            )

        self._error_reporter.record_batch_exception(
            project_id=project_id,
            run_ids=run_ids,
            attribute_paths=None,
            exception=exception,
        )
