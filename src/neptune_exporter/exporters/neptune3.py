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
from neptune_query.exceptions import NeptuneError, NeptuneWarning
import pyarrow as pa
import pandas as pd
from typing import Generator, Literal, Optional, Sequence
from pathlib import Path
import neptune_query as nq
from neptune_query import runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from neptune_exporter import model
from neptune_exporter.exporters.exporter import ProjectId, RunId


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


class Neptune3Exporter:
    def __init__(
        self,
        api_token: Optional[str] = None,
        series_attribute_batch_size: int = 128,
        file_attribute_batch_size: int = 16,
        file_series_attribute_batch_size: int = 8,
        max_workers: int = 16,
    ):
        self._initialize_client(api_token=api_token)
        self._series_attribute_batch_size = series_attribute_batch_size
        self._file_attribute_batch_size = file_attribute_batch_size
        self._file_series_attribute_batch_size = file_series_attribute_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._logger = logging.getLogger(__name__)

    def _initialize_client(self, api_token: Optional[str]) -> None:
        if api_token is not None:
            nq.set_api_token(api_token)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def list_projects(self) -> list[ProjectId]:
        raise NotImplementedError(
            "Listing projects is not implemented in neptune 3 client, list projects manually"
        )

    def list_runs(
        self, project_id: ProjectId, runs: Optional[str] = None
    ) -> list[RunId]:
        return nq_runs.list_runs(project=project_id, runs=runs)

    def download_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
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
            yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
        except Exception as e:
            self._handle_batch_exception(f"parameters for project {project_id}", e)
            # Yield empty batch to maintain interface contract
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

    def download_metrics(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        attributes_list = nq_runs.list_attributes(
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_METRIC_TYPES),
        )

        def fetch_and_convert_batch(batch_attributes):
            metrics_df = nq_runs.fetch_metrics(
                project=project_id,
                runs=run_ids,
                attributes=batch_attributes,
                include_time="absolute",
                include_point_previews=False,
                lineage_to_the_root=False,
                type_suffix_in_column_names=False,  # assume the type is always "float_series"
            )

            if not metrics_df.empty:
                converted_df = self._convert_metrics_to_schema(metrics_df, project_id)
                return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
            return None

        # Create batches of attributes
        attribute_batches = [
            attributes_list[i : i + self._series_attribute_batch_size]
            for i in range(0, len(attributes_list), self._series_attribute_batch_size)
        ]

        # Submit all batches to the executor
        futures = [
            self._executor.submit(fetch_and_convert_batch, batch_attributes)
            for batch_attributes in attribute_batches
        ]

        # Yield results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result is not None:
                    yield result
            except Exception as e:
                self._handle_batch_exception(
                    f"metrics batch {i} for project {project_id}", e
                )

    def _convert_metrics_to_schema(
        self, series_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert metrics (which are float series) DataFrame with multiindex to long format matching model.SCHEMA."""
        stacked_df = series_df.stack(
            [0], future_stack=True
        )  # index = [run, step, attribute_path_type], columns = [value_type(absolute_time, value)]
        stacked_df.index.names = ["run", "step", "attribute_path"]
        stacked_df = stacked_df.reset_index()

        return pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": stacked_df["run"],
                "attribute_path": stacked_df["attribute_path"],
                "attribute_type": "float_series",
                "step": stacked_df["step"].map(Decimal),
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
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
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

        def fetch_and_convert_batch(attributes_batch):
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
                return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
            return None

        # Create batches of attributes
        attribute_batches = [
            attribute_path_types[i : i + self._series_attribute_batch_size]
            for i in range(
                0, len(attribute_path_types), self._series_attribute_batch_size
            )
        ]

        # Submit all batches to the executor
        futures = [
            self._executor.submit(fetch_and_convert_batch, attributes_batch)
            for attributes_batch in attribute_batches
        ]

        # Yield results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result is not None:
                    yield result
            except Exception as e:
                self._handle_batch_exception(
                    f"series batch {i} for project {project_id}", e
                )

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
        stacked_df = stacked_df.reset_index()

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
                "step": stacked_df["step"].map(Decimal),
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
                ].map(dataclasses.asdict)
            else:
                raise ValueError(f"Unsupported series type: {attr_type}")

        result_df = result_df.drop(columns=["value"])

        return result_df

    def download_files(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
        destination: Path,
    ) -> Generator[pa.RecordBatch, None, None]:
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

        def fetch_and_convert_file_batch(batch_attributes):
            if not batch_attributes:
                return None

            # Fetch files table for this batch
            files_df = nq_runs.fetch_runs_table(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(name=batch_attributes, type=_FILE_TYPES),
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

            return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

        def fetch_and_convert_file_series_batch(batch_attributes):
            if not batch_attributes:
                return None

            # Fetch file series for this batch
            files_series_df = nq_runs.fetch_series(
                project=project_id,
                runs=run_ids,
                attributes=AttributeFilter(
                    name=batch_attributes, type=_FILE_SERIES_TYPES
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

            return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

        # Create batches of attributes
        file_attribute_batches = [
            file_attributes[i : i + self._file_attribute_batch_size]
            for i in range(0, len(file_attributes), self._file_attribute_batch_size)
        ]

        file_series_attribute_batches = [
            file_series_attributes[i : i + self._file_series_attribute_batch_size]
            for i in range(
                0, len(file_series_attributes), self._file_series_attribute_batch_size
            )
        ]

        # Submit all batches to the executor
        futures = []

        # Submit file batches
        for batch_attributes in file_attribute_batches:
            futures.append(
                self._executor.submit(fetch_and_convert_file_batch, batch_attributes)
            )

        # Submit file series batches
        for batch_attributes in file_series_attribute_batches:
            futures.append(
                self._executor.submit(
                    fetch_and_convert_file_series_batch, batch_attributes
                )
            )

        # Yield results as they complete
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result is not None:
                    yield result
            except Exception as e:
                self._handle_batch_exception(
                    f"files batch {i} for project {project_id}", e
                )

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
            series_stacked_df = series_stacked_df.reset_index()

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
                "step": melted_df["step"].map(Decimal),
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

    def _handle_batch_exception(self, batch_info: str, exception: Exception) -> None:
        """Handle exceptions that occur during batch processing."""
        if isinstance(exception, PermissionError):
            # Permission issues - user needs to fix their setup
            self._logger.error(
                f"Permission denied processing batch {batch_info}: {exception}"
            )
        elif isinstance(exception, FileNotFoundError):
            # File not found - could be user error or system issue
            self._logger.error(
                f"File not found processing batch {batch_info}: {exception}"
            )
        elif (
            isinstance(exception, OSError) and exception.errno == 28
        ):  # No space left on device
            # Critical system issue
            self._logger.critical(
                f"Disk full processing batch {batch_info}: {exception}"
            )
        elif isinstance(exception, (OSError, IOError)):
            # Other I/O errors - could be temporary or permanent
            self._logger.error(f"I/O error processing batch {batch_info}: {exception}")
        elif isinstance(exception, (NeptuneError, NeptuneWarning)):
            # Neptune-related errors
            self._logger.error(
                f"Neptune error processing batch {batch_info}: {exception}"
            )
        else:
            # Unexpected errors - definitely need investigation
            self._logger.error(
                f"Unexpected error processing batch {batch_info}: {exception}",
                exc_info=True,
            )
