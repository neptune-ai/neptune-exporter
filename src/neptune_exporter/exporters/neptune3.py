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
import pyarrow as pa
import pandas as pd
from typing import Generator, Optional, Sequence
from pathlib import Path
import neptune_query as nq
from neptune_query import runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter
from concurrent.futures import ThreadPoolExecutor

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
        attribute_batch_size: int = 128,
        max_workers: int = 16,
    ):
        self._initialize_client(api_token=api_token)
        self._attribute_batch_size = attribute_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _initialize_client(self, api_token: Optional[str]) -> None:
        if api_token is not None:
            nq.set_api_token(api_token)

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
                type_suffix_in_column_names=True,
            )

            if not metrics_df.empty:
                converted_df = self._convert_metrics_to_schema(metrics_df, project_id)
                return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
            return None

        # Create batches of attributes
        attribute_batches = [
            attributes_list[i : i + self._attribute_batch_size]
            for i in range(0, len(attributes_list), self._attribute_batch_size)
        ]

        # Submit all batches to the executor
        futures = [
            self._executor.submit(fetch_and_convert_batch, batch_attributes)
            for batch_attributes in attribute_batches
        ]

        # Yield results as they complete
        for future in futures:
            result = future.result()
            if result is not None:
                yield result

    def _convert_metrics_to_schema(
        self, series_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert metrics (which are float series) DataFrame with multiindex to long format matching model.SCHEMA."""
        stacked_df = series_df.stack(
            [0], future_stack=True
        )  # index = [run, step, attribute_path_type], columns = [value_type(absolute_time, value)]
        stacked_df.index.names = ["run", "step", "attribute_path_type"]
        stacked_df = stacked_df.reset_index()

        stacked_df[["attribute_path", "attribute_type"]] = stacked_df[
            "attribute_path_type"
        ].str.rsplit(":", n=1, expand=True)

        return pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": stacked_df["run"],
                "attribute_path": stacked_df["attribute_path"],
                "attribute_type": stacked_df["attribute_type"],
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
        attributes_list = nq_runs.list_attributes(
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_SERIES_TYPES),
        )

        def fetch_and_convert_batch(batch_attributes):
            series_df = nq_runs.fetch_series(
                project=project_id,
                runs=run_ids,
                attributes=batch_attributes,
                include_time="absolute",
                lineage_to_the_root=False,
            )

            if not series_df.empty:
                converted_df = self._convert_series_to_schema(series_df, project_id)
                return pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
            return None

        # Create batches of attributes
        attribute_batches = [
            attributes_list[i : i + self._attribute_batch_size]
            for i in range(0, len(attributes_list), self._attribute_batch_size)
        ]

        # Submit all batches to the executor
        futures = [
            self._executor.submit(fetch_and_convert_batch, batch_attributes)
            for batch_attributes in attribute_batches
        ]

        # Yield results as they complete
        for future in futures:
            result = future.result()
            if result is not None:
                yield result

    def _convert_series_to_schema(
        self, series_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert series DataFrame with multiindex to long format matching model.SCHEMA."""
        stacked_df = series_df.stack(
            [0], future_stack=True
        )  # index = [run, step, attribute_path], columns = [value_type(absolute_time, value)]
        stacked_df.index.names = ["run", "step", "attribute_path"]
        stacked_df = stacked_df.reset_index()

        stacked_df["attribute_type"] = stacked_df["value"].map(
            lambda x: "string_series" if isinstance(x, str) else "histogram_series"
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
        files_df = nq_runs.fetch_runs_table(  # index="run", cols="attribute" (=path)
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_FILE_TYPES),
            sort_by=Attribute(name="sys/id", type="string"),
            sort_direction="asc",
            type_suffix_in_column_names=True,
        )

        files_series_df = nq_runs.fetch_series(  # index=["run", "step"], column lvl1="path" lvl2=["value", "absolute_time"]
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_FILE_SERIES_TYPES),
            include_time="absolute",
            lineage_to_the_root=False,
            type_suffix_in_column_names=True,
        )

        file_paths_df = nq_runs.download_files(  # index=["run", "step"], column="path"
            files=files_df,
            destination=destination,
        )

        file_series_paths_df = (
            nq_runs.download_files(  # index=["run", "step"], column="path"
                files=files_series_df,
                destination=destination,
            )
        )

        yield pa.stack_batches(
            [file_paths_df, file_series_paths_df], schema=model.SCHEMA
        )
