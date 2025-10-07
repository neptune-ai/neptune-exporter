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

import pyarrow as pa
import pandas as pd
from typing import Generator, Optional, Sequence
from pathlib import Path
import neptune_query as nq
from neptune_query import runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter

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
    def __init__(self, api_token: Optional[str] = None):
        self._initialize_client(api_token=api_token)

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
        metrics_df = nq_runs.fetch_metrics(  # index=["run", "step"], column lvl1="path" lvl2=["value", "absolute_time"]
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_METRIC_TYPES),
            include_time="absolute",
            include_preview=False,
            lineage_to_the_root=False,
            type_suffix_in_column_names=True,
        )
        converted_df = self._convert_metrics_to_schema(metrics_df, project_id)
        yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

    def _convert_metrics_to_schema(
        self, metrics_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert metrics DataFrame with multiindex to long format matching model.SCHEMA."""
        # Reset index to convert multiindex (run, step) to columns
        metrics_df = metrics_df.reset_index()

        # Get the column levels - should be (path, [value, absolute_time])
        if not isinstance(metrics_df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns for metrics DataFrame")

        # Use vectorized stack() but handle dtypes by converting to object first
        # This avoids expensive merges while maintaining vectorization
        metrics_df_obj = metrics_df.astype(
            object
        )  # Convert to object to avoid dtype conflicts
        stacked_df = metrics_df_obj.stack([0, 1]).reset_index()
        stacked_df.columns = [
            "run",
            "step",
            "attribute_path",
            "time_component",
            "value",
        ]

        # Pivot to separate value and absolute_time columns
        pivoted_df = stacked_df.pivot_table(
            index=["run", "step", "attribute_path"],
            columns="time_component",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Flatten column names
        pivoted_df.columns.name = None

        # Create the schema-compliant DataFrame
        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": pivoted_df["run"],
                "attribute_path": pivoted_df["attribute_path"],
                "attribute_type": "float_series",  # Metrics are always float_series
                "step": pivoted_df["step"],
                "timestamp": pivoted_df["absolute_time"],
                "int_value": None,
                "float_value": pivoted_df["value"],  # Metrics are float values
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

        # PyArrow will handle dtype conversion based on the schema
        return result_df

    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        series_df = nq_runs.fetch_series(  # index=["run", "step"], column lvl1="path" lvl2=["value", "absolute_time"]
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_SERIES_TYPES),
            include_time="absolute",
            lineage_to_the_root=False,
            type_suffix_in_column_names=True,
        )
        converted_df = self._convert_series_to_schema(series_df, project_id)
        yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

    def _convert_series_to_schema(
        self, series_df: pd.DataFrame, project_id: str
    ) -> pd.DataFrame:
        """Convert series DataFrame with multiindex to long format matching model.SCHEMA."""
        # Reset index to convert multiindex (run, step) to columns
        series_df = series_df.reset_index()

        # Get the column levels - should be (path, [value, absolute_time])
        if not isinstance(series_df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns for series DataFrame")

        # Use vectorized stack() but handle dtypes by converting to object first
        # This avoids expensive merges while maintaining vectorization
        series_df_obj = series_df.astype(
            object
        )  # Convert to object to avoid dtype conflicts
        stacked_df = series_df_obj.stack([0, 1]).reset_index()
        stacked_df.columns = [
            "run",
            "step",
            "attribute_path",
            "time_component",
            "value",
        ]

        # Pivot to separate value and absolute_time columns
        pivoted_df = stacked_df.pivot_table(
            index=["run", "step", "attribute_path"],
            columns="time_component",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Flatten column names
        pivoted_df.columns.name = None

        # Extract attribute type from attribute_path (format: "path:type")
        pivoted_df[["attribute_path", "attribute_type"]] = pivoted_df[
            "attribute_path"
        ].str.rsplit(":", n=1, expand=True)

        # Create the schema-compliant DataFrame
        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": pivoted_df["run"],
                "attribute_path": pivoted_df["attribute_path"],
                "attribute_type": pivoted_df["attribute_type"],
                "step": pivoted_df["step"],
                "timestamp": pivoted_df["absolute_time"],
                "value": pivoted_df["value"],
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
        for attr_type in result_df["attribute_type"].unique():
            mask = result_df["attribute_type"] == attr_type

            if attr_type == "string_series":
                result_df.loc[mask, "string_value"] = result_df.loc[mask, "value"]
            elif attr_type == "histogram_series":
                # Convert Histogram objects to dict format for PyArrow struct schema
                histogram_values = result_df.loc[mask, "value"].apply(
                    lambda h: {"type": h.type, "edges": h.edges, "values": h.values}
                )
                result_df.loc[mask, "histogram_value"] = histogram_values
            else:
                raise ValueError(f"Unsupported series type: {attr_type}")

        result_df = result_df.drop(columns=["value"])

        # PyArrow will handle dtype conversion based on the schema
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
