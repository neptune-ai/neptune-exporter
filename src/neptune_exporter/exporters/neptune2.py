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

from decimal import Decimal
from neptune.attributes.attribute import Attribute
from neptune import attributes as na
from neptune.attributes.series.fetchable_series import FetchableSeries
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

import neptune
import neptune.exceptions
from neptune import management

from neptune_exporter import model
from neptune_exporter.exporters.exporter import ProjectId, RunId

_ATTRIBUTE_TYPE_MAP = {
    na.String: "string",
    na.Float: "float",
    na.Integer: "int",
    na.Datetime: "datetime",
    na.Boolean: "bool",
    na.Artifact: "artifact",  #
    na.File: "file",
    na.GitRef: "git_ref",  #
    na.NotebookRef: "notebook_ref",  #
    na.RunState: "run_state",  #
    na.FileSet: "file_set",  #
    na.FileSeries: "file_series",
    na.FloatSeries: "float_series",
    na.StringSeries: "string_series",
    na.StringSet: "string_set",
}

_PARAMETER_TYPES: Sequence[str] = (
    "float",
    "int",
    "string",
    "bool",
    "datetime",
    "string_set",
)
_METRIC_TYPES: Sequence[str] = ("float_series",)
_SERIES_TYPES: Sequence[str] = ("string_series",)
_FILE_TYPES: Sequence[str] = ("file",)
_FILE_SERIES_TYPES: Sequence[str] = ("file_series",)


class Neptune2Exporter:
    def __init__(self, api_token: Optional[str] = None):
        self._api_token = api_token

    def list_projects(self) -> list[ProjectId]:
        """List Neptune projects."""
        return management.get_project_list()

    def list_runs(
        self, project_id: ProjectId, runs: Optional[str] = None
    ) -> list[RunId]:
        """List Neptune runs."""
        with neptune.init_project(
            api_token=self._api_token, project=project_id, mode="read-only"
        ) as project:
            runs_table = project.fetch_runs_table(columns=[]).to_pandas()
            if len(runs_table):
                return list(runs_table["sys/id"])
            return []

    def download_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download parameters from Neptune runs."""
        for run_id in run_ids:
            try:
                all_data: list[dict[str, Any]] = []

                with neptune.init_run(
                    api_token=self._api_token,
                    project=project_id,
                    with_id=run_id,
                    mode="read-only",
                ) as run:
                    structure = run.get_structure()
                    all_parameter_values = run.fetch()

                    def get_value(values: dict[str, Any], path: list[str]) -> Any:
                        try:
                            for part in path:
                                values = values[part]
                            return values
                        except KeyError:
                            return None

                    for attribute in self._iterate_attributes(structure):
                        attribute_path = "/".join(attribute._path)
                        # TODO: filter by attribute._path

                        attribute_type = self._get_attribute_type(attribute)
                        if attribute_type not in _PARAMETER_TYPES:
                            continue

                        value = get_value(all_parameter_values, attribute._path)

                        all_data.append(
                            {
                                "run_id": run_id,
                                "attribute_path": attribute_path,
                                "attribute_type": attribute_type,
                                "value": value,
                            }
                        )
                if all_data:
                    converted_df = self._convert_parameters_to_schema(
                        all_data, project_id
                    )
                    yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

            except neptune.exceptions.MetadataContainerNotFound:
                pass

    def _convert_parameters_to_schema(
        self, all_data: list[dict[str, Any]], project_id: ProjectId
    ) -> pd.DataFrame:
        all_data_df = pd.DataFrame(all_data)

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": None,
                "timestamp": None,
                "value": all_data_df["value"],
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
        """Download metrics from Neptune runs."""
        for run_id in run_ids:
            try:
                all_data_dfs: list[pd.DataFrame] = []

                with neptune.init_run(
                    api_token=self._api_token,
                    project=project_id,
                    with_id=run_id,
                    mode="read-only",
                ) as run:
                    structure = run.get_structure()

                    for attribute in self._iterate_attributes(structure):
                        attribute_path = "/".join(attribute._path)
                        # TODO: filter by attribute._path

                        attribute_type = self._get_attribute_type(attribute)
                        if attribute_type not in _METRIC_TYPES:
                            continue

                        series_attribute: FetchableSeries = attribute
                        series_df = series_attribute.fetch_values()

                        series_df["run_id"] = run_id
                        series_df["attribute_path"] = attribute_path
                        series_df["attribute_type"] = attribute_type

                        all_data_dfs.append(series_df)

                if all_data_dfs:
                    converted_df = self._convert_metrics_to_schema(
                        all_data_dfs, project_id
                    )
                    yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

            except neptune.exceptions.MetadataContainerNotFound:
                pass

    def _convert_metrics_to_schema(
        self, all_data_dfs: list[pd.DataFrame], project_id: ProjectId
    ) -> pd.DataFrame:
        all_data_df = pd.concat(all_data_dfs)

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": all_data_df["step"].map(Decimal),
                "timestamp": all_data_df["timestamp"],
                "int_value": None,
                "float_value": all_data_df["value"],
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

        return result_df

    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download series data from Neptune runs."""
        for run_id in run_ids:
            try:
                all_data_dfs: list[pd.DataFrame] = []

                with neptune.init_run(
                    api_token=self._api_token,
                    project=project_id,
                    with_id=run_id,
                    mode="read-only",
                ) as run:
                    structure = run.get_structure()

                    for attribute in self._iterate_attributes(structure):
                        attribute_path = "/".join(attribute._path)
                        # TODO: filter by attribute._path

                        attribute_type = self._get_attribute_type(attribute)
                        if attribute_type not in _SERIES_TYPES:
                            continue

                        series_attribute: FetchableSeries = attribute
                        series_df = series_attribute.fetch_values()

                        series_df["run_id"] = run_id
                        series_df["attribute_path"] = attribute_path
                        series_df["attribute_type"] = attribute_type

                        all_data_dfs.append(series_df)

                if all_data_dfs:
                    converted_df = self._convert_series_to_schema(
                        all_data_dfs, project_id
                    )
                    yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

            except neptune.exceptions.MetadataContainerNotFound:
                pass

    def _convert_series_to_schema(
        self, all_data_dfs: list[pd.DataFrame], project_id: ProjectId
    ) -> pd.DataFrame:
        all_data_df = pd.concat(all_data_dfs)

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": all_data_df["step"].map(Decimal),
                "timestamp": all_data_df["timestamp"],
                "int_value": None,
                "float_value": None,
                "string_value": all_data_df["value"],
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

        return result_df

    def download_files(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
        destination: Path,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download files from Neptune runs."""
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)

        for run_id in run_ids:
            try:
                all_data_dfs: list[dict[str, Any]] = []

                with neptune.init_run(
                    api_token=self._api_token,
                    project=project_id,
                    with_id=run_id,
                    mode="read-only",
                ) as run:
                    structure = run.get_structure()

                    for attribute in self._iterate_attributes(structure):
                        attribute_path = "/".join(attribute._path)
                        # TODO: filter by attribute._path

                        attribute_type = self._get_attribute_type(attribute)
                        if attribute_type in _FILE_TYPES:
                            file_attribute: na.File = attribute

                            file_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_attribute.download(str(file_path))

                            all_data_dfs.append(
                                {
                                    "run_id": run_id,
                                    "step": None,
                                    "attribute_path": attribute_path,
                                    "attribute_type": attribute_type,
                                    "file_value": {
                                        "path": str(file_path.relative_to(destination))
                                    },
                                }
                            )
                        elif attribute_type in _FILE_SERIES_TYPES:
                            file_series_attribute: na.FileSeries = attribute

                            file_series_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            file_series_attribute.download(str(file_series_path))
                            file_paths = [
                                p for p in file_series_path.iterdir() if p.is_file()
                            ]

                            all_data_dfs.extend(
                                [
                                    {
                                        "run_id": run_id,
                                        "step": Decimal(file_path.stem),
                                        "attribute_path": attribute_path,
                                        "attribute_type": attribute_type,
                                        "file_value": {
                                            "path": str(
                                                file_path.relative_to(destination)
                                            )
                                        },
                                    }
                                    for file_path in file_paths
                                ]
                            )

                if all_data_dfs:
                    converted_df = self._convert_files_to_schema(
                        all_data_dfs, project_id
                    )
                    yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)

            except neptune.exceptions.MetadataContainerNotFound:
                pass

    def _convert_files_to_schema(
        self, all_data_dfs: list[dict[str, Any]], project_id: ProjectId
    ) -> pd.DataFrame:
        all_data_df = pd.DataFrame(all_data_dfs)
        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": all_data_df["step"],
                "timestamp": None,
                "int_value": None,
                "float_value": None,
                "string_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": all_data_df["file_value"],
                "histogram_value": None,
            }
        )
        return result_df

    def _iterate_attributes(
        self, structure: dict[str, Any]
    ) -> Generator[Attribute, None, None]:
        """Flatten nested namespace dictionary into list of paths."""
        for value in structure.values():
            if isinstance(value, dict):
                yield from self._iterate_attributes(value)
            elif isinstance(value, Attribute):
                yield value

    def _get_attribute_type(self, attribute: Attribute) -> str:
        attribute_class = type(attribute)
        return _ATTRIBUTE_TYPE_MAP.get(attribute_class, "unknown")
