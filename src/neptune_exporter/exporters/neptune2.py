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
from neptune.attributes.attribute import Attribute
from neptune import attributes
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
    attributes.String: "string",
    attributes.Float: "float",
    attributes.Integer: "int",
    attributes.Datetime: "datetime",
    attributes.Boolean: "bool",
    attributes.Artifact: "artifact",
    attributes.File: "file",
    attributes.GitRef: "git_ref",
    attributes.NotebookRef: "notebook_ref",
    attributes.RunState: "run_state",
    attributes.FileSet: "file_set",
    attributes.FileSeries: "file_series",
    attributes.FloatSeries: "float_series",
    attributes.StringSeries: "string_series",
    attributes.StringSet: "string_set",
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
        all_data: list[dict[str, Any]] = []

        for run_id in run_ids:
            try:
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
            except neptune.exceptions.MetadataContainerNotFound:
                continue

        if all_data:
            converted_df = self._convert_parameters_to_schema(all_data, project_id)
            yield pa.RecordBatch.from_pandas(converted_df, schema=model.SCHEMA)
        else:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

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
        if not run_ids:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)
            return

        all_data = []

        for run_id in run_ids:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                # Get run structure to find metric attributes
                # structure = run.get_structure()
                namespaces: list[str] = []  # self._flatten_namespaces(structure)

                for namespace in namespaces:
                    try:
                        attr_obj = run[namespace]
                        attr_type = self._get_attribute_type(attr_obj)

                        # Filter by attributes if specified
                        if attributes is not None:
                            if isinstance(attributes, str):
                                if namespace != attributes:
                                    continue
                            else:
                                if namespace not in attributes:
                                    continue

                        # Only process metric types (float_series)
                        if attr_type == "float_series":
                            # Fetch series values
                            series_df = attr_obj.fetch_values()

                            for _, row in series_df.iterrows():
                                all_data.append(
                                    {
                                        "project_id": project_id,
                                        "run_id": run_id,
                                        "attribute_path": namespace,
                                        "attribute_type": "float_series",
                                        "step": Decimal(str(row.get("step", 0))),
                                        "timestamp": row.get("timestamp"),
                                        "int_value": None,
                                        "float_value": row.get("value"),
                                        "string_value": None,
                                        "bool_value": None,
                                        "datetime_value": None,
                                        "string_set_value": None,
                                        "file_value": None,
                                        "histogram_value": None,
                                    }
                                )
                    except Exception:
                        # Skip attributes that can't be fetched
                        continue

        if all_data:
            df = pd.DataFrame(all_data)
            yield pa.RecordBatch.from_pandas(df, schema=model.SCHEMA)
        else:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download series data from Neptune runs."""
        if not run_ids:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)
            return

        all_data = []

        for run_id in run_ids:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                # Get run structure to find series attributes
                # structure = run.get_structure()
                namespaces: list[str] = []  # self._flatten_namespaces(structure)

                for namespace in namespaces:
                    try:
                        attr_obj = run[namespace]
                        attr_type = self._get_attribute_type(attr_obj)

                        # Filter by attributes if specified
                        if attributes is not None:
                            if isinstance(attributes, str):
                                if namespace != attributes:
                                    continue
                            else:
                                if namespace not in attributes:
                                    continue

                        # Only process series types
                        if attr_type in _SERIES_TYPES:
                            # Fetch series values
                            series_df = attr_obj.fetch_values()

                            for _, row in series_df.iterrows():
                                value = row.get("value")

                                # Handle different series types
                                if attr_type == "string_series":
                                    string_value = value
                                    histogram_value = None
                                elif attr_type == "histogram_series":
                                    string_value = None
                                    # Convert histogram to dict format
                                    if hasattr(value, "__dict__"):
                                        histogram_value = dataclasses.asdict(value)
                                    else:
                                        histogram_value = value
                                else:
                                    string_value = None
                                    histogram_value = None

                                all_data.append(
                                    {
                                        "project_id": project_id,
                                        "run_id": run_id,
                                        "attribute_path": namespace,
                                        "attribute_type": attr_type,
                                        "step": Decimal(str(row.get("step", 0))),
                                        "timestamp": row.get("timestamp"),
                                        "int_value": None,
                                        "float_value": None,
                                        "string_value": string_value,
                                        "bool_value": None,
                                        "datetime_value": None,
                                        "string_set_value": None,
                                        "file_value": None,
                                        "histogram_value": histogram_value,
                                    }
                                )
                    except Exception:
                        # Skip attributes that can't be fetched
                        continue

        if all_data:
            df = pd.DataFrame(all_data)
            yield pa.RecordBatch.from_pandas(df, schema=model.SCHEMA)
        else:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

    def download_files(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: None | str | Sequence[str],
        destination: Path,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download files from Neptune runs."""
        if not run_ids:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)
            return

        destination = destination.resolve()
        all_data: list[dict[str, Any]] = []

        for run_id in run_ids:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                # Get run structure to find file attributes
                # structure = run.get_structure()
                namespaces: list[str] = []  # self._flatten_namespaces(structure)

                for namespace in namespaces:
                    try:
                        attr_obj = run[namespace]
                        attr_type = self._get_attribute_type(attr_obj)

                        # Filter by attributes if specified
                        if attributes is not None:
                            if isinstance(attributes, str):
                                if namespace != attributes:
                                    continue
                            else:
                                if namespace not in attributes:
                                    continue

                        # Only process file types
                        if attr_type in _FILE_TYPES:
                            # Download single file
                            file_path = (
                                destination / f"{run_id}_{namespace.replace('/', '_')}"
                            )
                            attr_obj.download(str(file_path))

                            all_data.append(
                                {
                                    "project_id": project_id,
                                    "run_id": run_id,
                                    "attribute_path": namespace,
                                    "attribute_type": "file",
                                    "step": None,
                                    "timestamp": None,
                                    "int_value": None,
                                    "float_value": None,
                                    "string_value": None,
                                    "bool_value": None,
                                    "datetime_value": None,
                                    "string_set_value": None,
                                    "file_value": {
                                        "path": str(file_path.relative_to(destination))
                                    },
                                    "histogram_value": None,
                                }
                            )

                        elif attr_type in _FILE_SERIES_TYPES:
                            # Download file series
                            series_dir = (
                                destination / f"{run_id}_{namespace.replace('/', '_')}"
                            )
                            series_dir.mkdir(parents=True, exist_ok=True)
                            attr_obj.download(str(series_dir))

                            # Get series values to extract step and timestamp info
                            try:
                                series_df = attr_obj.fetch_values()
                                for _, row in series_df.iterrows():
                                    all_data.append(
                                        {
                                            "project_id": project_id,
                                            "run_id": run_id,
                                            "attribute_path": namespace,
                                            "attribute_type": "file_series",
                                            "step": Decimal(str(row.get("step", 0))),
                                            "timestamp": row.get("timestamp"),
                                            "int_value": None,
                                            "float_value": None,
                                            "string_value": None,
                                            "bool_value": None,
                                            "datetime_value": None,
                                            "string_set_value": None,
                                            "file_value": {
                                                "path": str(
                                                    series_dir.relative_to(destination)
                                                )
                                            },
                                            "histogram_value": None,
                                        }
                                    )
                            except Exception:
                                # If we can't get series info, just add one entry
                                all_data.append(
                                    {
                                        "project_id": project_id,
                                        "run_id": run_id,
                                        "attribute_path": namespace,
                                        "attribute_type": "file_series",
                                        "step": None,
                                        "timestamp": None,
                                        "int_value": None,
                                        "float_value": None,
                                        "string_value": None,
                                        "bool_value": None,
                                        "datetime_value": None,
                                        "string_set_value": None,
                                        "file_value": {
                                            "path": str(
                                                series_dir.relative_to(destination)
                                            )
                                        },
                                        "histogram_value": None,
                                    }
                                )
                    except Exception:
                        # Skip attributes that can't be fetched
                        continue

        if all_data:
            df = pd.DataFrame(all_data)
            yield pa.RecordBatch.from_pandas(df, schema=model.SCHEMA)
        else:
            yield pa.RecordBatch.from_pylist([], schema=model.SCHEMA)

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
