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

from typing import NewType, Optional, Sequence
from pathlib import Path
import neptune_query as nq
from neptune_query import runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter


ProjectId = NewType("ProjectId", str)
RunId = NewType("RunId", str)


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
            "Listing projects is not implemented in neptune 3 client, list project manually"
        )

    def list_runs(
        self, project_id: ProjectId, runs: Optional[str] = None
    ) -> list[RunId]:
        return nq_runs.list_runs(project_id, runs)

    def fetch_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: str | Sequence[str],
    ) -> None:
        _df = nq_runs.fetch_runs_table(  # index="run", cols="attribute" (=path)
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_PARAMETER_TYPES),
            sort_by=Attribute(name="sys/id", type="string"),
            sort_direction="asc",
            type_suffix_in_column_names=True,
        )

    def download_metrics(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: str | Sequence[str],
    ) -> None:
        _df = nq_runs.fetch_metrics(  # index=["run", "step"], column="path"
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_METRIC_TYPES),
            include_time=True,
            lineage_to_the_root=False,
            type_suffix_in_column_names=True,
        )

    def download_series(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: str | Sequence[str],
    ) -> None:
        _df = nq_runs.fetch_series(  # index=["run", "step"], column="path"
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_SERIES_TYPES),
            include_time=True,
            lineage_to_the_root=False,
            ype_suffix_in_column_names=True,
        )

    def download_artifacts(
        self,
        project_id: ProjectId,
        run_ids: list[RunId],
        attributes: str | Sequence[str],
        destination: Path,
    ) -> None:
        files_df = nq_runs.fetch_runs_table(  # index="run", cols="attribute" (=path)
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_FILE_TYPES),
            sort_by=Attribute(name="sys/id", type="string"),
            sort_direction="asc",
            type_suffix_in_column_names=True,
        )

        files_series_df = nq_runs.fetch_series(  # index=["run", "step"], column="path"
            project=project_id,
            runs=run_ids,
            attributes=AttributeFilter(name=attributes, type=_FILE_SERIES_TYPES),
            include_time=True,
            lineage_to_the_root=False,
            ype_suffix_in_column_names=True,
        )

        _file_paths_df = nq_runs.download_files(  # index=["run", "step"], column="path"
            files=files_df,
            destination=destination,
        )

        _file_series_paths_df = (
            nq_runs.download_files(  # index=["run", "step"], column="path"
                files=files_series_df,
                destination=destination,
            )
        )
