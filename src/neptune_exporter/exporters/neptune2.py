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

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from decimal import Decimal
from typing import cast
from neptune.attributes.attribute import Attribute
from neptune import attributes as na
from neptune.attributes.series.fetchable_series import FetchableSeries
import pandas as pd
import pyarrow as pa
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Sequence
import re
import logging
import json
import queue

import neptune
import neptune.exceptions
from neptune import management

from neptune_exporter import model
from neptune_exporter.exporters.exporter import NeptuneExporter
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.progress.listeners import (
    NoopProgressListener,
    ProgressListener,
)
from neptune_exporter.types import ProjectId, SourceRunId

_ATTRIBUTE_TYPE_MAP = {
    na.String: "string",
    na.Float: "float",
    na.Integer: "int",
    na.Datetime: "datetime",
    na.Boolean: "bool",
    na.Artifact: "artifact",  # save as a file
    na.File: "file",
    na.GitRef: "git_ref",  # ignore, seems not to be implemented
    na.NotebookRef: "notebook_ref",  # ignore, not implemented
    na.RunState: "run_state",  # ignore, just transient metadata
    na.FileSet: "file_set",
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
_FILE_SET_TYPES: Sequence[str] = ("file_set",)
_ARTIFACT_TYPES: Sequence[str] = ("artifact",)


class Neptune2Exporter(NeptuneExporter):
    def __init__(
        self,
        error_reporter: ErrorReporter,
        api_token: Optional[str] = None,
        max_workers: int = 16,
        show_client_logs: bool = False,
        include_trashed_runs: bool = False,
    ):
        self._error_reporter = error_reporter
        self._include_trashed_runs = include_trashed_runs
        self._api_token = api_token
        self._quantize_base = Decimal("1.000000")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
        self._logger = logging.getLogger(__name__)
        self._show_client_logs = show_client_logs
        self._initialize_client(show_client_logs=show_client_logs)

    def _initialize_client(self, show_client_logs: bool) -> None:
        if show_client_logs:
            logging.getLogger("neptune").setLevel(logging.INFO)
        else:
            logging.getLogger("neptune").setLevel(logging.ERROR)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def list_projects(self) -> list[ProjectId]:
        """List Neptune projects."""
        return cast(list[ProjectId], management.get_project_list())

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
        with neptune.init_project(
            api_token=self._api_token, project=project_id, mode="read-only"
        ) as project:
            runs_table = project.fetch_runs_table(
                query=query,
                columns=["sys/id"],
                trashed=None if self._include_trashed_runs else False,
                progress_bar=None if self._show_client_logs else False,
                ascending=True,
            ).to_pandas()
            if not len(runs_table):
                return []

            if runs is not None:
                runs_table = runs_table[runs_table["sys/id"].str.match(runs)]
            return list(runs_table["sys/id"])

    def download_parameters(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download parameters from Neptune runs."""
        future_to_run_id = {
            self._executor.submit(
                self._process_run_parameters, project_id, run_id, attributes
            ): run_id
            for run_id in run_ids
        }

        # Yield results as they complete
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                result = future.result()
                if result is not None:
                    yield result
            except Exception as e:
                self._handle_run_exception(project_id, run_id, e)

    def _process_run_parameters(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attributes: None | str | Sequence[str],
    ) -> Optional[pa.RecordBatch]:
        """Process parameters for a single run."""
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

            attribute_path, attribute_type = None, None
            for attribute in self._iterate_attributes(structure):
                attribute_path, attribute_type = None, None
                try:
                    attribute_path = "/".join(attribute._path)

                    # Filter by attribute path if attributes filter is provided
                    if not self._should_include_attribute(attribute_path, attributes):
                        continue

                    attribute_type = self._get_attribute_type(attribute)
                    if attribute_type not in _PARAMETER_TYPES:
                        continue

                    if attribute_type == "string_set":
                        value = attribute.fetch()
                    else:
                        value = get_value(all_parameter_values, attribute._path)

                    all_data.append(
                        {
                            "run_id": run_id,
                            "attribute_path": attribute_path,
                            "attribute_type": attribute_type,
                            "value": value,
                        }
                    )
                except Exception as e:
                    self._handle_attribute_exception(
                        project_id, run_id, attribute_path, attribute_type, e
                    )

        if all_data:
            converted_df = self._convert_parameters_to_schema(all_data, project_id)
            return self._record_batch_from_pandas(converted_df)
        return None

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
        """Download metrics from Neptune runs."""
        listener = progress or NoopProgressListener()
        for run_id, data in self._run_attribute_workers(
            project_id,
            run_ids,
            attributes,
            _METRIC_TYPES,
            self._metric_worker,
            on_attribute_total=lambda rid, total: listener.on_run_total(
                "metrics", rid, total
            ),
            on_attribute_done=lambda rid: listener.on_run_advance("metrics", rid, 1),
        ):
            if not data:
                continue
            try:
                converted_df = self._convert_metrics_to_schema(data, project_id)
                if converted_df is not None:
                    yield self._record_batch_from_pandas(converted_df)
            except Exception as e:
                self._handle_run_exception(project_id, run_id, e)

    def _convert_metrics_to_schema(
        self, all_data_dfs: list[pd.DataFrame], project_id: ProjectId
    ) -> Optional[pd.DataFrame]:
        all_data_df = pd.concat(all_data_dfs)

        if all_data_df.empty:
            return None

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": all_data_df["step"].map(
                    lambda x: Decimal(x).quantize(self._quantize_base)
                    if pd.notna(x)
                    else None
                ),
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
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download series data from Neptune runs."""
        listener = progress or NoopProgressListener()
        for run_id, data in self._run_attribute_workers(
            project_id,
            run_ids,
            attributes,
            _SERIES_TYPES,
            self._series_worker,
            on_attribute_total=lambda rid, total: listener.on_run_total(
                "series", rid, total
            ),
            on_attribute_done=lambda rid: listener.on_run_advance("series", rid, 1),
        ):
            if not data:
                continue
            try:
                converted_df = self._convert_series_to_schema(data, project_id)
                if converted_df is not None:
                    yield self._record_batch_from_pandas(converted_df)
            except Exception as e:
                self._handle_run_exception(project_id, run_id, e)

    def _convert_series_to_schema(
        self, all_data_dfs: list[pd.DataFrame], project_id: ProjectId
    ) -> Optional[pd.DataFrame]:
        all_data_df = pd.concat(all_data_dfs)

        if all_data_df.empty:
            return None

        result_df = pd.DataFrame(
            {
                "project_id": project_id,
                "run_id": all_data_df["run_id"],
                "attribute_path": all_data_df["attribute_path"],
                "attribute_type": all_data_df["attribute_type"],
                "step": all_data_df["step"].map(
                    lambda x: Decimal(x).quantize(self._quantize_base)
                    if pd.notna(x)
                    else None
                ),
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
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        destination: Path,
        progress: ProgressListener | None = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Download files from Neptune runs."""
        listener = progress or NoopProgressListener()
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)

        def file_worker(
            project_id,
            run_id,
            attribute_queue,
            on_attribute_done,
        ):
            return self._file_worker(
                project_id,
                run_id,
                attribute_queue,
                destination=destination,
                on_attribute_done=on_attribute_done,
            )

        for run_id, data in self._run_attribute_workers(
            project_id,
            run_ids,
            attributes,
            (
                *_FILE_TYPES,
                *_ARTIFACT_TYPES,
                *_FILE_SERIES_TYPES,
                *_FILE_SET_TYPES,
            ),
            file_worker,
            on_attribute_total=lambda rid, total: listener.on_run_total(
                "files", rid, total
            ),
            on_attribute_done=lambda rid: listener.on_run_advance("files", rid, 1),
        ):
            if not data:
                continue
            try:
                converted_df = self._convert_files_to_schema(data, project_id)
                if converted_df is not None:
                    yield self._record_batch_from_pandas(converted_df)
            except Exception as e:
                self._handle_run_exception(project_id, run_id, e)

    def _convert_files_to_schema(
        self, all_data_dfs: list[dict[str, Any]], project_id: ProjectId
    ) -> Optional[pd.DataFrame]:
        all_data_df = pd.DataFrame(all_data_dfs)

        if all_data_df.empty:
            return None

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

    def _run_attribute_workers(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attributes: None | str | Sequence[str],
        allowed_types: Sequence[str],
        worker_fn: Callable[
            [
                ProjectId,
                SourceRunId,
                "queue.Queue[tuple[str, str]]",
                Optional[Callable[[SourceRunId], None]],
            ],
            list[Any],
        ],
        on_attribute_total: Optional[Callable[[SourceRunId, int], None]] = None,
        on_attribute_done: Optional[Callable[[SourceRunId], None]] = None,
    ) -> Generator[tuple[SourceRunId, list[Any]], None, None]:
        if not run_ids:
            return

        run_iter = iter(run_ids)
        listing_parallelism = min(len(run_ids), max(1, self._max_workers // 2))

        listing_futures: dict = {}
        worker_futures: dict = {}
        pending_runs: dict[SourceRunId, tuple["queue.Queue[tuple[str, str]]", int]] = {}
        run_data: dict[SourceRunId, list[Any]] = {}
        pending_counts: dict[SourceRunId, int] = {}

        def submit_listing(run_id: SourceRunId) -> None:
            future = self._executor.submit(
                self._list_run_attributes,
                project_id,
                run_id,
                attributes,
                allowed_types,
            )
            listing_futures[future] = run_id

        for _ in range(listing_parallelism):
            try:
                submit_listing(next(run_iter))
            except StopIteration:
                break

        def select_run_for_worker() -> Optional[SourceRunId]:
            for run_id, (attr_queue, _) in pending_runs.items():
                if pending_counts[run_id] == 0 and attr_queue.qsize() > 0:
                    return run_id
            best_run_id = None
            best_score = -1.0
            for run_id, (attr_queue, _) in pending_runs.items():
                remaining = attr_queue.qsize()
                if remaining <= 0:
                    continue
                assigned = pending_counts[run_id]
                if assigned == 0:
                    continue
                if remaining <= assigned:
                    continue
                score = remaining / assigned
                if score > best_score:
                    best_score = score
                    best_run_id = run_id
            return best_run_id

        while listing_futures or worker_futures or pending_runs:
            while len(worker_futures) < self._max_workers and pending_runs:
                run_id = select_run_for_worker()
                if run_id is None:
                    break
                attribute_queue, _ = pending_runs[run_id]
                if attribute_queue.qsize() <= pending_counts[run_id]:
                    break
                future = self._executor.submit(
                    worker_fn,
                    project_id,
                    run_id,
                    attribute_queue,
                    on_attribute_done,
                )
                worker_futures[future] = run_id
                pending_counts[run_id] += 1

            futures = list(listing_futures.keys()) + list(worker_futures.keys())
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                if future in listing_futures:
                    run_id = listing_futures.pop(future)
                    try:
                        attribute_list = future.result()
                    except Exception as e:
                        self._handle_run_exception(project_id, run_id, e)
                        attribute_list = []
                    if attribute_list:
                        attribute_queue = queue.Queue()
                        for item in attribute_list:
                            attribute_queue.put(item)
                        pending_runs[run_id] = (
                            attribute_queue,
                            len(attribute_list),
                        )
                        run_data[run_id] = []
                        pending_counts[run_id] = 0
                    if on_attribute_total is not None:
                        on_attribute_total(run_id, len(attribute_list))
                    try:
                        next_run_id = next(run_iter)
                    except StopIteration:
                        next_run_id = None
                    if next_run_id is not None:
                        submit_listing(next_run_id)
                else:
                    run_id = worker_futures.pop(future)
                    try:
                        result = future.result()
                        if result:
                            run_data[run_id].extend(result)
                    except Exception as e:
                        self._handle_run_exception(project_id, run_id, e)
                    pending_counts[run_id] -= 1
                    if pending_counts[run_id] == 0:
                        data = run_data.pop(run_id, [])
                        pending_runs.pop(run_id, None)
                        pending_counts.pop(run_id, None)
                        yield run_id, data

    def _metric_worker(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_queue: "queue.Queue[tuple[str, str]]",
        on_attribute_done: Optional[Callable[[SourceRunId], None]] = None,
    ) -> list[pd.DataFrame]:
        data: list[pd.DataFrame] = []
        try:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                while True:
                    try:
                        attribute_path, attribute_type = attribute_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        attribute = run.get_attribute(attribute_path)
                        if attribute is None:
                            continue
                        series_attribute = cast(FetchableSeries, attribute)
                        series_df = series_attribute.fetch_values(
                            progress_bar=None if self._show_client_logs else False
                        )
                        if series_df.empty:
                            continue
                        series_df["run_id"] = run_id
                        series_df["attribute_path"] = attribute_path
                        series_df["attribute_type"] = attribute_type
                        data.append(series_df)
                    except Exception as e:
                        self._handle_attribute_exception(
                            project_id, run_id, attribute_path, attribute_type, e
                        )
                    finally:
                        if on_attribute_done is not None:
                            on_attribute_done(run_id)
        except Exception as e:
            self._handle_run_exception(project_id, run_id, e)
        return data

    def _series_worker(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_queue: "queue.Queue[tuple[str, str]]",
        on_attribute_done: Optional[Callable[[SourceRunId], None]] = None,
    ) -> list[pd.DataFrame]:
        data: list[pd.DataFrame] = []
        try:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                while True:
                    try:
                        attribute_path, attribute_type = attribute_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        attribute = run.get_attribute(attribute_path)
                        if attribute is None:
                            continue
                        series_attribute = cast(FetchableSeries, attribute)
                        series_df = series_attribute.fetch_values(
                            progress_bar=None if self._show_client_logs else False
                        )
                        if series_df.empty:
                            continue
                        series_df["run_id"] = run_id
                        series_df["attribute_path"] = attribute_path
                        series_df["attribute_type"] = attribute_type
                        data.append(series_df)
                    except Exception as e:
                        self._handle_attribute_exception(
                            project_id, run_id, attribute_path, attribute_type, e
                        )
                    finally:
                        if on_attribute_done is not None:
                            on_attribute_done(run_id)
        except Exception as e:
            self._handle_run_exception(project_id, run_id, e)
        return data

    def _file_worker(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_queue: "queue.Queue[tuple[str, str]]",
        *,
        destination: Path,
        on_attribute_done: Optional[Callable[[SourceRunId], None]] = None,
    ) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        try:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                while True:
                    try:
                        attribute_path, attribute_type = attribute_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        attribute = run.get_attribute(attribute_path)
                        if attribute is None:
                            continue

                        if attribute_type in _FILE_TYPES:
                            file_attribute = cast(na.File, attribute)
                            file_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_attribute.download(
                                str(file_path),
                                progress_bar=(
                                    None if self._show_client_logs else False
                                ),
                            )
                            data.append(
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
                        elif attribute_type in _ARTIFACT_TYPES:
                            artifact_attribute = cast(na.Artifact, attribute)
                            artifact_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            artifact_path.mkdir(parents=True, exist_ok=True)
                            artifact_files_list = artifact_attribute.fetch_files_list()
                            files_list_data_path = artifact_path / "files_list.json"
                            serialized_data = [
                                file_data.to_dto() for file_data in artifact_files_list
                            ]
                            with open(files_list_data_path, "w") as opened:
                                json.dump(serialized_data, opened)
                            data.append(
                                {
                                    "run_id": run_id,
                                    "step": None,
                                    "attribute_path": attribute_path,
                                    "attribute_type": attribute_type,
                                    "file_value": {
                                        "path": str(
                                            files_list_data_path.relative_to(
                                                destination
                                            )
                                        )
                                    },
                                }
                            )
                        elif attribute_type in _FILE_SERIES_TYPES:
                            file_series_attribute = cast(na.FileSeries, attribute)
                            file_series_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            file_series_attribute.download(
                                str(file_series_path),
                                progress_bar=(
                                    None if self._show_client_logs else False
                                ),
                            )
                            file_paths = [
                                p for p in file_series_path.iterdir() if p.is_file()
                            ]
                            data.extend(
                                [
                                    {
                                        "run_id": run_id,
                                        "step": Decimal(file_path.stem).quantize(
                                            self._quantize_base
                                        ),
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
                        elif attribute_type in _FILE_SET_TYPES:
                            file_set_attribute = cast(na.FileSet, attribute)
                            file_set_path = (
                                destination / project_id / run_id / attribute_path
                            )
                            file_set_path.mkdir(parents=True, exist_ok=True)
                            file_set_attribute.download(
                                str(file_set_path),
                                progress_bar=(
                                    None if self._show_client_logs else False
                                ),
                            )
                            data.append(
                                {
                                    "run_id": run_id,
                                    "step": None,
                                    "attribute_path": attribute_path,
                                    "attribute_type": attribute_type,
                                    "file_value": {
                                        "path": str(
                                            file_set_path.relative_to(destination)
                                        )
                                    },
                                }
                            )
                    except Exception as e:
                        self._handle_attribute_exception(
                            project_id, run_id, attribute_path, attribute_type, e
                        )
                    finally:
                        if on_attribute_done is not None:
                            on_attribute_done(run_id)
        except Exception as e:
            self._handle_run_exception(project_id, run_id, e)
        return data

    def _list_run_attributes(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attributes: None | str | Sequence[str],
        allowed_types: Sequence[str],
    ) -> list[tuple[str, str]]:
        try:
            with neptune.init_run(
                api_token=self._api_token,
                project=project_id,
                with_id=run_id,
                mode="read-only",
            ) as run:
                structure = run.get_structure()
        except Exception as e:
            self._handle_run_exception(project_id, run_id, e)
            return []

        attribute_list: list[tuple[str, str]] = []
        for attribute in self._iterate_attributes(structure):
            attribute_path = "/".join(attribute._path)
            if not self._should_include_attribute(attribute_path, attributes):
                continue
            attribute_type = self._get_attribute_type(attribute)
            if attribute_type not in allowed_types:
                continue
            attribute_list.append((attribute_path, attribute_type))
        return attribute_list

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

    def _should_include_attribute(
        self, attribute_path: str, attributes: None | str | Sequence[str]
    ) -> bool:
        """Check if an attribute should be included based on the attributes filter."""
        if attributes is None:
            return True

        if isinstance(attributes, str):
            # Treat as regex pattern
            return bool(re.search(attributes, attribute_path))
        elif isinstance(attributes, Sequence):
            # Treat as exact attribute names to match
            return attribute_path in attributes

        return True

    def _handle_run_exception(
        self, project_id: ProjectId, run_id: SourceRunId, exception: Exception
    ) -> None:
        """Handle exceptions that occur during run processing."""
        if isinstance(exception, neptune.exceptions.NeptuneException):
            # Other Neptune-specific errors
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id} because of neptune client error.",
                exc_info=True,
            )
        elif isinstance(exception, OSError):
            # Other I/O errors - could be temporary or permanent
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id} because of I/O error.",
                exc_info=True,
            )
        else:
            # Unexpected errors - definitely need investigation
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id} because of unexpected error.",
                exc_info=True,
            )
        self._error_reporter.record_exception(
            project_id=project_id,
            run_id=run_id,
            attribute_path=None,
            attribute_type=None,
            exception=exception,
        )

    def _handle_attribute_exception(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_path: Optional[str],
        attribute_type: Optional[str],
        exception: Exception,
    ) -> None:
        """Handle exceptions that occur during attribute processing."""
        if attribute_path is None:
            self._handle_run_exception(project_id, run_id, exception)
            return

        # Silently ignore only FetchAttributeNotFoundException for sys/group_tags (no log, no error jsonl)
        if attribute_path == "sys/group_tags" and isinstance(
            exception, neptune.exceptions.FetchAttributeNotFoundException
        ):
            return

        if isinstance(exception, neptune.exceptions.NeptuneException):
            # Other Neptune-specific errors
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id}, attribute {attribute_path} ({attribute_type}) because of neptune client error.",
                exc_info=True,
            )
        elif isinstance(exception, OSError):
            # Other I/O errors - could be temporary or permanent
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id}, attribute {attribute_path} ({attribute_type}) because of I/O error.",
                exc_info=True,
            )
        else:
            # Unexpected errors - definitely need investigation
            self._logger.warning(
                f"Skipping project {project_id}, run {run_id}, attribute {attribute_path} ({attribute_type}) because of unexpected error.",
                exc_info=True,
            )
        self._error_reporter.record_exception(
            project_id=project_id,
            run_id=run_id,
            attribute_path=attribute_path,
            attribute_type=attribute_type,
            exception=exception,
        )
