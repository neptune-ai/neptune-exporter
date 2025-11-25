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

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Optional

from neptune_exporter.types import ProjectId, SourceRunId


@dataclass
class ErrorSummary:
    exception_count: int


class ErrorReporter:
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._summary = ErrorSummary(exception_count=0)

    def get_summary(self) -> ErrorSummary:
        with self._lock:
            return ErrorSummary(exception_count=self._summary.exception_count)

    def record_exception(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_path: Optional[str],
        attribute_type: Optional[str],
        exception: Exception,
    ) -> None:
        with self._lock:
            self._summary.exception_count += 1
            self._write_record(
                project_id=project_id,
                run_id=run_id,
                attribute_path=attribute_path,
                attribute_type=attribute_type,
                exception=exception,
            )

    def record_batch_exception(
        self,
        project_id: ProjectId,
        run_ids: list[SourceRunId],
        attribute_paths: Optional[list[str]],
        exception: Exception,
    ) -> None:
        with self._lock:
            for run_id in run_ids:
                if attribute_paths is None:
                    self._summary.exception_count += 1
                    self._write_record(
                        project_id=project_id,
                        run_id=run_id,
                        attribute_path=None,
                        attribute_type=None,
                        exception=exception,
                    )
                else:
                    for attribute_path in attribute_paths:
                        self._summary.exception_count += 1
                        self._write_record(
                            project_id=project_id,
                            run_id=run_id,
                            attribute_path=attribute_path,
                            attribute_type=None,
                            exception=exception,
                        )

    def _write_record(
        self,
        project_id: ProjectId,
        run_id: SourceRunId,
        attribute_path: Optional[str],
        attribute_type: Optional[str],
        exception: Exception,
    ) -> None:
        with open(self.path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "project_id": project_id,
                        "run_id": run_id,
                        "attribute_path": attribute_path,
                        "attribute_type": attribute_type,
                        "exception": exception.__class__.__name__,
                    }
                )
                + "\n"
            )
