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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence
import threading

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from neptune_exporter.logging_utils import get_rich_console
from neptune_exporter.types import ProjectId, SourceRunId

ProgressKind = Literal["parameters", "metrics", "series", "files"]


class ProgressListener:
    def on_project_total(self, project_id: ProjectId, total: int) -> None:
        return

    def on_project_advance(self, project_id: ProjectId, advance: int) -> None:
        return

    def on_run_started(self, run_id: SourceRunId) -> None:
        return

    def on_run_finished(self, run_id: SourceRunId) -> None:
        return

    def on_run_total(self, kind: ProgressKind, run_id: SourceRunId, total: int) -> None:
        return

    def on_run_advance(
        self, kind: ProgressKind, run_id: SourceRunId, advance: int = 1
    ) -> None:
        return

    def on_batch_total(
        self, kind: ProgressKind, run_ids: Sequence[SourceRunId], total: int
    ) -> None:
        return

    def on_batch_advance(
        self, kind: ProgressKind, run_ids: Sequence[SourceRunId], advance: int
    ) -> None:
        return


class NoopProgressListener(ProgressListener):
    pass


class ProgressListenerFactory(ABC):
    @abstractmethod
    def create_live(self) -> Live:
        pass

    @abstractmethod
    def create_listener(self, live: Live) -> ProgressListener:
        pass


class ProjectProgress:
    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._tasks: dict[ProjectId, TaskID] = {}

    def set_total(self, project_id: ProjectId, total: int) -> None:
        description = f"Project {project_id}"
        task_id = self._tasks.get(project_id)
        if task_id is None:
            task_id = self._progress.add_task(description, total=total)
            self._tasks[project_id] = task_id
            return
        self._progress.update(task_id, total=total, description=description)

    def advance(self, project_id: ProjectId, advance: int) -> None:
        task_id = self._tasks.get(project_id)
        if task_id is None:
            return
        self._progress.update(task_id, advance=advance)


class RunAttributeProgress:
    def __init__(self, progress: Progress, kind: ProgressKind) -> None:
        self._progress = progress
        self._kind = kind
        self._tasks: dict[SourceRunId, TaskID] = {}
        self._totals: dict[SourceRunId, int] = {}
        self._completed: dict[SourceRunId, int] = {}
        self._lock = threading.Lock()

    def set_total(self, run_id: SourceRunId, total: int) -> None:
        if total <= 0:
            return
        with self._lock:
            task_id = self._tasks.get(run_id)
            if task_id is None:
                task_id = self._progress.add_task(f"{run_id} {self._kind}", total=total)
                self._tasks[run_id] = task_id
                self._completed[run_id] = 0
            else:
                self._progress.update(
                    task_id, total=total, description=f"{run_id} {self._kind}"
                )
            self._totals[run_id] = total

    def advance(self, run_id: SourceRunId, advance: int = 1) -> None:
        with self._lock:
            task_id = self._tasks.get(run_id)
            if task_id is None:
                return
            self._progress.update(task_id, advance=advance)
            self._completed[run_id] = self._completed.get(run_id, 0) + advance
            total = self._totals.get(run_id)
            if total is not None and self._completed[run_id] >= total:
                self._progress.remove_task(task_id)
                self._tasks.pop(run_id, None)
                self._totals.pop(run_id, None)
                self._completed.pop(run_id, None)

    def remove(self, run_id: SourceRunId) -> None:
        with self._lock:
            task_id = self._tasks.pop(run_id, None)
            if task_id is None:
                return
            self._progress.remove_task(task_id)
            self._totals.pop(run_id, None)
            self._completed.pop(run_id, None)


class RunListProgress:
    def __init__(self, progress: Progress, max_lines: int = 16) -> None:
        self._progress = progress
        self._max_lines = max_lines
        self._tasks: dict[SourceRunId, TaskID] = {}
        self._hidden: set[SourceRunId] = set()
        self._more_task_id: TaskID | None = None

    def add_run(self, run_id: SourceRunId) -> None:
        if run_id in self._tasks or run_id in self._hidden:
            return
        if len(self._tasks) < self._max_lines:
            task_id = self._progress.add_task(str(run_id), total=1, completed=0)
            self._tasks[run_id] = task_id
            return
        self._hidden.add(run_id)
        self._update_more()

    def remove_run(self, run_id: SourceRunId) -> None:
        task_id = self._tasks.pop(run_id, None)
        if task_id is not None:
            self._progress.remove_task(task_id)
        if run_id in self._hidden:
            self._hidden.remove(run_id)
        self._update_more()

    def _update_more(self) -> None:
        hidden_count = len(self._hidden)
        if hidden_count == 0:
            if self._more_task_id is not None:
                self._progress.remove_task(self._more_task_id)
                self._more_task_id = None
            return
        description = f"... +{hidden_count} more"
        if self._more_task_id is None:
            self._more_task_id = self._progress.add_task(
                description, total=1, completed=1
            )
        else:
            self._progress.update(self._more_task_id, description=description)


class AggregateAttributeProgress:
    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task_id: TaskID | None = None
        self._kind: ProgressKind | None = None
        self._total: int | None = None
        self._completed = 0

    def _format_description(self, kind: ProgressKind) -> str:
        return str(kind)

    def set_total(self, kind: ProgressKind, total: int) -> None:
        if total <= 0:
            if self._task_id is not None:
                self._progress.remove_task(self._task_id)
            self._task_id = None
            self._kind = None
            self._total = None
            self._completed = 0
            return
        if self._task_id is None or self._kind != kind:
            if self._task_id is not None:
                self._progress.remove_task(self._task_id)
            self._kind = kind
            self._total = total
            self._completed = 0
            self._task_id = self._progress.add_task(
                self._format_description(kind), total=total
            )
            return
        self._total = total
        self._completed = 0
        self._progress.update(
            self._task_id,
            total=total,
            completed=0,
            description=self._format_description(kind),
        )

    def advance(self, kind: ProgressKind, advance: int) -> None:
        if self._task_id is None or self._kind != kind:
            return
        self._completed += advance
        self._progress.update(
            self._task_id,
            advance=advance,
            description=self._format_description(kind),
        )
        if self._total is not None and self._completed >= self._total:
            self._progress.remove_task(self._task_id)
            self._task_id = None
            self._kind = None
            self._total = None
            self._completed = 0


class Neptune2ProgressListener(ProgressListener):
    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._project_progress = ProjectProgress(progress)
        self._kind_progress: dict[ProgressKind, RunAttributeProgress] = {}

    def on_project_total(self, project_id: ProjectId, total: int) -> None:
        self._project_progress.set_total(project_id, total)

    def on_project_advance(self, project_id: ProjectId, advance: int) -> None:
        self._project_progress.advance(project_id, advance)

    def _get(self, kind: ProgressKind) -> RunAttributeProgress:
        if kind not in self._kind_progress:
            self._kind_progress[kind] = RunAttributeProgress(self._progress, kind)
        return self._kind_progress[kind]

    def on_run_total(self, kind: ProgressKind, run_id: SourceRunId, total: int) -> None:
        self._get(kind).set_total(run_id, total)

    def on_run_advance(
        self, kind: ProgressKind, run_id: SourceRunId, advance: int = 1
    ) -> None:
        self._get(kind).advance(run_id, advance)

    def on_run_finished(self, run_id: SourceRunId) -> None:
        for progress in self._kind_progress.values():
            progress.remove(run_id)


class Neptune3ProgressListener(ProgressListener):
    def __init__(
        self,
        project_progress: Progress,
        run_progress: Progress,
        attribute_progress: Progress,
        max_run_lines: int = 16,
    ) -> None:
        self._run_list = RunListProgress(run_progress, max_lines=max_run_lines)
        self._aggregate = AggregateAttributeProgress(attribute_progress)
        self._project_progress = ProjectProgress(project_progress)

    def on_project_total(self, project_id: ProjectId, total: int) -> None:
        self._project_progress.set_total(project_id, total)

    def on_project_advance(self, project_id: ProjectId, advance: int) -> None:
        self._project_progress.advance(project_id, advance)

    def on_run_started(self, run_id: SourceRunId) -> None:
        self._run_list.add_run(run_id)

    def on_run_finished(self, run_id: SourceRunId) -> None:
        self._run_list.remove_run(run_id)

    def on_batch_total(
        self, kind: ProgressKind, run_ids: Sequence[SourceRunId], total: int
    ) -> None:
        self._aggregate.set_total(kind, total)

    def on_batch_advance(
        self, kind: ProgressKind, run_ids: Sequence[SourceRunId], advance: int
    ) -> None:
        self._aggregate.advance(kind, advance)


class NoopProgressListenerFactory(ProgressListenerFactory):
    def create_live(self) -> Live:
        return Live(
            Group(),
            console=get_rich_console(),
            refresh_per_second=10,
            transient=False,
            auto_refresh=False,
        )

    def create_listener(self, live: Live) -> ProgressListener:
        return NoopProgressListener()


class Neptune2ProgressListenerFactory(ProgressListenerFactory):
    def create_live(self) -> Live:
        return Live(
            Group(),
            console=get_rich_console(),
            refresh_per_second=10,
            transient=False,
        )

    def create_listener(self, live: Live) -> ProgressListener:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="progress.description"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=get_rich_console(),
            transient=False,
            auto_refresh=False,
        )
        live.update(Group(progress))
        return Neptune2ProgressListener(progress)


class Neptune3ProgressListenerFactory(ProgressListenerFactory):
    def __init__(self, *, max_run_lines: int = 16) -> None:
        self._max_run_lines = max_run_lines

    def create_live(self) -> Live:
        return Live(
            Group(),
            console=get_rich_console(),
            refresh_per_second=10,
            transient=False,
        )

    def create_listener(self, live: Live) -> ProgressListener:
        project_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="progress.description"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=get_rich_console(),
            transient=False,
            auto_refresh=False,
        )
        run_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="progress.description"),
            console=get_rich_console(),
            transient=False,
            auto_refresh=False,
        )
        attribute_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="progress.description"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=get_rich_console(),
            transient=False,
            auto_refresh=False,
        )
        live.update(Group(project_progress, run_progress, attribute_progress))
        return Neptune3ProgressListener(
            project_progress,
            run_progress,
            attribute_progress,
            max_run_lines=self._max_run_lines,
        )
