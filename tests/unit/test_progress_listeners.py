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

from rich.progress import Progress

from neptune_exporter.progress.listeners import ProjectProgress


def test_project_progress_set_total_resets_completed_on_reuse():
    """Project progress should reset completion when reconfigured for a new phase."""
    progress = Progress()
    project_progress = ProjectProgress(progress)

    project_id = "workspace/project"

    project_progress.set_total(project_id, 2)
    project_progress.advance(project_id, 2)

    project_progress.set_total(project_id, 3)

    task = progress.tasks[0]
    assert task.total == 3
    assert task.completed == 0
