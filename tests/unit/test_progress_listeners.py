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


def test_project_progress_start_sets_indeterminate_task():
    """Project progress should show an indeterminate spinner while listing."""
    progress = Progress()
    project_progress = ProjectProgress(progress)

    project_id = "workspace/project"
    project_progress.start(project_id, phase="listing runs")

    task = progress.tasks[0]
    assert task.total is None
    assert task.description == "Project workspace/project (listing runs)"


def test_project_progress_shows_newest_projects_in_live_window():
    """Live view should keep the newest project rows when project count grows."""
    progress = Progress()
    project_progress = ProjectProgress(progress, max_lines=2)

    project_progress.set_total("workspace/project-1", 1)
    project_progress.set_total("workspace/project-2", 1)
    project_progress.set_total("workspace/project-3", 1)

    assert len(progress.tasks) == 2
    assert progress.tasks[0].description == "Project workspace/project-2"
    assert progress.tasks[1].description == "Project workspace/project-3"


def test_project_progress_show_all_restores_hidden_projects():
    """Final render should include hidden projects from the live window."""
    progress = Progress()
    project_progress = ProjectProgress(progress, max_lines=2)

    project_progress.set_total("workspace/project-1", 1)
    project_progress.set_total("workspace/project-2", 1)
    project_progress.set_total("workspace/project-3", 1)
    project_progress.show_all()

    assert len(progress.tasks) == 3
    assert progress.tasks[0].description == "Project workspace/project-1"
    assert progress.tasks[1].description == "Project workspace/project-2"
    assert progress.tasks[2].description == "Project workspace/project-3"
