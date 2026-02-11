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

"""Project selection and validation for export CLI."""

from collections.abc import Callable, Sequence
import re

import click


def resolve_export_project_ids(
    project_ids: Sequence[str],
    workspace: str | None,
    project_patterns: Sequence[str],
    project_exclude_patterns: Sequence[str],
    env_project: str | None,
    discover_workspace_projects: Callable[[str], list[str]],
) -> list[str]:
    """Resolve effective project IDs for export command."""
    has_workspace = workspace is not None
    has_filters = bool(project_patterns or project_exclude_patterns)
    has_explicit_ids = bool(project_ids)

    if has_filters and not has_workspace:
        raise click.BadParameter(
            "--project-pattern and --project-exclude-pattern require --workspace."
        )

    if has_explicit_ids and (has_workspace or has_filters):
        raise click.BadParameter(
            "Cannot use --project-ids/-p together with --workspace, "
            "--project-pattern, or --project-exclude-pattern."
        )

    if has_workspace:
        if workspace is None:
            raise RuntimeError("workspace must be provided in discovery mode")

        include_regexes = _compile_regexes(
            project_patterns, option_name="--project-pattern"
        )
        exclude_regexes = _compile_regexes(
            project_exclude_patterns, option_name="--project-exclude-pattern"
        )

        resolved = discover_workspace_projects(workspace)

        if include_regexes:
            resolved = [
                project_id
                for project_id in resolved
                if any(
                    regex.search(_project_name_from_project_id(project_id))
                    for regex in include_regexes
                )
            ]

        if exclude_regexes:
            resolved = [
                project_id
                for project_id in resolved
                if not any(
                    regex.search(_project_name_from_project_id(project_id))
                    for regex in exclude_regexes
                )
            ]

        resolved = _dedupe_preserve_order(resolved)
        if not resolved:
            filters = []
            if project_patterns:
                filters.append(f"include={list(project_patterns)}")
            if project_exclude_patterns:
                filters.append(f"exclude={list(project_exclude_patterns)}")
            filters_suffix = f" with filters ({', '.join(filters)})" if filters else ""
            raise click.BadParameter(
                f"No projects found in workspace '{workspace}'{filters_suffix}."
            )
    else:
        if project_ids:
            resolved = list(project_ids)
        elif env_project:
            resolved = [env_project]
        else:
            raise click.BadParameter(
                "No project IDs provided. Either use --project-ids/-p option, "
                "set NEPTUNE_PROJECT environment variable, or use --workspace."
            )

        resolved = _dedupe_preserve_order(resolved)

    for project_id in resolved:
        if not project_id.strip():
            raise click.BadParameter(
                "Project ID cannot be empty. Please provide a valid project ID."
            )

    return resolved


def _compile_regexes(
    patterns: Sequence[str], option_name: str
) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as e:
            raise click.BadParameter(
                f"Invalid regex for {option_name}: {pattern!r} ({e})"
            ) from e
    return compiled


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _project_name_from_project_id(project_id: str) -> str:
    if "/" not in project_id:
        return project_id
    return project_id.split("/", maxsplit=1)[1]
