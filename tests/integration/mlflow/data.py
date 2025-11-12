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

"""Test data for MLflow integration tests."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

TEST_PROJECT_ID = "test-project"
TEST_EXPERIMENT_NAME = "test-experiment"
TEST_RUN_IDS = ["RUN-001", "RUN-002", "RUN-003"]
TEST_NOW = datetime(2025, 1, 1, 0, 0, 0, 0, timezone.utc)


@dataclass
class RunData:
    """Test data for a single run."""

    run_id: str
    params: dict[str, Any]
    metrics: dict[str, list[tuple[Decimal, float]]]
    string_series: dict[str, list[tuple[Decimal, str]]]
    histogram_series: dict[str, list[tuple[Decimal, dict[str, Any]]]]
    files: dict[str, str]  # attribute_path -> filename
    artifacts: dict[str, str]  # attribute_path -> filename
    file_series: dict[
        str, list[tuple[Decimal, str]]
    ]  # attribute_path -> [(step, filename), ...]
    file_sets: dict[str, str]  # attribute_path -> directory name (for file_set)


def get_test_runs() -> list[RunData]:
    """Get test run data for MLflow integration tests."""
    runs = []
    for i, run_id in enumerate(TEST_RUN_IDS):
        runs.append(
            RunData(
                run_id=run_id,
                params={
                    "sys/name": TEST_EXPERIMENT_NAME,
                    "test/param/int": 42 + i,
                    "test/param/float": 3.14 + i,
                    "test/param/string": f"test-value-{i}",
                    "test/param/bool": i % 2 == 0,
                    "test/param/datetime": TEST_NOW,
                    "test/param/string_set": [f"tag-{i}-{j}" for j in range(3)],
                },
                metrics={
                    "test/metric/accuracy": [
                        (Decimal(str(step)), 0.5 + step * 0.1 + i * 0.01)
                        for step in range(5)
                    ],
                },
                string_series={
                    "test/string_series/logs": [
                        (Decimal(str(step)), f"log-entry-{i}-{step}")
                        for step in range(3)
                    ],
                },
                histogram_series={
                    "test/histogram_series/gradients": [
                        (
                            Decimal(str(step)),
                            {
                                "type": "histogram",
                                "edges": [0.0, 1.0, 2.0, 3.0],
                                "values": [10.0 + step, 20.0 + step, 30.0 + step],
                            },
                        )
                        for step in range(3)
                    ],
                },
                files={
                    "test/file/data": f"test_file_{run_id}.txt",
                },
                artifacts={
                    "test/artifact/data": f"test_artifact_{run_id}/files_list.json",
                },
                file_series={
                    "test/file_series/data": [
                        (Decimal(str(step)), f"test_file_series_{run_id}_{step}.txt")
                        for step in range(3)
                    ],
                },
                file_sets={
                    "test/file_set/data": f"test_file_set_{run_id}",
                },
            )
        )
    return runs


TEST_RUNS = get_test_runs()
