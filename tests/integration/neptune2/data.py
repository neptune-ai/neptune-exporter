import itertools
import uuid
from dataclasses import (
    dataclass,
)
from datetime import (
    datetime,
    timezone,
)
from typing import Any

from neptune.types import File as NeptuneFile


@dataclass
class ExperimentData:
    name: str
    run_id: str
    config: dict[str, Any]
    string_sets: dict[str, list[str]]
    float_series: dict[str, list[tuple[float, float]]]
    string_series: dict[str, list[tuple[float, str]]]
    files: dict[str, NeptuneFile]
    file_series: dict[str, list[tuple[float, NeptuneFile]]]
    file_sets: dict[str, list[str]]  # just file globs, not NeptuneFile objects
    artifacts: dict[str, str]

    @property
    def all_attribute_names(self) -> set[str]:
        return set(
            itertools.chain(
                self.config.keys(),
                self.string_sets.keys(),
                self.float_series.keys(),
                self.string_series.keys(),
                self.files.keys(),
                self.file_series.keys(),
                self.file_sets.keys(),
                self.artifacts.keys(),
            )
        )


TEST_DATA_VERSION = "2025-10-07"
TEST_PATH = f"test/exporter-{TEST_DATA_VERSION}"
TEST_NOW = datetime(2025, 1, 1, 0, 0, 0, 0, timezone.utc)

TEST_DATA = [
    ExperimentData(
        name="test_exporter_empty",
        run_id=str(uuid.uuid4()),
        config={},
        string_sets={},
        float_series={},
        string_series={},
        files={},
        file_series={},
        file_sets={},
        artifacts={},
    )
] + [
    ExperimentData(
        name=f"test_exporter_{i}",
        run_id=str(uuid.uuid4()),
        config={
            f"{TEST_PATH}/int-value": i,
            f"{TEST_PATH}/float-value": i * 0.1,
            f"{TEST_PATH}/string-value": f"hello_{i}",
            f"{TEST_PATH}/bool-value": i % 2 == 0,
            f"{TEST_PATH}/datetime-value": datetime(
                2025, 1, 1, i, 0, 0, 0, timezone.utc
            ),
        },
        string_sets={
            "sys/tags": [f"string-set_{i}_{j}" for j in range(5)],
        },
        float_series={
            f"{TEST_PATH}/float-series-value_{j}": [
                (k * 0.1, i * 100 + j + k * 0.01) for k in range(100)
            ]
            for j in range(5)
        },
        string_series={
            f"{TEST_PATH}/string-series-value_{j}": [
                (k * 0.1, f"string-series_{i}_{j}_{k}") for k in range(100)
            ]
            for j in range(5)
        },
        files={
            f"{TEST_PATH}/files/file-value": NeptuneFile.from_content(
                f"Binary content {i}"
            ),
            f"{TEST_PATH}/files/file-value.txt": NeptuneFile.from_content(
                content=f"Text content {i}", extension="txt"
            ),
        },
        file_series={
            f"{TEST_PATH}/files/file-series-value_{j}": [
                (
                    k,
                    NeptuneFile.from_path(
                        "tests/integration/neptune2/resources/1x1#000000.gif"
                    ),
                )
                for k in range(3)
            ]
            for j in range(2)
        },
        file_sets={
            f"{TEST_PATH}/files/file-set-value_{j}": [
                "tests/integration/neptune2/resources/1x1#000000.gif"
            ]
            for j in range(2)
        },
        artifacts={
            f"{TEST_PATH}/files/artifact-value": "tests/integration/neptune2/resources/1x1#000000.gif"
        },
    )
    for i in range(3)
]
