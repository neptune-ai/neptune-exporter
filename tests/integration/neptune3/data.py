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

from neptune_scale.types import File as ScaleFile
from neptune_scale.types import Histogram as ScaleHistogram

from neptune_query.internal.retrieval.attribute_types import (
    Histogram as FetcherHistogram,
)
from neptune_query.types import Histogram as OHistogram


@dataclass
class ExperimentData:
    name: str
    run_id: str
    config: dict[str, Any]
    string_sets: dict[str, list[str]]
    float_series: dict[str, list[tuple[float, float]]]
    string_series: dict[str, list[tuple[float, str]]]
    histogram_series: dict[str, list[tuple[float, ScaleHistogram]]]
    files: dict[str, bytes]
    file_series: dict[str, list[tuple[float, bytes]]]

    @property
    def all_attribute_names(self) -> set[str]:
        return set(
            itertools.chain(
                self.config.keys(),
                self.string_sets.keys(),
                self.float_series.keys(),
                self.string_series.keys(),
                self.histogram_series.keys(),
                self.files.keys(),
                self.file_series.keys(),
            )
        )

    def output_histogram_series(self) -> dict[str, list[FetcherHistogram]]:
        return {
            key: [
                OHistogram(type="COUNTING", edges=value.bin_edges, values=value.counts)
                for value in values
            ]
            for key, values in self.histogram_series.items()
        }


TEST_DATA_VERSION = "2025-10-07"
TEST_PATH = f"test/exporter-{TEST_DATA_VERSION}"
TEST_NOW = datetime(2025, 1, 1, 0, 0, 0, 0, timezone.utc)

TEST_DATA = [
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
            f"{TEST_PATH}/string-set-value": [f"string-set_{i}_{j}" for j in range(5)],
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
        histogram_series={
            f"{TEST_PATH}/histogram-series-value_{j}": [
                (
                    k,
                    ScaleHistogram(
                        bin_edges=list(range(6)),
                        counts=[i * 1000 + j * 100 + k * 10 + n for n in range(5)],
                    ),
                )
                for k in range(10)
            ]
            for j in range(5)
        },
        files={
            f"{TEST_PATH}/files/file-value": f"Binary content {i}".encode("utf-8"),
            f"{TEST_PATH}/files/file-value.txt": ScaleFile(
                f"Text content {i}".encode("utf-8"), mime_type="text/plain"
            ),
        },
        file_series={
            f"{TEST_PATH}/file-series-value_{j}": [
                (k, f"file-series_{i}_{j}_{k}".encode("utf-8")) for k in range(3)
            ]
            for j in range(2)
        },
    )
    for i in range(3)
]
