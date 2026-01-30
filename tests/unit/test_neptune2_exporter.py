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

from decimal import Decimal
from unittest.mock import Mock
import pandas as pd
import pyarrow as pa

from neptune_exporter import model
from neptune_exporter.exporters.error_reporter import ErrorReporter
from neptune_exporter.exporters.neptune2 import Neptune2Exporter


def test_download_metrics_handles_pyarrow_chunked_arrays(monkeypatch):
    """Ensure download_metrics can convert pyarrow chunked columns into a batch."""
    exporter = Neptune2Exporter(error_reporter=Mock(spec=ErrorReporter))

    def fake_workers(*_args, **_kwargs):
        yield "RUN-1", [pd.DataFrame()]

    def just_raise(_project_id, _run_id, e):
        raise e

    chunks = pa.chunked_array([[1.0], [2.0]], type=pa.float64())
    float_series = pd.Series(chunks, dtype="float64[pyarrow]")

    df = pd.DataFrame(
        {
            "project_id": ["proj", "proj"],
            "run_id": ["RUN-1", "RUN-1"],
            "attribute_path": ["metrics/a", "metrics/a"],
            "attribute_type": ["float_series", "float_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * 2,
            "int_value": [None, None],
            "float_value": float_series,
            "string_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
            "file_value": [None, None],
            "histogram_value": [None, None],
        }
    )

    monkeypatch.setattr(exporter, "_run_attribute_workers", fake_workers)
    monkeypatch.setattr(exporter, "_convert_metrics_to_schema", lambda *_: df)
    monkeypatch.setattr(exporter, "_handle_run_exception", just_raise)

    batches = list(exporter.download_metrics("org/proj", ["RUN-1"], None))

    exporter.close()

    assert len(batches) == 1
    assert batches[0].num_rows == 2
    assert batches[0].schema == model.SCHEMA
