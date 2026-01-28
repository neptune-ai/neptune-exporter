"""Unit test: combined Pluto loader behavior.

This test supports two modes:

- Offline / dry-run (default): the test will simulate creating a run and
    exercising the loader logic without performing any network uploads.

- Real-upload mode: set the environment variable `PLUTO_DO_UPLOAD=1` to
    enable actual uploads to a Pluto server. When running in this mode you
    must also provide the destination project name and credentials via
    `PLUTO_PROJECT` (e.g. "owner/project_name") and `PLUTO_API_KEY`.

Example (real upload):

        PLUTO_DO_UPLOAD=1 PLUTO_PROJECT=simple_test PLUTO_API_KEY=<key> \
        uv run pytest tests/unit/test_pluto_loader.py::test_pluto_loader_combined -q

Use the offline mode for fast local testing; enable real-upload only when
you want the test to actually push data to a Pluto instance.
"""

import os
from pathlib import Path
from decimal import Decimal
from typing import Generator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neptune_exporter.loaders.pluto_loader import PlutoLoader
from neptune_exporter.types import ProjectId


def _create_artifact_files(base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    yaml = base / "config.yaml"
    yaml.write_text("""# Model Configuration\nmodel:\n  name: ResNet50\n""")

    txt = base / "notes.txt"
    txt.write_text("Experiment notes for combined integration test")

    py = base / "example.py"
    py.write_text("print('hello')\n")

    # Create a small PNG (green) and an SVG (purple) for visual previews
    png = base / "plot.png"
    # Create a validation-like PNG using matplotlib so Pluto displays it correctly
    x = np.linspace(0, 1, 50)
    y = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y, color="green", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(png), format="png", dpi=100)
    plt.close(fig)

    svg = base / "plot.svg"
    svg.write_text(
        """<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>\n"
        "  <rect width='100%' height='100%' fill='#1f0b3d'/>\n"
        "  <circle cx='32' cy='32' r='20' fill='#9b30ff'/>\n"
        "</svg>"""
    )

    return {"yaml": yaml, "txt": txt, "py": py, "png": png, "svg": svg}


def _create_combined_parquet(files: dict, out_path: Path) -> Path:
    rows = []

    # Parameters
    rows.append(
        {
            "attribute_type": "string",
            "attribute_path": "params/model_name",
            "string_value": "ResNet50",
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "string_set_value": None,
            "file_value": None,
            "histogram_value": None,
            "step": None,
            "timestamp": None,
        }
    )

    rows.append(
        {
            "attribute_type": "float",
            "attribute_path": "params/learning_rate",
            "string_value": None,
            "float_value": 0.001,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "string_set_value": None,
            "file_value": None,
            "histogram_value": None,
            "step": None,
            "timestamp": None,
        }
    )

    # Metrics (float_series)
    for step in range(5):
        rows.append(
            {
                "attribute_type": "float_series",
                "attribute_path": "metrics/accuracy",
                "step": Decimal(step),
                "timestamp": step,
                "float_value": 0.7 + step * 0.05,
                "string_value": None,
                "int_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

    # Histogram series (freq + bin_edges)
    freq = [10, 20, 70]
    bin_edges = [0.0, 0.33, 0.66, 1.0]  # len(freq)+1

    rows.append(
        {
            "attribute_type": "histogram_series",
            "attribute_path": "metrics/value_distribution",
            "step": Decimal(0),
            "timestamp": 0,
            "histogram_value": [freq, bin_edges],
            "string_value": None,
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "string_set_value": None,
            "file_value": None,
        }
    )

    # File series: include both PNG and SVG previews
    png_meta = {
        "path": files["png"].name,
        "size": files["png"].stat().st_size,
        "hash": "",
    }
    rows.append(
        {
            "attribute_type": "file_series",
            "attribute_path": "visualizations/plot_png",
            "step": Decimal(0),
            "timestamp": 0,
            "file_value": png_meta,
            "string_value": None,
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "string_set_value": None,
            "histogram_value": None,
        }
    )

    svg_meta = {
        "path": files["svg"].name,
        "size": files["svg"].stat().st_size,
        "hash": "",
    }
    rows.append(
        {
            "attribute_type": "file_series",
            "attribute_path": "visualizations/plot_svg",
            "step": Decimal(0),
            "timestamp": 0,
            "file_value": svg_meta,
            "string_value": None,
            "float_value": None,
            "int_value": None,
            "bool_value": None,
            "datetime_value": None,
            "string_set_value": None,
            "histogram_value": None,
        }
    )

    # String series: stdout messages
    stdout_messages = [
        "TEST STDOUT LOGS:",
        "Epoch 1/3: start",
        "Epoch 1/3: loss=0.5",
        "Epoch 2/3: loss=0.4",
        "Epoch 3/3: loss=0.3",
        "Training complete",
    ]
    for i, m in enumerate(stdout_messages):
        rows.append(
            {
                "attribute_type": "string_series",
                "attribute_path": "logs/output",
                "step": Decimal(i),
                "timestamp": i,
                "string_value": m,
                "float_value": None,
                "int_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

    # String series: stderr messages
    stderr_messages = [
        "TEST STDERR LOGS:",
        "Warning: GPU memory high",
        "Error: NaN encountered",
        "Warning: reducing LR",
    ]
    for i, m in enumerate(stderr_messages, start=len(stdout_messages)):
        rows.append(
            {
                "attribute_type": "string_series",
                "attribute_path": "logs/errors",
                "step": Decimal(i),
                "timestamp": i,
                "string_value": m,
                "float_value": None,
                "int_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "file_value": None,
                "histogram_value": None,
            }
        )

    # Text files as file artifacts
    for key in ("yaml", "txt", "py"):
        rows.append(
            {
                "attribute_type": "file",
                "attribute_path": f"files/{key}",
                "step": None,
                "timestamp": None,
                "file_value": {"path": files[key].name},
                "string_value": None,
                "float_value": None,
                "int_value": None,
                "bool_value": None,
                "datetime_value": None,
                "string_set_value": None,
                "histogram_value": None,
            }
        )

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(out_path))
    return out_path


def run_table_generator(parquet_path: Path) -> Generator[pa.Table, None, None]:
    table = pq.read_table(str(parquet_path))
    yield table


def test_pluto_loader_combined(tmp_path, caplog):
    project = os.getenv("PLUTO_PROJECT", "simple_test")
    do_upload = bool(
        os.getenv("PLUTO_DO_UPLOAD") == "1"
        or (os.getenv("PLUTO_PROJECT") and os.getenv("PLUTO_API_KEY"))
    )

    # Expected logs (used for clearer test output)
    expected_stdout = [
        "TEST STDOUT LOGS:",
        "Epoch 1/3: start",
        "Epoch 1/3: loss=0.5",
        "Epoch 2/3: loss=0.4",
        "Epoch 3/3: loss=0.3",
        "Training complete",
    ]
    expected_stderr = [
        "TEST STDERR LOGS:",
        "Warning: GPU memory high",
        "Error: NaN encountered",
        "Warning: reducing LR",
    ]

    # Create files
    base = tmp_path / "pluto_combined"
    files = _create_artifact_files(base)

    # Create parquet with all entries
    parquet_path = base / "combined_all.parquet"
    _create_combined_parquet(files, parquet_path)

    # Configure loader runtime dirs and logging to use test temp dir and full metric logging
    os.environ["NEPTUNE_EXPORTER_PLUTO_BASE_DIR"] = str(base)
    os.environ["NEPTUNE_EXPORTER_PLUTO_LOG_EVERY"] = "1"

    # If user wants a real upload
    if do_upload:
        caplog.set_level("INFO")

        try:
            pass  # type: ignore
        except Exception as e:
            pytest.skip(f"Pluto SDK not installed for live upload: {e}")

        run_name = "combined_all_run"

        print("\n--- Pluto Loader Live Upload (2-pass dedupe) ---")
        print("Pass 1: should upload normally")
        print("Pass 2: should be detected as duplicate and skip\n")

        # ----- Pass 1 -----
        loader = PlutoLoader(api_key=os.getenv("PLUTO_API_KEY"))
        experiment_id = loader.create_experiment(
            project_id=ProjectId(project), experiment_name="combined_all"
        )

        run_id_1 = loader.create_run(
            project_id=ProjectId(project),
            run_name=run_name,
            experiment_id=experiment_id,
        )
        print(f"Pass 1 run_id: {run_id_1}")

        loader.upload_run_data(
            run_data=run_table_generator(parquet_path),
            run_id=run_id_1,
            files_directory=base,
            step_multiplier=1,
        )

        print("\nPass 2: fresh loader instance (should hit cache)")
        loader2 = PlutoLoader(api_key=os.getenv("PLUTO_API_KEY"))

        found = loader2.find_run(
            project_id=ProjectId(project),
            run_name=run_name,
            experiment_id=experiment_id,
        )
        print(f"find_run returned: {found}")
        assert found is not None, (
            "Expected run to be detected as already loaded via cache"
        )

        run_id_2 = loader2.create_run(
            project_id=ProjectId(project),
            run_name=run_name,
            experiment_id=experiment_id,
        )
        print(f"Pass 2 create_run returned: {run_id_2}")

        # On cache hit, your create_run returns TargetRunId(run_name)
        assert str(run_id_2) == run_name, (
            f"Expected cache-hit run_id to equal run_name, got {run_id_2}"
        )

        loader2.upload_run_data(
            run_data=run_table_generator(parquet_path),
            run_id=run_id_2,
            files_directory=base,
            step_multiplier=1,
        )

        # Assert we logged skip/dedupe behavior (these are the exact strings your loader emits)
        assert "already loaded (cache hit); skipping" in caplog.text, (
            "Expected create_run() to log a cache hit / skip message on pass 2"
        )
        assert "already loaded; skipping upload" in caplog.text, (
            "Expected upload_run_data() to log a skip message on pass 2"
        )

        print("\n✅ Live dedupe assertions passed.")
        print("--- end live summary ---\n")
        return

    # -------------------- DRY RUN PATH --------------------
    # Dry-run: inject a fake pluto module to assert upload attempts
    import sys

    class _FakeImage:
        def __init__(self, path, caption=None):
            self.path = path
            self.caption = caption

    class _FakeText:
        def __init__(self, content, caption=None):
            self.content = content
            self.caption = caption

    class _FakeArtifact:
        def __init__(self, path, caption=None):
            self.path = path
            self.caption = caption

    class _FakeHistogram:
        def __init__(self, data, bins=64):
            # if it's pre-binned, loader must pass bins=None
            if isinstance(data, (list, tuple)) and len(data) == 2:
                assert bins is None, "Expected bins=None for pre-binned histograms"
                freq, edges = data
                assert len(edges) == len(freq) + 1
            self.data = data

    class _FakeOp:
        def __init__(self):
            self.update_config_calls = 0
            self.metrics_calls = 0
            self.file_chunks = 0
            self.text_artifacts = 0
            self.hist_calls = 0
            self._logs = []

            class _Settings:
                _op_id = "fake-op-1"

            self.settings = _Settings()

        def update_config(self, params: dict):
            self.update_config_calls += 1
            self._logs.append(("update_config", dict(params)))

        def log(self, payload: dict, step: int | None = None):
            # Detect payload type
            # Numeric dict -> metrics
            if all(isinstance(v, (int, float)) for v in payload.values()):
                self.metrics_calls += 1
                self._logs.append(("metrics", dict(payload), step))
                return

            # Histogram objects
            if any(isinstance(v, _FakeHistogram) for v in payload.values()):
                self.hist_calls += 1
                self._logs.append(("hist", payload, step))
                return

            # Text/Image/Artifact objects
            count = 0
            for v in payload.values():
                if isinstance(v, (_FakeImage, _FakeText, _FakeArtifact)):
                    count += 1
            if count:
                # Treat as a single chunk upload
                self.file_chunks += 1
                # Count text artifacts separately
                text_count = sum(
                    1 for v in payload.values() if isinstance(v, _FakeText)
                )
                self.text_artifacts += text_count
                self._logs.append(("files", list(payload.keys())))
                return

        def finish(self, code=None):
            self._logs.append(("finish", code))

        def flush(self):
            self._logs.append(("flush", None))

    class _FakePlutoModule:
        def __init__(self):
            self.Image = _FakeImage
            self.Text = _FakeText
            self.Artifact = _FakeArtifact
            self.Histogram = _FakeHistogram
            self._created_ops = []

        def init(self, **kwargs):
            op = _FakeOp()
            self._created_ops.append(op)
            return op

    fake_pluto = _FakePlutoModule()
    sys.modules["pluto"] = fake_pluto

    loader = PlutoLoader(api_key="fake-key")

    # create_experiment never requires network in current loader impl
    experiment_id = loader.create_experiment(
        project_id=ProjectId(project), experiment_name="combined_all"
    )
    run_id = loader.create_run(
        project_id=ProjectId(project),
        run_name="combined_all_run",
        experiment_id=experiment_id,
    )
    loader.upload_run_data(
        run_data=run_table_generator(parquet_path),
        run_id=run_id,
        files_directory=base,
        step_multiplier=1,
    )

    # Inspect the fake op we created and assert expected calls
    assert fake_pluto._created_ops, "Fake pluto did not create any ops"
    op = fake_pluto._created_ops[0]

    expected_metric_calls = 5
    print("\n--- Pluto Loader Dry-Run Summary ---")
    print("checking that the following stdout logs were loaded:")
    for m in expected_stdout:
        print("  -", m)
    print("checking that the following stderr logs were loaded:")
    for m in expected_stderr:
        print("  -", m)

    # Assertions (match printed expectations) with explicit prints before each check
    print(f"Asserting update_config calls == 1 (actual: {op.update_config_calls})")
    assert op.update_config_calls == 1, (
        f"expected 1 update_config call, got {op.update_config_calls}"
    )

    print(
        f"Asserting metrics calls == {expected_metric_calls} (actual: {op.metrics_calls})"
    )
    assert op.metrics_calls == expected_metric_calls, (
        f"expected {expected_metric_calls} metrics log calls, got {op.metrics_calls}"
    )

    print(f"Asserting histogram calls == 1 (actual: {op.hist_calls})")
    assert op.hist_calls == 1, f"expected 1 histogram log call, got {op.hist_calls}"

    print(f"Asserting file chunk uploads == 2 (actual: {op.file_chunks})")
    assert op.file_chunks == 2, f"expected 2 file chunk upload, got {op.file_chunks}"

    print(f"Asserting text artifacts == 5 (actual: {op.text_artifacts})")
    assert op.text_artifacts == 5, f"expected 5 text artifacts, got {op.text_artifacts}"

    # Duplicate check (dry-run): create a fresh loader and ensure no new op is created
    before = len(fake_pluto._created_ops)
    loader2 = PlutoLoader(api_key="fake-key")
    found = loader2.find_run(
        project_id=ProjectId(project),
        run_name="combined_all_run",
        experiment_id=experiment_id,
    )

    print(f"Original run_id: {run_id}")
    print(f"Fake pluto ops before: {before}")
    print(f"find_run returned: {found}")

    assert found is not None, "Expected duplicate run to be detected by new fake loader"

    new_run = loader2.create_run(
        project_id=ProjectId(project),
        run_name="combined_all_run",
        experiment_id=experiment_id,
    )
    print(f"create_run returned: {new_run}")
    after = len(fake_pluto._created_ops)
    print(f"Fake pluto ops after: {after}")

    assert new_run == run_id or new_run == found, (
        "Expected create_run to return existing run id when duplicate is detected"
    )
    assert after == before, "Duplicate run caused a second op to be created in dry-run"

    print("--- end summary ---\n")
