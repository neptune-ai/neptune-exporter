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
from pathlib import Path
from unittest.mock import Mock, patch, call

import pandas as pd
import pyarrow as pa
import pytest

from neptune_exporter.loaders.goodseed_loader import GoodseedLoader
from neptune_exporter import model


def _make_loader(**kwargs):
    """Create a GoodseedLoader with goodseed mocked as available."""
    with patch("neptune_exporter.loaders.goodseed_loader.GOODSEED_AVAILABLE", True):
        return GoodseedLoader(**kwargs)


def _make_table(data: dict) -> pa.Table:
    """Create a PyArrow table from a dict with all schema columns filled."""
    n = len(next(iter(data.values())))
    defaults = {
        "project_id": ["test-project"] * n,
        "run_id": ["RUN-1"] * n,
        "attribute_path": [""] * n,
        "attribute_type": [""] * n,
        "step": [None] * n,
        "timestamp": [None] * n,
        "int_value": [None] * n,
        "float_value": [None] * n,
        "string_value": [None] * n,
        "bool_value": [None] * n,
        "datetime_value": [None] * n,
        "string_set_value": [None] * n,
        "file_value": [None] * n,
        "histogram_value": [None] * n,
    }
    defaults.update(data)
    df = pd.DataFrame(defaults)
    return pa.Table.from_pandas(df, schema=model.SCHEMA)


def _table_gen(*tables):
    """Wrap tables in a generator as expected by upload_run_data."""
    for t in tables:
        yield t


# Initialization


def test_init_raises_without_goodseed():
    """Test that init raises when goodseed is not installed."""
    with patch("neptune_exporter.loaders.goodseed_loader.GOODSEED_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="GoodSeed is not installed"):
            GoodseedLoader()


# find_run


def test_find_run_not_found():
    """Test find_run returns None when run doesn't exist."""
    loader = _make_loader()
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = False
    with patch("goodseed.config.get_run_db_path", return_value=mock_path):
        result = loader.find_run("test-project", "RUN-123", None)
        assert result is None


def test_find_run_exists():
    """Test find_run returns run ID when run exists."""
    loader = _make_loader()
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    with patch("goodseed.config.get_run_db_path", return_value=mock_path):
        result = loader.find_run("test-project", "RUN-123", None)
        assert result == "RUN-123"


# create_run


def test_create_run():
    """Test create_run stores pending run info."""
    loader = _make_loader()
    run_id = loader.create_run("workspace/project", "RUN-123", "my-experiment")

    assert run_id == "RUN-123"
    assert loader._pending_run is not None
    assert loader._pending_run["run_name"] == "RUN-123"
    assert loader._pending_run["project"] == "workspace/project"
    assert loader._pending_run["experiment_name"] == "my-experiment"


# upload_run_data - parameters


def test_upload_parameters():
    """Test uploading all parameter types as GoodSeed configs."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": [
                "config/lr",
                "config/epochs",
                "config/name",
                "config/debug",
                "config/started",
                "config/tags",
            ],
            "attribute_type": [
                "float",
                "int",
                "string",
                "bool",
                "datetime",
                "string_set",
            ],
            "float_value": [0.001, None, None, None, None, None],
            "int_value": [None, 100, None, None, None, None],
            "string_value": [None, None, "experiment-1", None, None, None],
            "bool_value": [None, None, None, True, None, None],
            "datetime_value": [
                None,
                None,
                None,
                None,
                pd.Timestamp("2024-01-15 10:30:00", tz="UTC"),
                None,
            ],
            "string_set_value": [None, None, None, None, None, ["tag1", "tag2"]],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1", "experiment")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    # Should have been called at least twice: once for origin metadata, once for params
    assert mock_run.log_configs.call_count >= 2

    # Collect all logged configs across calls
    all_configs = {}
    for c in mock_run.log_configs.call_args_list:
        all_configs.update(c[0][0])

    assert all_configs["config/lr"] == 0.001
    assert all_configs["config/epochs"] == 100
    assert all_configs["config/name"] == "experiment-1"
    assert all_configs["config/debug"] is True
    assert all_configs["config/tags"] == "tag1,tag2"
    # Neptune origin metadata
    assert all_configs["neptune/project_id"] == "test-project"
    assert all_configs["neptune/run_id"] == "RUN-1"


def test_upload_skips_sys_attributes():
    """Test that internal sys/ attributes are skipped but useful ones pass through."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": [
                "sys/id",
                "sys/state",
                "sys/name",
                "sys/creation_time",
                "sys/modification_time",
                "sys/trashed",
                "sys/failed",
                "config/lr",
            ],
            "attribute_type": [
                "string",
                "string",
                "string",
                "datetime",
                "datetime",
                "bool",
                "bool",
                "float",
            ],
            "string_value": ["RUN-1", "idle", "my-run", None, None, None, None, None],
            "float_value": [None, None, None, None, None, None, None, 0.01],
            "datetime_value": [
                None,
                None,
                None,
                pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-16 12:00:00", tz="UTC"),
                None,
                None,
                None,
            ],
            "bool_value": [None, None, None, None, None, True, False, None],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1", "experiment")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    all_configs = {}
    for c in mock_run.log_configs.call_args_list:
        all_configs.update(c[0][0])

    # sys/id, sys/state, sys/name should be skipped
    assert "sys/id" not in all_configs
    assert "sys/state" not in all_configs
    assert "sys/name" not in all_configs
    # These sys/ attributes should be preserved
    assert "sys/creation_time" in all_configs
    assert "sys/modification_time" in all_configs
    assert all_configs["sys/trashed"] is True
    assert all_configs["sys/failed"] is False
    assert all_configs["config/lr"] == 0.01


def test_upload_includes_monitoring_attributes():
    """Test that monitoring/* attributes are imported (used by GoodSeed frontend)."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["monitoring/gpu/0/memory", "config/lr"],
            "attribute_type": ["float", "float"],
            "float_value": [85.5, 0.01],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    all_configs = {}
    for c in mock_run.log_configs.call_args_list:
        all_configs.update(c[0][0])

    assert all_configs["monitoring/gpu/0/memory"] == 85.5
    assert all_configs["config/lr"] == 0.01


# upload_run_data - metrics


def test_upload_metrics():
    """Test uploading float_series as GoodSeed metrics."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["train/loss", "train/loss", "train/acc"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [Decimal("0"), Decimal("1"), Decimal("0")],
            "float_value": [0.9, 0.5, 0.6],
            "timestamp": [
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
            ],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    assert mock_run.log_metrics.call_count == 3

    calls = mock_run.log_metrics.call_args_list
    assert calls[0] == call({"train/loss": 0.9}, step=0)
    assert calls[1] == call({"train/loss": 0.5}, step=1)
    assert calls[2] == call({"train/acc": 0.6}, step=0)


def test_upload_metrics_with_step_multiplier():
    """Test that step_multiplier scales decimal steps to integers."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["loss"],
            "attribute_type": ["float_series"],
            "step": [Decimal("1.5")],
            "float_value": [0.5],
            "timestamp": [pd.Timestamp("2024-01-15", tz="UTC")],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1000
        )

    mock_run.log_metrics.assert_called_once_with({"loss": 0.5}, step=1500)


def test_upload_metrics_skips_nan_values():
    """Test that metric rows with NaN float_value or step are skipped."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["loss", "loss", "loss"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [Decimal("0"), None, Decimal("2")],
            "float_value": [0.9, 0.5, None],
            "timestamp": [
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
            ],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    # Only the first row has both valid step and float_value
    mock_run.log_metrics.assert_called_once_with({"loss": 0.9}, step=0)


# upload_run_data - string series


def test_upload_string_series():
    """Test uploading string_series as GoodSeed string series."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["logs/info", "logs/info"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("0"), Decimal("1")],
            "string_value": ["Training started", "Epoch 1 done"],
            "timestamp": [
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
            ],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    assert mock_run.log_string_series.call_count == 2
    calls = mock_run.log_string_series.call_args_list
    assert calls[0] == call({"logs/info": "Training started"}, step=0)
    assert calls[1] == call({"logs/info": "Epoch 1 done"}, step=1)


def test_upload_string_series_missing_step_defaults_to_zero():
    """Test that string series rows with missing step default to step=0."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["logs/info"],
            "attribute_type": ["string_series"],
            "step": [None],
            "string_value": ["No step provided"],
            "timestamp": [pd.Timestamp("2024-01-15", tz="UTC")],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    mock_run.log_string_series.assert_called_once_with(
        {"logs/info": "No step provided"}, step=0
    )


# upload_run_data - skipped types


def test_upload_skips_unsupported_types():
    """Test that files and histograms are skipped without crashing the run."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": [
                "artifacts/model",
                "artifacts/checkpoint",
                "hist/weights",
            ],
            "attribute_type": ["file", "file_series", "histogram_series"],
            "file_value": [{"path": "model.pt"}, {"path": "ckpt.pt"}, None],
            "step": [None, None, Decimal("1")],
            "histogram_value": [
                None,
                None,
                {"type": "auto", "edges": [0.0, 1.0], "values": [5.0]},
            ],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    # Run should complete successfully despite unsupported types
    mock_run.close.assert_called_once()
    mock_run.log_metrics.assert_not_called()
    mock_run.log_string_series.assert_not_called()


# upload_run_data - metadata extraction from sys/ attributes


def test_experiment_name_from_sys_name():
    """Test that sys/name is used as experiment_name for the GoodSeed Run."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["sys/name"],
            "attribute_type": ["string"],
            "string_value": ["my-experiment-name"],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ) as mock_run_class:
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    mock_run_class.assert_called_once()
    _, kwargs = mock_run_class.call_args
    assert kwargs["experiment_name"] == "my-experiment-name"


def test_created_at_from_sys_creation_time():
    """Test that sys/creation_time is passed as created_at to the GoodSeed Run."""
    loader = _make_loader()
    mock_run = Mock()

    creation_time = pd.Timestamp("2024-01-15 10:30:00", tz="UTC")

    table = _make_table(
        {
            "attribute_path": ["sys/creation_time"],
            "attribute_type": ["datetime"],
            "datetime_value": [creation_time],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ) as mock_run_class:
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    mock_run_class.assert_called_once()
    _, kwargs = mock_run_class.call_args
    assert kwargs["created_at"] == creation_time.isoformat()


# upload_run_data - error handling


def test_upload_not_prepared():
    """Test that upload_run_data raises when create_run wasn't called."""
    loader = _make_loader()

    with pytest.raises(RuntimeError, match="not prepared"):
        loader.upload_run_data(_table_gen(), "RUN-1", Path("/files"), step_multiplier=1)


def test_upload_closes_run_on_error():
    """Test that run is closed with 'failed' status on error."""
    loader = _make_loader()
    mock_run = Mock()
    mock_run.log_configs.side_effect = [None, Exception("DB error")]

    table = _make_table(
        {
            "attribute_path": ["config/lr"],
            "attribute_type": ["float"],
            "float_value": [0.01],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        with pytest.raises(Exception, match="DB error"):
            loader.upload_run_data(
                _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
            )

    mock_run.close.assert_called_once_with(status="failed")


def test_upload_cleans_up_state():
    """Test that internal state is reset after upload (success or failure)."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["config/lr"],
            "attribute_type": ["float"],
            "float_value": [0.01],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    assert loader._active_run is None
    assert loader._current_run_id is None
    assert loader._pending_run is None


# upload_run_data - multiple chunks


def test_upload_multiple_chunks():
    """Test that multiple data chunks are processed correctly."""
    loader = _make_loader()
    mock_run = Mock()

    # First chunk: parameters + sys/name
    table1 = _make_table(
        {
            "attribute_path": ["sys/name", "config/lr"],
            "attribute_type": ["string", "float"],
            "string_value": ["my-exp", None],
            "float_value": [None, 0.01],
        }
    )

    # Second chunk: metrics
    table2 = _make_table(
        {
            "attribute_path": ["train/loss", "train/loss"],
            "attribute_type": ["float_series", "float_series"],
            "step": [Decimal("0"), Decimal("1")],
            "float_value": [0.9, 0.5],
            "timestamp": [
                pd.Timestamp("2024-01-15", tz="UTC"),
                pd.Timestamp("2024-01-15", tz="UTC"),
            ],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run("test-project", "RUN-1")
        loader.upload_run_data(
            _table_gen(table1, table2), "RUN-1", Path("/files"), step_multiplier=1
        )

    # Configs from chunk 1 + metrics from chunk 2
    assert mock_run.log_configs.call_count >= 2  # origin + params
    assert mock_run.log_metrics.call_count == 2
    mock_run.close.assert_called_once()


# upload_run_data - fork metadata as configs


def test_upload_fork_metadata():
    """Test that fork info is stored as Neptune origin configs."""
    loader = _make_loader()
    mock_run = Mock()

    table = _make_table(
        {
            "attribute_path": ["config/lr"],
            "attribute_type": ["float"],
            "float_value": [0.01],
        }
    )

    with patch(
        "neptune_exporter.loaders.goodseed_loader.goodseed.Run",
        return_value=mock_run,
    ):
        loader.create_run(
            "test-project",
            "RUN-1",
            parent_run_id="RUN-0",
            fork_step=50.0,
        )
        loader.upload_run_data(
            _table_gen(table), "RUN-1", Path("/files"), step_multiplier=1
        )

    # First log_configs call should be origin metadata
    first_configs = mock_run.log_configs.call_args_list[0][0][0]
    assert first_configs["neptune/parent_run_id"] == "RUN-0"
    assert first_configs["neptune/fork_step"] == 50.0
