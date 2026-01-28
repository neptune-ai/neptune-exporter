#!/usr/bin/env python3
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

import gc
import logging
import os
import re
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import pyarrow as pa

from neptune_exporter.types import ProjectId, TargetExperimentId, TargetRunId
from neptune_exporter.loaders.loader import DataLoader


class PlutoLoader(DataLoader):
    """Loads Neptune-exported parquet data into Pluto (pluto-ml), memory-hardened."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
        skip_runtime_attachments: bool = True,
    ) -> None:
        self._logger = logging.getLogger(__name__)

        # Add StreamHandler so logs go to stdout (Pluto's ConsoleHandler will capture them)
        # Only add if not already present to avoid duplicate logs
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s %(name)s:%(levelname)s: %(message)s")
            )
            self._logger.addHandler(stream_handler)

        self.api_key = api_key
        self.host = host
        self.name_prefix = name_prefix
        self._skip_runtime_attachments = skip_runtime_attachments

        # Smaller defaults; Pluto tends to buffer internally.
        self._batch_rows: int = int(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_BATCH_ROWS", "10000")
        )
        # Minimum of 1000 to prevent inefficient tiny batches
        self._batch_rows = max(1000, self._batch_rows)

        # Default to downsampling; lossless metrics with 900k steps is unfeasible
        self._log_every: int = int(os.getenv("NEPTUNE_EXPORTER_PLUTO_LOG_EVERY", "50"))
        if self._log_every <= 0:
            self._log_every = 1

        # Flush buffered metrics when buffer reaches this many steps
        self._flush_every: int = int(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_FLUSH_EVERY", "1000")
        )
        if self._flush_every < 100:
            self._flush_every = 100

        # File upload chunking
        self._file_chunk_size: int = int(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_FILE_CHUNK_SIZE", "100")
        )
        if self._file_chunk_size < 1:
            self._file_chunk_size = 100
        self._file_chunk_sleep: float = float(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_FILE_CHUNK_SLEEP", "0.5")
        )

        # Optional hard cap to prevent pathological runs (0 = disabled)
        self._max_files_per_run: int = int(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_MAX_FILES_PER_RUN", "0")
        )
        if self._max_files_per_run < 0:
            self._max_files_per_run = 0

        # Base dir
        self._pluto_base_dir = Path(
            os.getenv("NEPTUNE_EXPORTER_PLUTO_BASE_DIR", ".")
        ).resolve()
        self._pluto_base_dir.mkdir(parents=True, exist_ok=True)

        # Duplicate cache (no legacy fallback)
        cache_default = self._pluto_base_dir / ".pluto_upload_cache.txt"
        env_cache_path = os.getenv("NEPTUNE_EXPORTER_PLUTO_LOADED_CACHE")
        self._loaded_cache_path = Path(env_cache_path or cache_default).resolve()

        self._loaded_run_keys: set[str] = set()
        try:
            # Read current cache (single supported name)
            if self._loaded_cache_path.exists():
                self._loaded_run_keys = set(
                    line.strip()
                    for line in self._loaded_cache_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
        except Exception:
            self._logger.debug(
                "Failed reading local Pluto loaded-run cache", exc_info=True
            )

        # Log resolved cache path for visibility
        self._logger.info(
            "Pluto loaded-run cache path: %s", str(self._loaded_cache_path)
        )

        # Import Pluto
        try:
            import pluto  # type: ignore

            self._pluto = pluto
            self._client = None
            if hasattr(pluto, "Client"):
                try:
                    self._client = pluto.Client(api_key=api_key, host=host)
                except Exception:
                    self._logger.debug(
                        "pluto.Client failed; will rely on pluto.init()", exc_info=True
                    )
            else:
                self._client = pluto
        except Exception:
            self._pluto = None
            self._client = None
            self._logger.debug("pluto SDK not available", exc_info=True)

        if show_client_logs:
            logging.getLogger("pluto").setLevel(logging.INFO)
        else:
            logging.getLogger("pluto").setLevel(logging.ERROR)

        self._ops: dict[str, object] = {}
        self._run_id_to_key: dict[str, str] = {}
        self._last_experiment_name: Optional[str] = None

    def _ensure_pluto(self):
        if self._pluto is None:
            raise RuntimeError("Pluto SDK not available. Install `pluto-ml`.")
        return self._pluto

    def _run_key(self, project_id: ProjectId, run_name: str) -> str:
        return f"{project_id}::{run_name}"

    def _as_list(self, x):
        """Convert numpy/arrow-ish containers to plain Python lists."""
        try:
            if hasattr(x, "tolist"):
                return x.tolist()
        except Exception:
            pass
        return list(x)

    def _mark_run_loaded(self, key: str) -> None:
        self._loaded_run_keys.add(key)
        try:
            self._loaded_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._loaded_cache_path.open("a", encoding="utf-8") as f:
                f.write(key + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # fsync may not be available on some filesystems; non-fatal
                    self._logger.debug(
                        "os.fsync call failed (non-fatal)", exc_info=True
                    )
            self._logger.info(
                "Pluto: recorded loaded run in cache: %s -> %s",
                key,
                str(self._loaded_cache_path),
            )
        except Exception:
            self._logger.error(
                "Failed writing local Pluto loaded-run cache at %s",
                str(self._loaded_cache_path),
                exc_info=True,
            )

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.\s/]", "_", str(attribute_path))
        if len(sanitized) > 250:
            sanitized = sanitized[:250]
        return sanitized

    def _convert_step_to_int_optional(
        self, step: Optional[Decimal], step_multiplier: int
    ) -> Optional[int]:
        if step is None or pd.isna(step):
            return None
        return int(float(step) * step_multiplier)

    def _ensure_runtime_dirs(self, pluto_project: str, run_name: str) -> None:
        if not self._skip_runtime_attachments:
            return
        p = (
            self._pluto_base_dir
            / ".pluto"
            / pluto_project
            / run_name
            / "files"
            / "runtime"
        )
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._logger.debug(
                "Failed to create runtime dir (non-fatal): %s", p, exc_info=True
            )

    def _maybe_flush(self, op: object) -> None:
        try:
            if hasattr(op, "flush"):
                op.flush()
        except Exception:
            self._logger.debug("op.flush failed (non-fatal)", exc_info=True)

    def create_experiment(
        self, project_id: ProjectId, experiment_name: str
    ) -> TargetExperimentId:
        # Allow overriding project name via env var (useful for multiple imports)
        override = os.getenv("NEPTUNE_EXPORTER_PLUTO_PROJECT_NAME")
        if override:
            target_name = override
        else:
            target_name = str(project_id)
            if self.name_prefix:
                target_name = f"{self.name_prefix}/{target_name}"

        self._last_experiment_name = experiment_name
        self._logger.info("Pluto client not required; using project name as id")
        self._logger.info("Using Pluto project/experiment %s", target_name)
        return TargetExperimentId(target_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Check if run was already loaded (via cache only).

        Pluto SDK doesn't have list_runs/find_run methods, so we only check
        the local cache file. To enable true deduplication, use run_id parameter
        in pluto.init() so Pluto resumes existing runs by external ID.
        """
        # Use the actual Pluto project/experiment name (experiment_id if provided)
        pluto_project = str(experiment_id) if experiment_id else str(project_id)
        key = f"{pluto_project}::{run_name}"
        if key in self._loaded_run_keys:
            return TargetRunId(run_name)
        return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        pluto = self._ensure_pluto()

        # Use the actual Pluto project name (experiment_id if provided) to build a stable key
        pluto_project = str(experiment_id) if experiment_id else str(project_id)
        key = f"{pluto_project}::{run_name}"
        if key in self._loaded_run_keys:
            self._logger.info("Run '%s' already loaded (cache hit); skipping", run_name)
            return TargetRunId(run_name)

        tags = ["import:neptune", f"import_project:{project_id}"]

        # Use Neptune's experiment name (sys/name) as the Pluto display name if available,
        # falling back to the custom_run_id passed as run_name
        display_name = self._last_experiment_name or run_name
        self._last_experiment_name = None

        self._ensure_runtime_dirs(pluto_project, run_name)

        # Generate stable external_id from project_id + run_name
        # This allows Pluto to detect and resume existing runs instead of creating duplicates
        external_id = f"{project_id}::{run_name}"

        init_kwargs = dict(
            dir=str(self._pluto_base_dir),
            project=pluto_project,
            name=display_name,
            config=None,
            tags=tags,
            run_id=external_id,  # Enable Pluto-side deduplication
        )
        # Pass api_key/host to pluto.init when available to avoid interactive auth prompts
        if self.api_key:
            init_kwargs["api_key"] = self.api_key
        if self.host:
            init_kwargs["host"] = self.host

        op = pluto.init(**init_kwargs)

        run_id = getattr(getattr(op, "settings", None), "_op_id", None) or run_name
        run_id_str = str(run_id)

        self._ops[run_id_str] = op
        self._run_id_to_key[run_id_str] = key

        self._logger.info(
            "Created Pluto Op run '%s' with id %s (external_id=%s)",
            run_name,
            run_id_str,
            external_id,
        )
        return TargetRunId(run_id_str)

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        run_id_str = str(run_id)
        self._logger.info("Starting upload_run_data for run %s", run_id_str)

        op = self._ops.get(run_id_str)

        if op is None:
            if any(k.endswith(f"::{run_id_str}") for k in self._loaded_run_keys):
                self._logger.info(
                    "Run '%s' already loaded; skipping upload", run_id_str
                )
                return
            raise RuntimeError(f"No active Pluto Op for run {run_id_str}")

        pluto = self._ensure_pluto()
        # Preserve loaded key before _finish_op clears internal maps
        loaded_key: Optional[str] = self._run_id_to_key.get(run_id_str)
        self._logger.info(
            "Pluto SDK ensured, starting data processing for run %s", run_id_str
        )

        # Log detected configuration
        self._logger.info(
            "Pluto loader config: BATCH_ROWS=%d, LOG_EVERY=%d, FLUSH_EVERY=%d, "
            "FILE_CHUNK_SIZE=%d, FILE_CHUNK_SLEEP=%.1f",
            self._batch_rows,
            self._log_every,
            self._flush_every,
            self._file_chunk_size,
            self._file_chunk_sleep,
        )

        any_logged = False
        op_log_call_count = 0  # Track total op.log() calls

        # Accumulate all data to log once at the end (avoids thread exhaustion)
        # Metrics: stream in chunks to bound memory (don't accumulate all 900k steps)
        # This avoids RAM explosion and allows real-time logging as data arrives
        streamed_metrics_count: int = 0
        buffered_metrics: dict[
            int | None, dict[str, float]
        ] = {}  # step -> {metric_name: value}
        all_files_list: list[
            dict[str, object]
        ] = []  # List of file batches to log chunked
        stdout_lines: list[str] = []  # Collect all stdout string_series
        stderr_lines: list[str] = []  # Collect all stderr string_series
        all_hists: dict[int | None, dict[str, object]] = {}  # step -> {name: Histogram}
        all_file_items_queued = 0

        # Counters for detailed logging
        params_count: int = 0
        metric_series_count: int = 0
        file_series_count: int = 0
        file_count: int = 0
        artifact_count: int = 0
        string_series_count: int = 0
        histogram_series_count: int = 0

        last_heartbeat_ts = time.time()

        try:
            batch_count = 0
            total_rows_processed = 0
            for part in run_data:
                self._logger.debug("Received part from run_data generator")
                for batch in part.to_batches(max_chunksize=self._batch_rows):
                    batch_count += 1
                    total_rows_processed += len(batch)
                    if batch_count % 10 == 0:
                        self._logger.info(
                            "Processing parquet batch %d (%d rows, %d total rows)",
                            batch_count,
                            len(batch),
                            total_rows_processed,
                        )
                    batch_table = pa.Table.from_batches([batch])
                    df = batch_table.to_pandas(split_blocks=True, self_destruct=True)

                    # -------- Params --------
                    param_types = {
                        "float",
                        "int",
                        "string",
                        "bool",
                        "datetime",
                        "string_set",
                    }
                    params_df = df[df["attribute_type"].isin(param_types)]
                    if not params_df.empty:
                        params: dict[str, object] = {}
                        for _, row in params_df.iterrows():
                            params_count += 1
                            name = self._sanitize_attribute_name(row["attribute_path"])
                            atype = row["attribute_type"]
                            if atype == "float" and pd.notna(row.get("float_value")):
                                params[name] = float(row["float_value"])
                            elif atype == "int" and pd.notna(row.get("int_value")):
                                params[name] = int(row["int_value"])
                            elif atype == "string" and pd.notna(
                                row.get("string_value")
                            ):
                                params[name] = str(row["string_value"])
                            elif atype == "bool" and pd.notna(row.get("bool_value")):
                                params[name] = bool(row["bool_value"])
                            elif atype == "datetime" and pd.notna(
                                row.get("datetime_value")
                            ):
                                params[name] = str(row["datetime_value"])
                            elif (
                                atype == "string_set"
                                and row.get("string_set_value") is not None
                            ):
                                params[name] = list(row["string_set_value"])
                        if params and hasattr(op, "update_config"):
                            try:
                                self._logger.info(
                                    "Pluto: update_config with %d params", len(params)
                                )
                                op.update_config(params)
                                op_log_call_count += 1
                                any_logged = True
                            except Exception:
                                self._logger.debug(
                                    "op.update_config failed", exc_info=True
                                )

                    # -------- Metrics (float_series) --------
                    metrics_df = df[df["attribute_type"] == "float_series"]
                    if not metrics_df.empty:
                        metric_series_count += len(metrics_df)
                        for _, row in metrics_df.iterrows():
                            step = self._convert_step_to_int_optional(
                                row.get("step"), step_multiplier
                            )
                            if step is None or pd.isna(row.get("float_value")):
                                continue

                            # Standard downsampling
                            if self._log_every > 1 and (step % self._log_every) != 0:
                                continue

                            attr_name = self._sanitize_attribute_name(
                                row.get("attribute_path")
                            )
                            value = float(row.get("float_value"))

                            # Stream: buffer and flush when buffer gets large to avoid RAM explosion
                            # Note: if multiple values logged for same (step, metric), we keep only the last one
                            # (same behavior as dict overwrite; acceptable since Neptune export should have 1 per step)
                            if step not in buffered_metrics:
                                buffered_metrics[step] = {}
                            buffered_metrics[step][attr_name] = value

                            # Flush buffer when it reaches threshold
                            if len(buffered_metrics) >= self._flush_every:
                                self._logger.info(
                                    "Pluto: flushing %d metric steps (buffer full)",
                                    len(buffered_metrics),
                                )
                                for buf_step in sorted(
                                    [
                                        k
                                        for k in buffered_metrics.keys()
                                        if k is not None
                                    ]
                                ):
                                    metrics_dict = buffered_metrics[buf_step]
                                    try:
                                        if hasattr(op, "log"):
                                            op.log(metrics_dict, step=buf_step)
                                        streamed_metrics_count += 1
                                        op_log_call_count += 1
                                        any_logged = True
                                    except Exception:
                                        self._logger.debug(
                                            "Failed to stream metrics for step %s",
                                            buf_step,
                                            exc_info=True,
                                        )
                                buffered_metrics.clear()

                    # -------- Collect Files / Text / Hists (log at end) --------
                    other_df = df[
                        df["attribute_type"].isin(
                            [
                                "file",
                                "artifact",
                                "file_series",
                                "string_series",
                                "histogram_series",
                            ]
                        )
                    ]

                    if not other_df.empty:
                        file_related = other_df[
                            other_df["attribute_type"].isin(
                                ["file", "artifact", "file_series"]
                            )
                        ]
                        if not file_related.empty:
                            self._logger.debug(
                                "Processing batch with %d file/artifact/file_series rows",
                                len(file_related),
                            )

                    for _, row in other_df.iterrows():
                        atype = row.get("attribute_type")
                        apath = self._sanitize_attribute_name(row.get("attribute_path"))
                        step = self._convert_step_to_int_optional(
                            row.get("step"), step_multiplier
                        )

                        # Only downsample histogram_series (not file_series or string_series)
                        # We want to preserve all files and all log lines
                        if step is not None and self._log_every > 1:
                            if atype in ("histogram_series",):
                                if (step % self._log_every) != 0:
                                    continue

                        # --- Files (collect only) ---
                        if atype in ("file", "artifact", "file_series"):
                            if atype == "file":
                                file_count += 1
                            elif atype == "artifact":
                                artifact_count += 1
                            elif atype == "file_series":
                                file_series_count += 1

                            if (
                                self._max_files_per_run
                                and all_file_items_queued >= self._max_files_per_run
                            ):
                                self._logger.debug(
                                    "Reached max_files_per_run limit (%d); skipping further files",
                                    self._max_files_per_run,
                                )
                                continue

                            if isinstance(row.get("file_value"), dict):
                                fv = row.get("file_value")
                                file_path = files_directory / fv.get("path", "")

                                # Log file discovery
                                self._logger.debug(
                                    "Processing file: path=%s, attribute=%s, type=%s, step=%s, exists=%s",
                                    fv.get("path", ""),
                                    apath,
                                    atype,
                                    step,
                                    file_path.exists(),
                                )

                                if file_path.exists():
                                    try:
                                        # Detect file types for appropriate preview:
                                        # - Images: use pluto.Image() for inline preview
                                        # - Text: use pluto.Text() for code/log preview
                                        # - Others: use pluto.Artifact() for generic file storage
                                        file_ext = file_path.suffix.lower()
                                        is_image = file_ext in {
                                            ".png",
                                            ".jpg",
                                            ".jpeg",
                                            ".gif",
                                            ".bmp",
                                            ".webp",
                                            ".svg",
                                        }
                                        is_text = file_ext in {
                                            ".txt",
                                            ".log",
                                            ".patch",
                                            ".diff",
                                            ".md",
                                            ".py",
                                            ".json",
                                            ".yaml",
                                            ".yml",
                                            ".csv",
                                        }

                                        if is_image and hasattr(pluto, "Image"):
                                            art = pluto.Image(
                                                str(file_path), caption=apath
                                            )
                                        elif is_text and hasattr(pluto, "Text"):
                                            # Pass file path directly to pluto.Text (don't read into memory)
                                            art = pluto.Text(
                                                str(file_path), caption=apath
                                            )
                                        elif hasattr(pluto, "Artifact"):
                                            art = pluto.Artifact(
                                                str(file_path), caption=apath
                                            )
                                        else:
                                            continue

                                        # Include step in key for file_series only if non-zero (flatten single-step series)
                                        if step is not None and step > 0:
                                            key = f"{apath}/step_{step}"
                                        else:
                                            key = apath
                                        all_files_list.append({key: art})
                                        all_file_items_queued += 1

                                        # Log every 50 files and at regular intervals
                                        if all_file_items_queued % 50 == 0:
                                            self._logger.info(
                                                "Collected %d files so far (current: %s)...",
                                                all_file_items_queued,
                                                apath,
                                            )

                                        now = time.time()
                                        if now - last_heartbeat_ts > 30:
                                            self._logger.info(
                                                "Still collecting files... Total queued: %d",
                                                all_file_items_queued,
                                            )
                                            last_heartbeat_ts = now
                                    except Exception as e:
                                        self._logger.error(
                                            "Failed to create artifact/image for %s: %s",
                                            apath,
                                            e,
                                            exc_info=True,
                                        )
                                else:
                                    if not file_path.exists():
                                        self._logger.warning(
                                            "File does not exist: %s (attribute: %s)",
                                            file_path,
                                            apath,
                                        )

                        # --- String Series (logs - dual route: print + Text artifacts) ---
                        # NOTE: For large datasets (289k+ logs), Logs tab may fail with 502
                        # but Text artifacts in Files tab will still work
                        elif atype == "string_series":
                            string_series_count += 1
                            if pd.notna(row.get("string_value")):
                                text_value = str(row.get("string_value"))
                                # Collect for Text artifacts (Files tab)
                                if (
                                    "stderr" in apath.lower()
                                    or "error" in apath.lower()
                                ):
                                    stderr_lines.append(text_value)
                                    print(text_value, file=sys.stderr, flush=True)
                                else:
                                    stdout_lines.append(text_value)
                                    print(text_value, file=sys.stdout, flush=True)

                        elif atype == "histogram_series":
                            histogram_series_count += 1
                            hv = row.get("histogram_value")
                            if hv is None:
                                continue

                            try:
                                # Arrow scalar -> python
                                if hasattr(hv, "as_py"):
                                    hv = hv.as_py()

                                # Expect pre-binned: [freq, bin_edges]
                                if not (hasattr(hv, "__len__") and len(hv) == 2):
                                    self._logger.warning(
                                        "Skipping histogram %s at step=%s: not a [freq, bin_edges] pair",
                                        apath,
                                        step,
                                    )
                                    continue

                                freq, bin_edges = hv

                                # Normalize inner containers
                                if hasattr(freq, "as_py"):
                                    freq = freq.as_py()
                                if hasattr(bin_edges, "as_py"):
                                    bin_edges = bin_edges.as_py()

                                try:
                                    import numpy as np

                                    if isinstance(freq, np.ndarray):
                                        freq = freq.tolist()
                                    if isinstance(bin_edges, np.ndarray):
                                        bin_edges = bin_edges.tolist()
                                except Exception:
                                    pass

                                freq = [float(x) for x in list(freq)]
                                bin_edges = [float(x) for x in list(bin_edges)]

                                if len(bin_edges) != len(freq) + 1:
                                    self._logger.warning(
                                        "Skipping histogram %s at step=%s: edges(%d) != freq(%d)+1",
                                        apath,
                                        step,
                                        len(bin_edges),
                                        len(freq),
                                    )
                                    continue

                                hist = pluto.Histogram([freq, bin_edges], bins=None)
                                if step not in all_hists:
                                    all_hists[step] = {}
                                all_hists[step][apath] = hist

                            except Exception as e:
                                self._logger.error(
                                    "Failed to log histogram %s at step=%s: %s",
                                    apath,
                                    step,
                                    e,
                                    exc_info=True,
                                )

                    del df
                    del batch_table
                    gc.collect()

            # Loop finished processing all batches
            self._logger.info(
                "Finished processing all %d batches from run_data", batch_count
            )

            # -------- Log all collected data at the end --------

            # Flush any remaining buffered metrics
            if buffered_metrics:
                self._logger.info(
                    "Pluto: flushing final %d metric steps", len(buffered_metrics)
                )
                for buf_step in sorted(
                    [k for k in buffered_metrics.keys() if k is not None]
                ):
                    metrics_dict = buffered_metrics[buf_step]
                    try:
                        if hasattr(op, "log"):
                            op.log(metrics_dict, step=buf_step)
                        streamed_metrics_count += 1
                        op_log_call_count += 1
                        any_logged = True
                    except Exception:
                        self._logger.debug(
                            "Failed to log final metrics batch for step %s",
                            buf_step,
                            exc_info=True,
                        )
                buffered_metrics.clear()

            # Log files in chunks (50-200 at a time) to avoid overload + 502s
            # Each chunk: log, flush, short sleep
            file_chunk_size = self._file_chunk_size
            file_sleep_seconds = self._file_chunk_sleep

            self._logger.info(
                "Starting file upload phase with %d total files, chunk_size=%d, sleep=%fs",
                len(all_files_list),
                file_chunk_size,
                file_sleep_seconds,
            )

            if all_files_list:
                self._logger.info(
                    "Pluto: logging %d files in chunks of %d (sleep=%fs between chunks)",
                    len(all_files_list),
                    file_chunk_size,
                    file_sleep_seconds,
                )

            for i in range(0, len(all_files_list), file_chunk_size):
                chunk = all_files_list[i : i + file_chunk_size]
                if chunk:
                    try:
                        # Merge all artifacts in chunk into one dict
                        files_dict: dict[str, object] = {}
                        for item_dict in chunk:
                            files_dict.update(item_dict)

                        chunk_start = i
                        chunk_end = min(i + len(chunk), len(all_files_list))
                        self._logger.info(
                            "Pluto: uploading file chunk %d/%d (items %d-%d, %d artifacts)...",
                            (i // file_chunk_size) + 1,
                            (len(all_files_list) + file_chunk_size - 1)
                            // file_chunk_size,
                            chunk_start,
                            chunk_end - 1,
                            len(files_dict),
                        )

                        # Log the actual file names being uploaded for this chunk
                        file_names = [key.split("__")[0] for key in files_dict.keys()][
                            :5
                        ]  # Show first 5
                        if len(file_names) < len(files_dict):
                            self._logger.debug(
                                "Uploading files: %s ... and %d more",
                                ", ".join(file_names),
                                len(files_dict) - len(file_names),
                            )
                        else:
                            self._logger.debug(
                                "Uploading files: %s", ", ".join(file_names)
                            )

                        if hasattr(op, "log"):
                            op.log(files_dict)
                        op_log_call_count += 1
                        self._maybe_flush(op)
                        any_logged = True

                        self._logger.info(
                            "File chunk %d/%d uploaded successfully",
                            (i // file_chunk_size) + 1,
                            (len(all_files_list) + file_chunk_size - 1)
                            // file_chunk_size,
                        )

                        if file_sleep_seconds > 0:
                            self._logger.debug(
                                "Sleeping for %.2fs before next chunk...",
                                file_sleep_seconds,
                            )
                            time.sleep(file_sleep_seconds)
                    except Exception as e:
                        self._logger.error(
                            "Failed to log file chunk %d-%d: %s (non-fatal)",
                            chunk_start,
                            chunk_end - 1,
                            e,
                            exc_info=True,
                        )

            # String series already printed to stdout during batch processing
            # Text entries (from logs/*.txt files) are handled as file artifacts, not logged here
            # Nothing to do for texts - they've already been collected and will be reported in summary

            # Log hists by step
            if all_hists:
                self._logger.info("Pluto: logging %d histogram steps", len(all_hists))

            for step, hist_dict in sorted(
                [(k, v) for k, v in all_hists.items() if k is not None],
                key=lambda x: (x[0] is None, x[0]),
            ):
                if not hist_dict:
                    continue
                try:
                    self._logger.info(
                        "Pluto: uploading %d histograms at step=%s",
                        len(hist_dict),
                        step,
                    )
                    if hasattr(op, "log"):
                        op.log(hist_dict, step=step)
                    op_log_call_count += 1
                    any_logged = True
                    self._maybe_flush(op)
                except Exception as e:
                    self._logger.error(
                        "Histogram log failed at step=%s keys=%s: %s",
                        step,
                        list(hist_dict.keys()),
                        e,
                        exc_info=True,
                    )

            self._maybe_flush(op)

            # Log string_series as 2 Text artifacts (stdout.txt and stderr.txt)
            if (stdout_lines or stderr_lines) and hasattr(pluto, "Text"):
                texts_dict: dict[str, object] = {}
                if stdout_lines:
                    stdout_content = "\n".join(stdout_lines)
                    texts_dict["logs/stdout"] = pluto.Text(
                        stdout_content, caption="stdout"
                    )
                if stderr_lines:
                    stderr_content = "\n".join(stderr_lines)
                    texts_dict["logs/stderr"] = pluto.Text(
                        stderr_content, caption="stderr"
                    )

                if texts_dict:
                    self._logger.info(
                        "Pluto: logging %d log text artifacts (stdout/stderr)",
                        len(texts_dict),
                    )
                    try:
                        if hasattr(op, "log"):
                            op.log(texts_dict)
                        op_log_call_count += 1
                        any_logged = True
                    except Exception:
                        self._logger.error(
                            "Failed to log string_series text artifacts", exc_info=True
                        )

            self._logger.info(
                "Pluto run %s COMPLETE: %d op.log() calls, %d metric steps (log_every=%s), "
                "%d file artifacts, %d text entries (string_series), %d histogram steps",
                run_id_str,
                op_log_call_count,
                streamed_metrics_count,
                self._log_every,
                all_file_items_queued,
                string_series_count,
                len(all_hists),
            )

        except KeyboardInterrupt:
            self._logger.warning("Pluto loader interrupted by user (Ctrl+C)")
            self._finish_op(op, run_id_str, interrupted=True)
            # Don't raise - let it exit cleanly so status update completes
            return
        finally:
            # Only finish if not already finished by KeyboardInterrupt handler
            if run_id_str in self._ops:
                self._finish_op(op, run_id_str)

        if not any_logged:
            raise RuntimeError(
                f"Pluto loader finished run {run_id_str} but did not log any data."
            )

        # Mark run as loaded using preserved key, after successful upload
        if loaded_key:
            self._mark_run_loaded(loaded_key)

        self._logger.info("Uploaded data for Pluto Op run %s", run_id_str)

    def _finish_op(
        self, op: object, run_id_str: str, interrupted: bool = False
    ) -> None:
        """Finish/close a Pluto Op to release threads and flush pending uploads."""
        try:
            if hasattr(op, "finish"):
                # If interrupted (Ctrl+C), mark run as canceled (SIGINT=2)
                op.finish(code=signal.SIGINT if interrupted else None)
            elif hasattr(op, "close"):
                op.close()
        except Exception:
            self._logger.debug(
                "Failed to finish Pluto Op %s", run_id_str, exc_info=True
            )
        finally:
            self._ops.pop(run_id_str, None)
            self._run_id_to_key.pop(run_id_str, None)
            gc.collect()
