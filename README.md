# Neptune Exporter

CLI tool to move Neptune experiments (version 2.x or 3.x) to disk as parquet + files, with an option to load them into MLflow or Weights & Biases.

## What it does
- Streams runs from Neptune to local storage; artifacts are downloaded alongside the parquet.
- Skips runs that were already exported (presence of `part_0.parquet`), making exports resumable.
- Loads parquet data into MLflow or W&B while preserving run structure (forks, steps, attributes) as closely as possible.
- Prints a human-readable summary of what is on disk.

## Requirements
- Python 3.13 (managed via [uv](https://github.com/astral-sh/uv)).
- Neptune credentials: `NEPTUNE_API_TOKEN` (or pass `--api-token`) and either `--project-ids` or `NEPTUNE_PROJECT`.
- Target credentials when loading:
  - MLflow: `MLFLOW_TRACKING_URI` or `--mlflow-tracking-uri`.
  - W&B: `WANDB_ENTITY`/`--wandb-entity` and `WANDB_API_KEY`/`--wandb-api-key`.

Install dependencies in the repo:

```bash
uv sync
```

Run all CLI commands via uv:

```bash
uv run neptune-exporter --help
```

## Quick start
1) Export Neptune data to disk (core):
```bash
uv run neptune-exporter export \
  -p "my-org/my-project" \
  --exporter neptune3 \
  --data-path ./exports/data \
  --files-path ./exports/files
```
Notable options:
- `--exporter` (required): `neptune3` or `neptune2` (match your workspace version).
- `-r/--runs`: Neptune run ID filter, regex supported. In Neptune 3.x run id is user-chosen or auto-generated, stored in `sys/custom_run_id`, in Neptune 2.x it's auto-generated `sys/id`, e.g. `SAN-1`.
- `-a/--attributes`: One value = regex, multiple values = exact attribute names.
- `-c/--classes` and `--exclude`: Include/exclude `parameters`, `metrics`, `series`, `files`.
- `--include-archived-runs`: Include archived/trashed runs.
- `--api-token`: Pass the token explicitly instead of `NEPTUNE_API_TOKEN`.
- `--no-progress`, `-v/--verbose`, `--log-file`: Progress/logging controls for the CLI.

### Export examples
- Export everything from a project:  
  `uv run neptune-exporter export -p "workspace/proj" --exporter neptune3`
- Export only parameters/metrics from runs matching a pattern:  
  `uv run neptune-exporter export -p "workspace/proj" --exporter neptune3 -r "RUN-.*" -c parameters -c metrics`
- Export specific attributes by pattern:  
  `uv run neptune-exporter export -p "workspace/proj" --exporter neptune3 -a "metrics/accuracy" -a "metrics/loss" -a "config/.*"`
- Export with Neptune 2.x client and split data/files to different locations:  
  `uv run neptune-exporter export -p "workspace/proj" --exporter neptune2 --data-path /mnt/fast/exports/data --files-path /mnt/cold/exports/files`

2) Inspect what was exported:
```bash
uv run neptune-exporter summary --data-path ./exports/data
```

3) Load into a target (optional):
```bash
# MLflow
uv run neptune-exporter load \
  --loader mlflow \
  --mlflow-tracking-uri "http://localhost:5000" \
  --data-path ./exports/data \
  --files-path ./exports/files

# W&B
uv run neptune-exporter load \
  --loader wandb \
  --wandb-entity my-org \
  --wandb-api-key "$WANDB_API_KEY" \
  --data-path ./exports/data \
  --files-path ./exports/files
```
`--step-multiplier` converts Neptune’s decimal steps into integers for MLflow/W&B (which only accept ints). If your Neptune steps contain decimals, pick a single multiplier (e.g., `1000` for millisteps) and use it consistently for all loads so every series stays aligned.
Default is `1` (no scaling).

## Data layout on disk
- Parquet path (default `./exports/data`):
  - One directory per project, sanitized for filesystem safety (digest suffix added) but the parquet columns keep the real `project_id`/`run_id`.
  - Each run is split into `run_id_part_<n>.parquet` (Snappy-compressed); parts roll over around 50 MB compressed.
- Files path (default `./exports/files`):
  - Mirrors the sanitized project directory; file artifacts and file series are saved relative to that root.
  - Kept separate from the parquet path so you can place potentially large artifacts on different/cheaper storage.

### Parquet schema
All records use `src/neptune_exporter/model.py::SCHEMA`:

| Column | Type | Description |
| --- | --- | --- |
| `project_id` | string | Neptune project identifier (e.g., `workspace-name/project-name`) |
| `run_id` | string | Neptune run identifier (Neptune 3.x: user-chosen or auto-generated, stored in `sys/custom_run_id`. Neptune 2.x: auto-generated `sys/id`, e.g. `SAN-1`) |
| `attribute_path` | string | Full attribute path (e.g., `metrics/accuracy`, `metrics/loss`, `files/dataset_desc.json`) |
| `attribute_type` | string | One of: `float`, `int`, `string`, `bool`, `datetime`, `string_set`, `float_series`, `string_series`, `histogram_series`, `file`, `file_series` |
| `step` | decimal(18,6) | Decimal step value (per series/metric/file series) |
| `timestamp` | timestamp(ms, UTC) | Populated for time-based records (metrics/series/file series); null for parameters/files |
| `int_value`/`float_value`/`string_value`/`bool_value`/`datetime_value`/`string_set_value` | typed columns | Payload depending on `attribute_type` |
| `file_value` | struct{path} | Relative path to downloaded file payload |
| `histogram_value` | struct{type,edges,values} | Histogram payload |

## Export flow
- Runs are listed per project and streamed in batches; already-exported runs (those with `part_0.parquet`) are skipped so reruns are resumable. Use this with care: if a run was exported and later received new data in Neptune, that new data will not be picked up unless you re-export to a fresh location.
- Data is written per run into parquet parts (~50 MB compressed per part), keeping memory usage low.
- Artifacts and file series are downloaded alongside parquet under `--files-path/<sanitized_project_id>/...`.
- A run is considered complete once `part_0.parquet` exists; use a fresh `--data-path` if you need a clean re-export.

## Loading flow
- Data is streamed run-by-run from parquet, using the same `--step-multiplier` to turn decimal steps into integers. Keep the multiplier consistent across loads when your Neptune steps are floats.
- MLflow loader:
  - Experiments are named `<project_id>/<experiment_name>` (prefixed with `--name-prefix` if provided).
  - Attribute paths are sanitized to MLflow’s key rules (alphanumeric + `_-. /`, max 250 chars).
  - Metrics/series use the integer step; files are uploaded as artifacts from `--files-path`.
  - MLflow saves parentship/fork relationships as tags (no native forks).
- W&B loader:
  - Requires `--wandb-entity`; project names derive from `project_id` (plus optional `--name-prefix`, sanitized).
  - String series become W&B Tables, histograms use `wandb.Histogram`, files/file series become artifacts. Forked runs from Neptune 3.x are handled best-effort (W&B has limited preview support).
- If a target run with the same name already exists in the experiment/project, the loader skips uploading that run to avoid duplicates.

## Experiment/run mapping to targets
- MLflow:
  - Each unique `project_id` + `sys/name` pair becomes an MLflow experiment named `<project_id>/<sys/name>` (prefixed by `--name-prefix` if provided).
  - Runs are created inside that experiment using Neptune `run_id` (or `custom_run_id` when present) as the run name. Fork relationships are ignored by MLflow.
- W&B:
  - Neptune `project_id` maps to the W&B project name (sanitized, plus optional `--name-prefix`).
  - `sys/name` becomes the W&B group, so all runs with the same `sys/name` land in the same group.
  - Runs are created with their Neptune `run_id` (or `custom_run_id`) as the run name. Forks from Neptune 3.x are mapped best-effort via `fork_from`; behavior depends on W&B’s fork support.

## Attribute/type mapping (detailed)
- Parameters (`float`, `int`, `string`, `bool`, `datetime`, `string_set`):
  - MLflow: logged as params (values stringified by the client).
  - W&B: logged as config with native types (string_set → list).
- Float series (`float_series`):
  - Both targets: logged as metrics using the integer step (`--step-multiplier` applied).
  - Timestamps are forwarded when present.
- String series (`string_series`):
  - MLflow: saved as artifacts (one text file per series).
  - W&B: logged as a Table with columns `step`, `value`, `timestamp`.
- Histogram series (`histogram_series`):
  - MLflow: uploaded as artifacts containing the histogram payload.
  - W&B: logged as `wandb.Histogram`.
- Files (`file`) and file series (`file_series`):
  - Downloaded to `--files-path/<sanitized_project_id>/...` with relative paths stored in `file_value.path`.
  - MLflow/W&B: uploaded as artifacts. File series include the step in the artifact name/path so steps remain distinguishable.
- Attribute names:
  - MLflow: sanitized to allowed chars (alphanumeric + `_-. /`), truncated at 250 chars.
  - W&B: sanitized to allowed pattern (`^[_a-zA-Z][_a-zA-Z0-9]*$`); invalid chars become `_`, and names are forced to start with a letter/underscore.

## Summary command
`uv run neptune-exporter summary` reads parquet files and prints counts of projects/runs, attribute type breakdowns, and basic step stats to help you verify the export before loading.

## License
Apache 2.0. See `LICENSE.txt`.
