# Development Guide

Notes for contributors extending the exporter (new loaders/targets, schema tweaks, etc.). User-facing usage lives in `README.md`.

## Local environment

- Install deps with `uv sync --dev`.
- Run checks with `uv run pre-commit run --all-files`.
- Add pre-commit to git hooks with `uv run pre-commit install`.
- Run tests with `uv run pytest -v`. Integration-style tests need:
  - `NEPTUNE2_E2E_API_TOKEN`, `NEPTUNE2_E2E_PROJECT`
  - `NEPTUNE3_E2E_API_TOKEN`, `NEPTUNE3_E2E_PROJECT`

## Code structure (src/neptune_exporter)

- `main.py`: Click CLI wiring for `export`, `load`, `summary`.
- `exporters/`: `Neptune2Exporter` (neptune-client) and `Neptune3Exporter` (neptune-query); both yield `pyarrow.RecordBatch` objects matching `model.SCHEMA`.
- `export_manager.py`: Orchestrates export per project/run, fans out batches per run, and skips runs already on disk.
- `storage/`: `ParquetWriter` (streaming parts per run, temp file cleanup) and `ParquetReader` (per-project/run streaming, metadata extraction).
- `loaders/`: Common `DataLoader` interface plus `MLflowLoader` and `WandBLoader` implementations.
- `loader_manager.py`: Topologically sorts runs (parents before forks), resumes runs if the target already has them, and streams parts to loaders.
- `summary_manager.py` & `validation/report_formatter.py`: Lightweight data introspection/printing for already-exported parquet.
- `model.py`: Central PyArrow schema.
- `utils.py`: Shared helpers (`sanitize_path_part` adds a digest to keep paths safe/unique).

## Data flow overview

1. Export (primary): exporter → `ExportManager` → `ParquetWriter` (+ file downloads). A run is considered complete when `*_part_0.parquet` exists; runs without it are rewritable.
2. Summary: `ParquetReader` → `SummaryManager` → `ReportFormatter`.
3. Load (optional): `ParquetReader` → `LoaderManager` → selected `DataLoader`.

Exports are resumable but not incremental: reruns skip completed runs, so new data added to an already-exported run will be missed unless you re-export to a fresh location.

## Adding or changing components

- **New loader** (e.g., another tracking backend):
  - Implement `DataLoader` methods (`create_experiment`, `find_run`, `create_run`, `upload_run_data`).
  - Handle attribute name sanitization and step conversion internally; `loader_manager` provides `step_multiplier` (keep it consistent when Neptune steps are floats).
  - Extend CLI choices in `main.py` and plumb target-specific options.
- **Schema changes**:
  - Update `model.SCHEMA`.
  - Ensure exporters populate the new columns and loaders ignore/handle them gracefully.
  - Add coverage in tests and, if necessary, bump parquet reader/writer logic.
- **Exporter tweaks**:
  - Keep outputs as PyArrow tables matching `model.SCHEMA`.
  - Continue batching to avoid large in-memory frames; follow the `download_*` generator pattern.
- **File handling**:
  - Artifacts are stored under `--files-path/<sanitized_project_id>/...`; keep the relative paths in `file_value.path` stable so loaders can find the payloads.
- **Forking**:
  - Fork metadata exists only in Neptune 3.x exports. W&B supports forks only in a limited/preview fashion — avoid relying on strict fidelity. MLflow does not support forking and saves parents as tags instead.

## Testing notes

- Prefer function-style pytest tests (no classes) and `unittest.mock.Mock` for doubles.
- Look at `tests/test_storage.py` and `tests/test_summary_manager.py` for patterns.
- When adding loader/exporter behavior, add small, focused tests around boundary cases (empty batches, missing metadata, bad attribute names).

## CI

GitHub Actions runs linting (ruff, mypy, license headers) and tests on Python 3.13 using uv. Workflows live in `.github/workflows/ci.yml`.
