# Neptune Exporter

Migration tool to help Neptune customers transition their data out of Neptune in case of acquisition.

## Development

### Setup

```bash
# Install dependencies
uv sync --dev

# Run linting and formatting
uv run pre-commit run --all-files

# Run tests
# Note: Integration tests require environment variables:
# - NEPTUNE2_E2E_API_TOKEN: Neptune 2.x API token for authentication
# - NEPTUNE2_E2E_PROJECT: Neptune 2.x project identifier for testing
# - NEPTUNE3_E2E_API_TOKEN: Neptune 3.x API token for authentication
# - NEPTUNE3_E2E_PROJECT: Neptune 3.x project identifier for testing
uv run pytest tests/ -v
```

### CI/CD

This project uses GitHub Actions for continuous integration. The CI workflow runs on every push and pull request and includes:

- **Pre-commit hooks**: Code formatting (ruff), linting (ruff), type checking (mypy), and license insertion
- **Tests**: Runs pytest on the test suite with coverage reporting
- **Test Reports**: Generates JUnit XML test reports with GitHub Actions integration
- **Python version**: Tests against Python 3.13

The workflow is defined in `.github/workflows/ci.yml` and uses `uv` for dependency management.

### Pre-commit

This project uses pre-commit hooks to ensure code quality. The hooks include:

- **ruff**: Code formatting and linting
- **mypy**: Type checking
- **license**: Automatic license header insertion

To install pre-commit hooks:

```bash
uv run pre-commit install
```

## Usage

The Neptune Exporter provides a complete migration pipeline from Neptune to other platforms like MLflow.

### Export Data from Neptune

Export Neptune experiment data to parquet files:

```bash
# Export all data from a project
neptune-exporter export -p "my-org/my-project"

# Export only parameters and metrics from specific runs
neptune-exporter export -p "my-org/my-project" -r "RUN-*" --exclude files

# Export specific attributes only (exact match)
neptune-exporter export -p "my-org/my-project" -a "learning_rate" -a "batch_size"

# Export attributes matching a pattern (regex)
neptune-exporter export -p "my-org/my-project" -a "config/.*"

# Use Neptune 2.x exporter
neptune-exporter export -p "my-org/my-project" --exporter neptune2

# Export with custom paths
neptune-exporter export -p "my-org/my-project" --data-path ./my-data --files-path ./my-files

# Overwrite existing data (re-export all runs)
neptune-exporter export -p "my-org/my-project" --overwrite
```

#### Resume and Overwrite Behavior

By default, the exporter will **resume** interrupted exports by skipping runs that have already been exported. This is useful when:

- An export was interrupted and you want to continue from where it left off
- You want to add new runs to an existing export
- You want to avoid re-downloading large amounts of data

Use the `--overwrite` flag to force re-export of all runs, which will:

- Delete existing project data before starting the export
- Re-export all runs from scratch
- Ensure no mixing of old and new data

**Note:** Resume works at the run level - if a run was partially exported, it will be skipped entirely and not resumed.

### Load Data to MLflow

Load exported parquet data to MLflow:

```bash
# Load all data from exported parquet files
neptune-exporter load

# Load specific projects
neptune-exporter load -p "my-org/my-project1" -p "my-org/my-project2"

# Load specific runs
neptune-exporter load -r "RUN-123" -r "RUN-456"

# Load with custom paths
neptune-exporter load --data-path ./my-data --files-path ./my-files

# Load to specific MLflow tracking URI
neptune-exporter load --mlflow-tracking-uri "http://localhost:5000"
```

### View Data Summary

Get a summary of exported data:

```bash
# Show summary of all exported data
neptune-exporter summary

# Show summary from custom data path
neptune-exporter summary --data-path ./my-data
```

### Complete Migration Example

```bash
# Step 1: Export data from Neptune
neptune-exporter export -p "my-org/my-project"

# Step 2: View what was exported
neptune-exporter summary

# Step 3: Load to MLflow
neptune-exporter load --mlflow-tracking-uri "http://localhost:5000"
```


## Data Type Mappings

### Neptune to MLflow

| Neptune Type | MLflow Type | Notes |
|--------------|-------------|-------|
| Parameters (float, int, string, bool, datetime, string_set) | Parameters | All values converted to strings |
| Float Series | Metrics | Steps converted from decimal to integer |
| String Series | Artifacts | Saved as text files |
| Histogram Series | Artifacts | Saved as CSV tables |
| Files | Artifacts | Direct file upload |
| File Series | Artifacts | Files with step information in path |

### Key Considerations

- **Step Conversion**: Neptune uses decimal steps, MLflow uses integer steps. Default multiplier is 1,000,000 (configurable with `--step-multiplier`)
- **Attribute Names**: Neptune attribute paths are sanitized to meet MLflow key constraints (max 250 chars, alphanumeric + special chars)
- **Experiment Names**: Neptune project IDs become MLflow experiment names (with optional prefix)
- **File Artifacts**: Files are uploaded as MLflow artifacts with preserved directory structure
