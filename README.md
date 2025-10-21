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

```bash
# Export all data from a project
neptune-exporter -p "my-org/my-project"

# Export only parameters and metrics from specific runs
neptune-exporter -p "my-org/my-project" -r "RUN-*" -e parameters -e metrics

# Export specific attributes only (exact match)
neptune-exporter -p "my-org/my-project" -a "learning_rate" -a "batch_size"

# Export attributes matching a pattern (regex)
neptune-exporter -p "my-org/my-project" -a "config/.*"

# Use Neptune 2.x exporter
neptune-exporter -p "my-org/my-project" --exporter neptune2

# Export to custom output directory
neptune-exporter -p "my-org/my-project" -o "/path/to/output"

# Export multiple projects
neptune-exporter -p "my-org/project1" -p "my-org/project2"
```

### Command Options

- `-p, --project-ids`: Neptune project IDs to export (required, can be specified multiple times)
- `-r, --runs`: Filter runs by pattern (e.g., 'RUN-*' or specific run ID)
- `-a, --attributes`: Filter attributes by name (can be specified multiple times)
- `-e, --export-classes`: Types of data to export (parameters, metrics, series, files)
- `--exporter`: Neptune exporter to use (neptune2, neptune3)
- `-o, --output-path`: Base path for exported data (default: ./exports)
- `--api-token`: Neptune API token (or use NEPTUNE_API_TOKEN environment variable)
