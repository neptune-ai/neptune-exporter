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
# Export data from Neptune to Parquet
neptune-exporter
```
