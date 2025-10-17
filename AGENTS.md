# Agent Instructions for Neptune Migration Project

Guidelines for LLM agents working on this project.

## Project Overview

Exports Neptune data to parquet files, then loads to target platforms (MLflow). Architecture: exporters → storage → loaders → validation.

## Key Conventions

### Code Style
- Use **uv** for Python management (`uv run` not `python`)
- **Use pytest with simple functions, NOT test classes**
- Follow patterns in `test_storage.py` and `test_summary_manager.py`
- Use `unittest.mock.Mock` for mocking

### Data Handling
- **Always use PyArrow Tables for internal processing** (not pandas)
- Convert to pandas only for external APIs
- Use actual `project_id` from parquet data, not sanitized directory names
- Follow data type mapping in `README.md`

### Architecture
- **Storage**: `ParquetWriter`/`ParquetReader` for file operations
- **Export**: `ExportManager` + `NeptuneExporter` (v2/v3)
- **Load**: `LoaderManager` + specific loaders (`MLflowLoader`)
- **Validation**: `SummaryManager` + `ReportFormatter`

## Common Patterns

### Error Handling
```python
try:
    # operation
    return result
except Exception as e:
    self._logger.error(f"Context: {e}")
    return None  # or empty structure
```

### Testing
```python
def test_function_name():
    """Test description."""
    mock_obj = Mock(spec=SomeClass)
    mock_obj.method.return_value = expected_value
    
    result = function_under_test(mock_obj)
    
    assert result == expected_result
    mock_obj.method.assert_called_once_with(expected_args)
```

## Common Pitfalls

- **Import errors**: Update all imports when moving files
- **Data type mixing**: Keep PyArrow internal, pandas external
- **Project ID confusion**: Use actual IDs from data, not directory names

## File Operations

- Use `pathlib.Path` for file operations
- Handle missing directories gracefully
- Update all imports when refactoring
- Use absolute paths when possible

## Quality Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] No linting errors (`uv run pre-commit run -a`)
- [ ] Uses PyArrow for internal processing
- [ ] Handles errors gracefully
- [ ] Updates imports if files moved
- [ ] Follows existing patterns

## Getting Started

1. Read `README.md` for overview
2. Check `model.py` for data schema
3. Look at existing tests for patterns
4. Follow data flow: export → storage → load → summary

**Remember**: Clean, maintainable code with clear separation of concerns. Follow existing patterns.
