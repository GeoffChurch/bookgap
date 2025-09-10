# Testing Guide

## Running Tests

### Default (Safe) Tests
```bash
pytest tests/
```
**Result**: Runs all tests except integration tests that incur API costs. This is the default behavior.

### Integration Tests Only  
```bash
pytest tests/ -m integration
```
**Result**: Runs only the integration tests that make real API calls (requires `OPENAI_API_KEY`).

### All Tests Including Integration
```bash
pytest tests/ -m ""
```
**Result**: Runs all tests including integration tests that may incur costs.

## Test Categories

- **Unit Tests**: Fast, no external dependencies
- **Integration Tests**: Marked with `@pytest.mark.integration`, make real API calls

## Environment Variables

- `OPENAI_API_KEY`: Required for integration tests that make real OpenAI API calls

## Cost Management

Integration tests are **skipped by default** to avoid unexpected API charges. They must be explicitly run using the `-m integration` flag.

## Type Checking

BookGap uses mypy for static type checking:

```bash
# Check all Python files
python -m mypy .

# Check specific modules
python -m mypy nextbook/

# Check with missing import stubs ignored
python -m mypy nextbook/ --ignore-missing-imports

# Check specific files
python -m mypy nextbook/config_manager.py nextbook/llm/llm_implementation.py
```

Note: Some mypy warnings related to third-party libraries (faiss, sklearn, tqdm) are expected and can be ignored.
