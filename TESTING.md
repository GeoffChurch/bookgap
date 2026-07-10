# Testing Guide

## Running Tests

### Default (Safe) Tests
```bash
uv run pytest tests/
```
**Result**: Runs all tests except integration tests that incur API costs. This is the default behavior.

### Integration Tests Only  
```bash
uv run pytest tests/ -m integration
```
**Result**: Runs only the integration tests that make real API calls (requires `OPENAI_API_KEY`).

### All Tests Including Integration
```bash
uv run pytest tests/ -m ""
```
**Result**: Runs all tests including integration tests that may incur costs.

## Test Categories

- **Unit Tests**: Fast, no external dependencies
- **Integration Tests**: Marked with `@pytest.mark.integration`, make real API calls

### Focused Checks

- **Projection Reuse Sanity Test**: Ensures proposal embeddings reuse the cached denoiser projection and stay normalized.

```bash
uv run pytest tests/llm/test_workflow_functions.py::TestMonitorAndDownloadResults::test_projection_applied_via_monitoring
```

- **Embedding Projection Unit Test**: Exercises `_compute_proposal_embeddings` directly with a mocked embedder.

```bash
uv run pytest tests/llm/test_workflow_functions.py::TestComputeProposalEmbeddings::test_projection_and_normalization_applied
```

## Environment Variables

- `OPENAI_API_KEY`: Required for integration tests that make real OpenAI API calls

## Cost Management

Integration tests are **skipped by default** to avoid unexpected API charges. They must be explicitly run using the `-m integration` flag.

## Type Checking

BookGap uses mypy for static type checking:

```bash
# Check all Python files
uv run python -m mypy .

# Check specific modules
uv run python -m mypy bookgap/

# Check with missing import stubs ignored
uv run python -m mypy bookgap/ --ignore-missing-imports

# Check bookgap/ and tests/ while excluding OLD directories
uv run python -m mypy bookgap/ tests/ --exclude 'bookgap/batch_utils_OLD' --ignore-missing-imports

# Check specific files
uv run python -m mypy bookgap/config_manager.py bookgap/llm/llm_implementation.py
```

**Notes**: 
- Prefix commands with: `uv run` (env defined by `pyproject.toml` + `uv.lock`).
- Some mypy warnings related to third-party libraries (faiss, sklearn, tqdm) are expected and can be ignored.
