# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BookGap is a research tool that identifies knowledge gaps in academic literature using embedding-based geometric analysis, then generates AI-powered textbook proposals to fill those gaps. It uses OpenAlex as its data source and OpenAI's Batch API for proposal generation.

## Environment Setup

All commands must be run inside the local conda environment:

```bash
conda run -p /home/gchurchill/src/bookgap/.conda <command>
```

Required environment variables: `OPENAI_API_KEY`, `OPENALEX_API_KEY` (loaded from `.env`).

## Common Commands

```bash
# Run unit tests (excludes integration tests by default)
conda run -p .conda pytest tests/

# Run a single test
conda run -p .conda pytest tests/path/to/test_file.py::TestClass::test_method

# Run integration tests (makes real API calls, costs money)
conda run -p .conda pytest tests/ -m integration

# Type checking
conda run -p .conda python -m mypy bookgap/ tests/ --exclude 'bookgap/batch_utils_OLD' --ignore-missing-imports

# Run the main pipeline
conda run -p .conda python batch_generator.py
conda run -p .conda python batch_generator.py --config-name standard batch.num_gaps=5
```

## Architecture

### Pipeline Flow

`batch_generator.py` (entry point) -> Hydra config resolution -> `Cloud` construction (data loading + embedding + denoising) -> gap detection per topic -> LLM proposal generation via OpenAI Batch API -> artifact export

### Core Data Model

- **`Cloud`** (`cloud.py`): The central data object. A frozen dataclass holding a `FilterPair`, index, deduplicator, denoiser config, and embedder model. Provides cached access to works, embeddings, and citation graphs from OpenAlex. Uses `joblib.Memory` for disk caching under `cache/`.
- **`SubCloud`** (`subcloud.py`): A lightweight filtered view over a parent `Cloud` (e.g., just books or just articles for a given topic). Does not copy data; indexes into the parent.
- **`WorksStore`** (`work_store.py`): Columnar storage layer using PyArrow/Parquet for works and embeddings.

### Gap Detection

Two algorithms selectable via `CellType` enum (`types.py`):
- **K-ball** (`llm/gap_algorithms/kball_algorithm.py`): Finds gaps using radius-based nearest-neighbor search
- **Voronoi** (`llm/gap_algorithms/voronoi_algorithm.py`): Uses Voronoi tessellation

Gap computation is cached per-topic via `@MEMORY.cache` in `gaps.py`.

### Embedding & Denoising

- **`Embedder`** (`embedder.py`): Wraps `sentence-transformers` models (GIST-small, Gemma-300M). Manages GPU memory via context manager.
- **Denoisers** (`denoisers.py`, `cloud_denoiser.py`): Dimensionality reduction/denoising of embeddings (PCA, Anti-LDA, RNOJD). The `ProjectiveDenoiserProjection` caches the projection matrix and is reused for proposal embeddings to maintain vector space consistency.

### LLM Proposal Generation

- **`protocols.py`** (`llm/protocols.py`): Pydantic models (`Proposal`, `Chapter`, `ParsedProposal`) and the `GapProtocol` interface
- **`workflow_functions.py`** (`llm/generators/workflow_functions.py`): Orchestrates the batch pipeline: gap computation -> context building -> OpenAI Batch API submission -> result parsing -> artifact generation
- **`BatchProposalGenerator`** (`llm/generators/batch_generator.py`): Manages OpenAI Batch API lifecycle
- **`run_state.py`** (`llm/generators/run_state.py`): Resumable run state management

### Configuration

Hydra structured configs using dataclasses:
- **`ExperimentConfig`** (`configs/config.py`): Top-level config composing `GapConfig`, `LLMConfig`, `BatchConfig`, `OutputConfig`, `CloudConfig`
- **`config_store.py`** (`configs/config_store.py`): Registers presets (e.g., `standard`)
- Override via CLI: `python batch_generator.py gap.radius_fraction=0.3 llm.temperature=0.35`

### Data Ingestion

- **`openalex_ingest.py`**: Fetches and embeds works from OpenAlex API, resolving hierarchical scopes (domain -> field -> subfield -> topic)
- **`openalex_scopes.py`**: Navigates OpenAlex taxonomy hierarchy with `resolve_filter_upward`

### Utilities

- `util.py`: Global singletons — `LOGGER`, `TIMER`, `MEMORY` (joblib cache), `ARTIFACTS_DIR`, `CACHE_DIR`. Also defines `Filter`, `FilterPair`, and the OpenAlex taxonomy enums (`Domain`, `Field`, `SubField`, `WorkType`).
- `artifacts_scripts/`: Post-hoc analysis scripts (LaTeX table generation, sweeps, curriculum analysis, LLM evaluation)

### Key Design Patterns

- Heavy use of `@MEMORY.cache` (joblib) and `@lru_cache` for expensive computations (embeddings, gap detection, OpenAlex fetches)
- Frozen dataclasses throughout for hashability (required by joblib caching)
- `FilterPair` splits cloud into two filters (e.g., books vs articles) that share the same embedding space
- Denoiser projection is computed once and cached on `Cloud`, then reused for proposal embeddings via `Cloud.denoiser_projection()`
