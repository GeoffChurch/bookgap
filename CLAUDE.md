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

## Long-Running Jobs

Multi-hour runs (snapshot ingest, full-corpus embedding, curriculum build,
sweeps) go through `bookgap.jobs`, which launches the command in a named
`systemd-run --user --scope` cgroup, writes a JSON manifest at
`cache/.jobs/<job_id>.json`, and redirects stdout/stderr to
`cache/.jobs/<job_id>.log`. The cgroup wrapping is what matters: a plain
`kill <pid>` on a `conda run` wrapper leaves the inner Python process
orphaned, but `systemctl --user stop` tears down the whole tree at once.

```bash
# Launch any command as a tracked job
conda run -p .conda python -m bookgap.jobs run --prefix curriculum_qc -- \
    conda run -p .conda python -m artifacts_scripts.curriculum.main ...

# List jobs (running, complete, failed, crashed, interrupted)
conda run -p .conda python -m bookgap.jobs list

# Gracefully stop (SIGTERM → worker_lifecycle → 'interrupted' status)
conda run -p .conda python -m bookgap.jobs stop <job_id>

# Tail a job's log
conda run -p .conda python -m bookgap.jobs log <job_id> --follow
```

A single CLI invocation is wrapped by `bookgap.jobs._exec`, which in turn
wraps it in `jobs.worker_lifecycle(job_id)` — so normal exit marks the
manifest `complete`, SIGTERM/SIGINT marks it `interrupted`, and any other
exception marks it `failed` with the error message recorded.

**Graceful stop semantics.** Python code inside a `worker_lifecycle` block
can poll `bookgap.jobs.stop_requested()` in long inner loops (e.g.
`_embed_sentence_transformer` in `cloud.py`) to exit at safe boundaries
so partial state on disk stays consistent.

**Where logs actually land.**

- `cache/.jobs/<job_id>.log` — the subprocess's raw stdout/stderr. Note
  that `conda run` buffers stdout, so this file may stay empty until the
  child exits or crashes.
- `outputs/<YYYY-MM-DD>/<HH-MM-SS>/main.log` — Hydra's own log; this is
  the live source of truth for any Hydra-driven pipeline (curriculum,
  batch_generator, etc.). Writes flush directly and bypass `conda run`'s
  stdout buffer.
- `artifacts/curriculum/<run_name>/pipeline.log` — the curriculum
  pipeline's own LOGGER output, mirrored alongside its artifacts.

**TUI.** `scope_manager` (`python scope_manager.py tui`) provides a
Textual-based browser for the OpenAlex taxonomy and surfaces the same
`cache/.jobs/*.json` manifests in a live jobs table — any job launched
via `bookgap.jobs run` or `scope_manager`'s `fetch`/`snapshot`/`embed`
actions shows up there.

**Module layout.**

- `bookgap/jobs.py` — generic systemd-run + manifest plumbing.
- `bookgap/scope_ops.py` — taxonomy-scoped worker actions
  (`fetch`/`snapshot`/`embed`), `launch_worker`, `run_worker`, and
  `has_running_job(action, scope, model)`. Delegates all job-lifecycle
  mechanics to `bookgap.jobs`.
- `scope_manager.py` — Textual TUI + Typer CLI over `scope_ops`; imports
  job-plumbing primitives (`discover_jobs`, `dismiss_finished_jobs`,
  `log_path`) directly from `bookgap.jobs`.

**Stale files.** Root-level `nohup.out`, `fetch.log`, `curriculum.log`,
`survey_llm.log` are residue from earlier pre-`bookgap.jobs` runs and
can be deleted.

## Architecture

### Pipeline Flow

`batch_generator.py` (entry point) -> Hydra config resolution -> `Cloud` construction (data loading + embedding + denoising) -> gap detection per topic -> LLM proposal generation via OpenAI Batch API -> artifact export

### Core Data Model

- **`Cloud`** (`cloud.py`): The central data object. A frozen dataclass holding a `FilterPair`, index, deduplicator, denoiser config, and embedder config. Provides cached access to works, embeddings, and citation graphs. Column-selective reads from per-scope Parquet files avoid loading the full corpus into memory.
- **`SubCloud`** (`subcloud.py`): A lightweight filtered view over a parent `Cloud` (e.g., just books or just articles for a given topic). Does not copy data; indexes into the parent.
- **Per-scope Parquet** (`openalex_ingest.py`): Each scope's works stored as a Parquet file under `cache/works_parquet/`. `corpus_dataset()` returns a lazy `pyarrow.dataset.Dataset` spanning all scopes with column-selective reads.
- **Embeddings store** (`cloud.py`): Dense embeddings stored in a separate Parquet file under `cache/embeddings_parquet/`, keyed by corpus fingerprint + embedder config. Decoupled from works so switching embedders doesn't require re-fetching.

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

- Heavy use of `@MEMORY.cache` (joblib) and `@lru_cache` for expensive computations (gap detection, scope resolution). Per-scope Parquet files replace joblib for works caching; embeddings have their own Parquet store.
- Frozen dataclasses throughout for hashability (required by joblib caching)
- `FilterPair` splits cloud into two filters (e.g., books vs articles) that share the same embedding space
- Denoiser projection is computed once and cached on `Cloud`, then reused for proposal embeddings via `Cloud.denoiser_projection()`

## Backlog and dead_ends

Two places encode the project's idea-state:

- **What might work** — `docs/backlog/` (organized by theme, with an `index.md`). Forward-looking ideas. If something interesting comes up mid-conversation — a new variant to try, a follow-up question, a refactor worth doing later — drop it there without asking first. One file per idea, self-contained enough to act on later without re-reading conversation context. Short is fine.

  A backlog entry may also be a **living/in-progress doc**: when an idea is actively being worked, its file may grow in place from a one-liner into a working doc (add a `## Status / working notes` section, updated as work proceeds) rather than being deleted or moved immediately. The one-line `index.md` entry stays; the entry file itself is allowed to grow. On completion it migrates to its forever home (`docs/<topic>.md` results + code) per the validated/refuted rules below, and the backlog entry is removed/transitioned.

- **What didn't work** — `docs/dead_ends/` (parallel structure to `docs/backlog/`, with its own `index.md`). Refuted ideas with their *diagnosis*. The point is to keep failed-but-instructive directions discoverable so we don't re-run the same experiment, AND so the *reason it failed* is captured (often more useful than the original idea).

When an idea in `docs/backlog/` is executed:
- **Validated** → delete the backlog file; the permanent home is the relevant code (with comments) plus a results doc under `docs/` if substantial. If the feature is promoted, this CLAUDE.md is updated to describe it.
- **Refuted** → move to the corresponding subdirectory under `docs/dead_ends/`, rewrite as a failure-and-diagnosis doc (what was tested, what was expected, what actually happened, why, and any salvage paths), and drop the entry from `docs/backlog/index.md` while adding it to `docs/dead_ends/index.md`. Don't just delete — the diagnosis is the artifact worth preserving.

Don't dump every refuted variant into `docs/dead_ends/` — only the *core ideas that sounded promising*. Saturation tweaks of a winning variant get a paragraph in the parent dead_ends entry, not their own files.

Cross-pollination from `~/src/mycooc/docs/backlog/` is welcome: ideas borrowed from the mycooc project should reference the original mycooc backlog file in their bookgap-side description.
