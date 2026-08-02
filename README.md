# BookGap

Research tool for finding knowledge gaps in academic literature and generating
AI textbook proposals to fill them. It analyzes embeddings of articles and
books (OpenAlex) to identify underexplored regions via geometric algorithms
(K-ball or Voronoi tessellation), samples the surrounding works as context,
generates structured textbook proposals with LLMs, and produces plots and
statistics about the gaps and proposals.

> **Reproducing the paper?** See **[REPRODUCING.md](REPRODUCING.md)**. It covers
> the (narrow) path through this repo that produces the published results, what
> it costs to run, and — importantly — the two things that *cannot* be
> reproduced from this repository. Most of what is here is not the paper.

## Ecosystem

This repo is one of three related projects:

- **bookgap** (this repo) — gap detection, proposal generation, and the
  post-hoc research scripts under `artifacts_scripts/`.
- **[curriculum](https://github.com/GeoffChurch/curriculum)** *(private)* — the
  standalone curriculum engine and evaluation layer, extracted from this repo on
  2026-07-10. Not required for gap detection or proposal generation; see Setup.
- **[learn-and-teach](https://github.com/GeoffChurch/learn-and-teach)** *(private)* — the
  judgment-first textbook *generation* successor (prose-orchestrated book
  builds). This repo remains the home of gap detection and the algorithmic
  research corpus.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). The environment
is fully specified by `pyproject.toml` + `uv.lock`.

```bash
git clone git@github.com:GeoffChurch/bookgap.git
cd bookgap
uv sync --extra dev
```

**The sibling `curriculum` checkout is no longer needed for gap detection or
proposal generation** — the one function `bookgap/` used from it is vendored in
`bookgap/_bisecting_kmeans.py`, with provenance recorded there. Parts of
`artifacts_scripts/` (the wiki, lexrank and keyword threads) and four test modules
still import it; those are side research threads, not the paper's pipeline.
`curriculum` is currently a private repository, so if you hit an `ImportError`
from one of those, that is why (the tests skip themselves instead). If you do have the sibling
checkout and its `pyproject.toml` changes, re-run `uv sync` to re-lock.

Environment variables, all read from `.env`:

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | proposal generation via the OpenAI Batch backend; the local-LLM paths use Ollama instead |
| `OPENALEX_API_KEY` | OpenAlex requests with a per-key budget |
| `OPENALEX_EMAIL` | OpenAlex's [polite pool](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication) — faster and more reliable in exchange for being identifiable. Optional; unset just means the common pool |
| `CONTACT_EMAIL` | the contact in the User-Agent sent to Wikipedia, whose [policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy) asks for one. Only the `artifacts_scripts/` Wikipedia threads use it |
| `QUALTRICS_BRAND_ID`, `QUALTRICS_OWNER_ID` | the Qualtrics tenant and account that a generated `.qsf` imports into. Only `artifacts_scripts/survey/generate_survey.py` uses them; unset produces a `.qsf` that imports nowhere |
| `MATHLIB4_REPO`, `BAZELISK` | a mathlib4 checkout and the `bazelisk` binary, for the code-gap adapters under `artifacts_scripts/gap_recover/adapters/`. `BAZELISK` falls back to `PATH` |

None of these have defaults baked into the source. That is deliberate: a hardcoded
contact address identifies whoever wrote it, which is a problem for anonymous
review and for publishing the repo.

## Usage

The proposal-generation entry point is Hydra-configured and resumable:

```bash
uv run python bookgap/batch_generator.py                          # default preset
uv run python bookgap/batch_generator.py --config-name standard batch.num_gaps=5
uv run python bookgap/batch_generator.py llm.temperature=0.35 gap.cell_type=VORONOI
uv run python bookgap/batch_generator.py --resume-output-dir outputs/<date>/<time>
```

Key config groups — `gap.*` (detection algorithm, radius, preprocessing),
`llm.*` (model, implementation, context size), `batch.*` (gaps per topic,
output paths), `data.*` (subclouds, filtering). Presets are registered in
`bookgap/configs/config_store.py`; compose them programmatically via:

```python
from hydra import initialize, compose
from bookgap.configs.config_store import register_configs

register_configs()
with initialize(config_path=None, version_base=None):
    cfg = compose(config_name="standard", overrides=["batch.num_gaps=5"])
```

The OpenAlex scope/topic manager ships as a console script:

```bash
uv run scope-manager --help
```

Long-running jobs go through `bookgap.jobs` (systemd-run cgroups + JSON
manifests under `cache/.jobs/`) — conventions in `CLAUDE.md`.

## Generated files

Outputs land in a structured directory under `artifacts/`:

```
artifacts/
└── subfield=1702, work_types=[book,article]/
    ├── gaps_analysis_results.csv           # main results with gap statistics
    ├── openai_structured_proposals/        # LLM-generated proposals (gap_N_proposal.json)
    ├── plots/                              # PCA plots per gap, gap-size histograms/QQ plots
    └── contexts/                           # context data per gap (gap_N_context.json)
```

Every generated artifact embeds the exact command that produced it (see the
provenance convention in `CLAUDE.md`).

## Development

### Testing

```bash
uv run pytest                  # unit tests; integration tests deselected by default
uv run pytest -m integration   # may incur API costs
```

See `TESTING.md`. Note: pytest's `addopts` ignore the (huge, gitignored)
`cache/` directory during collection — don't remove that flag.

### Architecture

- `bookgap/cloud.py` — the central `Cloud` frozen dataclass: works,
  embeddings, and citation graph, Parquet-backed. The denoiser's projection
  matrix is cached once and reapplied to proposal embeddings
  (`Cloud.denoiser_projection()`), keeping proposals in the same vector space
  as the cloud analysis.
- `bookgap/llm/gap_algorithms/` — K-ball and Voronoi gap detection.
- `bookgap/llm/generators/` — the OpenAI Batch API proposal workflow
  (`workflow_functions.py`); `bookgap/llm/protocols.py` holds the static
  Pydantic models (`Chapter`, `Proposal`, `ParsedProposal`).
- `bookgap/denoisers.py`, `bookgap/embedder.py` — embedding and denoising
  (PCA, Anti-LDA, RNOJD).
- `bookgap/configs/` — Hydra structured configs (`ExperimentConfig` +
  component groups, presets in `config_store.py`).
- `artifacts_scripts/` — post-hoc research/eval/sweep scripts that produce
  `artifacts/` (gap-detection evaluation harness, denoiser studies, paper
  assets, the legacy curriculum pipeline slated for replacement by the
  curriculum repo's generate app).

### Conventions

Idea-state discipline (`docs/backlog/`, `docs/dead_ends/`, each with an
index), artifact provenance, and job management are documented in `CLAUDE.md`.
