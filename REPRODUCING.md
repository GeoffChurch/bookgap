# Reproducing the paper

This document covers what is needed to reproduce the results in *Finding and
Filling Gaps in the Literature*. The [README](README.md) describes the project
generally; this file is scoped to the paper and is expected to age with it.

Read the **[What is not reproducible here](#what-is-not-reproducible-here)**
section first. Two parts of the paper cannot be regenerated from this repository,
and it is better to know that before starting.

---

## What the paper's pipeline actually is

The published results come from a narrow path through the codebase:

    bookgap/batch_generator.py          entry point, Hydra-configured
      -> bookgap/cloud.py               Cloud: works, embeddings, citation graph
      -> bookgap/denoisers.py           LPP projection (the paper's default)
      -> bookgap/gaps.py                gap detection, cached per topic
         bookgap/llm/gap_algorithms/    K-ball (default) and Voronoi
      -> bookgap/llm/generators/        proposal generation
         workflow_functions.py          orchestration
         ollama_local.py                the paper used gemma4:31b via Ollama

Most of the repository is *not* this. `artifacts_scripts/` holds around a dozen
independent research threads, none of which is needed to reproduce the paper. See
`docs/backlog/repo-architecture.md` for the full map.

## Prerequisites

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/)
- An OpenAlex API key (`OPENALEX_API_KEY`) for corpus ingest
- [Ollama](https://ollama.com/) with `gemma4:31b` pulled, for proposal generation
- `OPENAI_API_KEY` only if you use the OpenAI Batch backend instead

```sh
uv sync --extra dev
```

Note the disk cost: a full ingest of the AI subfield is on the order of
hundreds of gigabytes under `cache/`, and embedding it is GPU-hours. Neither the
corpus nor the embeddings are distributed with this repository — they are
rebuilt from the OpenAlex API.

## Steps

**1. Ingest the corpus.** The paper uses the Artificial Intelligence subfield
(OpenAlex `subfield.id:1702`), pre-filtered to works with abstracts: 1,561,123
articles and 17,632 books before cleaning, 1,465,427 and 17,183 after.

```sh
uv run scope-manager --help          # taxonomy browser + fetch/snapshot/embed
```

Long ingests should go through `bookgap.jobs` so they survive disconnection;
see CLAUDE.md.

**2. Embed and denoise.** The paper reports two embedders: GIST-small
(`avsolatorio/GIST-small-Embedding-v0`, 384D, dropping 10 dimensions) and
EmbeddingGemma (`google/embeddinggemma-300m`, 768D, dropping 394), both leaving
a 374D subspace after LPP. Both are fetched from HuggingFace by identifier and
are subject to their own licences — EmbeddingGemma is under the Gemma Terms of
Use, which are open-weights but not OSI open source.

**3. Detect gaps and generate proposals.**

```sh
uv run python bookgap/batch_generator.py --config-name standard
```

The paper's default run is GIST + LPP + `gemma4:31b` over the 77 AI topics, with
10 in-gap articles and 5 nearby books in each prompt, and gaps defined as
3/4-balls. Results land under `artifacts/`, and every generated file embeds the
command that produced it.

**4. Regenerate the paper's tables and figures.** The paper lives in a separate
repository, which consumes this one across a deliberately small surface — four
generator modules:

| paper artifact | generator |
| --- | --- |
| `artifacts/baseline_vs_pipeline_table.tex` | `artifacts_scripts.paper.baseline_vs_pipeline_table` |
| `artifacts/unused/per_topic_performance.tex` | `artifacts_scripts.paper.per_topic_performance_scatter` |
| `artifacts/proposal_appendix.tex` | `artifacts_scripts.paper.generate_proposal_appendix` |
| `artifacts/selected_books_table.tex` | `artifacts_scripts.survey.verify_survey_alignment` |

From the paper repo, `make artifacts` runs them.

---

## What is not reproducible here

### The human evaluation (Table 4, Table 8, section 6.6)

**The survey responses are not distributed and never will be.** They contain
participant email addresses, IP addresses, geolocation, and panel
re-identification keys. Participants gave that data to receive their results,
not for publication. It has been removed from this repository's git history.

Consequently `artifacts_scripts/survey/verify_survey_alignment.py` cannot run
against a clone, and the two human-evaluation tables cannot be regenerated.
The reported numbers stand on the paper alone.

What *is* included, so the methodology can be inspected and re-run end to end:

- `select_survey_books.py` — how the three arms were chosen: for each topic, a
  more-cited book, a less-cited book, and one generated proposal, ranked by a
  Gaussian-kernel-weighted temporal citation percentile with bootstrap CIs.
  Running it generates `selected_books.csv`.
- `generate_survey.py` — builds the Qualtrics instrument from that CSV. It needs
  `QUALTRICS_BRAND_ID` and `QUALTRICS_OWNER_ID` for the generated `.qsf` to
  import into your own Qualtrics organization.
- `verify_survey_alignment.py` — the analysis, runnable if you conduct your own
  survey in the same format.

### The exact corpus snapshot

OpenAlex is a living index. Re-ingesting today yields a slightly different
corpus than the paper's 2026 snapshot: works are added, abstracts are corrected,
topic assignments shift. Expect small numeric differences, not identical tables.

---

## Known gaps in this document

This section is honest bookkeeping rather than a disclaimer.

- The exact ingest invocation used for the paper's snapshot is not recorded
  here, only the scope. Recovering it from the `cache/.jobs/` manifests is a
  tracked task.
- `artifacts_scripts/paper/generate_latex_tables.py` writes to this repo's
  `artifacts/`, not the paper repo's copy, and running it with default arguments
  produces different numbers than the paper reports — the paper's tables come
  from a specific 2026-05-22 run. Do not run it expecting a no-op.
- Several `artifacts_scripts/` threads depend on the sibling `curriculum` repo,
  which is not public. The paper's pipeline does not.
