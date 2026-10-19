# Reproducing the paper

This document covers what is needed to reproduce the results in *Finding and
Filling Gaps in the Literature*. The [README](README.md) describes the project
generally; this file is scoped to the paper and is expected to age with it.

Read the **[What is not reproducible here](#what-is-not-reproducible-here)**
section first. Two parts of the paper cannot be regenerated from this repository,
and it is better to know that before starting.

---

## What the paper's pipeline actually is

The published numbers come from 27 runs plus a random-context baseline, all from
a narrow path through the codebase:

    artifacts_scripts/paper/regen_sweep.py   the 27 runs behind the paper's tables
      -> bookgap/llm/generators/
         workflow_functions.py              run_batch_workflow: per topic, gap ->
                                            prompt -> proposal -> scoring
         ollama_local.py                    gemma4:31b via Ollama
      -> bookgap/cloud.py                   works (per-scope Parquet), embeddings
                                            (memory-mapped .npy), citation graph
      -> bookgap/denoisers.py               LPP (the default) and the Appendix B
                                            alternatives
      -> bookgap/gaps.py                    gap detection, cached per topic
         bookgap/llm/gap_algorithms/        K-ball (the paper's)
    artifacts_scripts/run_random_baseline.py  the random-context baseline (Table IV)

`bookgap/batch_generator.py` is the general Hydra entry point onto the same
workflow; the paper's runs build each configuration in code instead
(`regen_sweep.py`, via `gemma4_quantitative_sweep._build_config`), so that every
cell is defined in one reviewable place.

Most of the repository is *not* this. `artifacts_scripts/` holds around a dozen
independent research threads, none of which is needed to reproduce the paper.

## Prerequisites

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/): `uv sync --extra dev`.
  The environment is pinned by `uv.lock`; the paper's runs used
  sentence-transformers 5.6.0 and torch 2.13.0.
- [Ollama](https://ollama.com/) with `gemma4:31b` pulled. The runs used Ollama
  0.32.14 and the model digest `17ba34c06c80…` (the Q4_K_M quantization, also
  tagged `gemma4:31b-it-q4_K_M`).
- One GPU with room for the 31B model at Q4 plus an embedder. The runs used one
  48 GB RTX 6000 Ada. Run one GPU job at a time: concurrent jobs slow each other
  far more than they overlap.
- Disk: the AI subfield's works are about 0.9 GB of Parquet; the embeddings are
  2.4 GB (GIST) and 4.7 GB (EmbeddingGemma).
- No API keys. The corpus comes from OpenAlex's public snapshot on S3 and the
  proposals from the local model. `OPENALEX_API_KEY` and `OPENAI_API_KEY` serve
  only the live-API ingest and the OpenAI Batch backend, which the paper does not
  use.

## Steps

**1. Ingest the corpus.** The paper uses the Artificial Intelligence subfield
(OpenAlex subfield 1702), books and articles, from the OpenAlex snapshot as of
2025-11-06:

```sh
uv run scope-manager snapshot 17/1702 --work-types book,article --snapshot-date 2025-11-06
```

`scope-manager` launches this as a tracked background job itself (`bookgap.jobs`).

Only works with abstracts are kept. That gives 1,698,538 articles and 16,316
books; after the syntactic filter, syntactic deduplication, dropping works without
a publication date, and semantic deduplication, 1,583,103 articles and 14,605
books remain for GIST (semantic deduplication runs on each embedder's vectors,
and EmbeddingGemma keeps 1,593,438 and 14,692). The paper's corpus was ingested
as part of a larger one (all of OpenAlex domain 3, all work types, the same
snapshot date); the per-topic Parquet files for subfield 1702 are the same either
way. See [the exact corpus snapshot](#the-exact-corpus-snapshot) before relying
on a re-ingest matching.

**2. Embed.** Nothing to run: each run embeds the corpus on first use and caches
it under `cache/embeddings_npy/`. The paper reports two embedders, GIST-small
(`avsolatorio/GIST-small-Embedding-v0`, 384D, dropping 10 dimensions) and
EmbeddingGemma (`google/embeddinggemma-300m`, 768D, dropping 394), both leaving a
374D subspace after LPP. EmbeddingGemma takes about three GPU-hours to embed the
corpus; GIST far less. Both are fetched from Hugging Face by identifier and are
subject to their own licences (EmbeddingGemma's Gemma Terms of Use are
open-weights but not OSI open source).

Check EmbeddingGemma's vectors before trusting them: an earlier environment
(May 2026) computed them wrongly without any error, at cosine 0.4–0.8 against the
correct ones. The model card's example similarities are a quick test; GIST was
unaffected.

**3. Run the 27 cells and the baseline.** One chain per embedder, run one after
the other:

```sh
uv run python -m bookgap.jobs run --prefix regen_gist -- \
    uv run python -m artifacts_scripts.paper.regen_sweep --embedder gist --topics all
uv run python -m bookgap.jobs run --prefix regen_gemma -- \
    uv run python -m artifacts_scripts.paper.regen_sweep --embedder gemma --topics all
uv run python -m artifacts_scripts.run_random_baseline \
    --run-dir outputs/regen_2026-10/runs_t77/gist_a10_b5_dropk10_lpp_t77_r0.75-<hash> \
    --output-dir outputs/regen_2026-10/random_baseline
```

The cells are the context ablation (Table III), the denoiser comparison
(Table II, and Appendix B's intervals), and, GIST only, the radius sweep
(Table V). The paper runs every cell on all 77 topics (`--topics all`), landing
under `outputs/regen_2026-10/runs_t77/`, with the radius runs under
`radius_runs/`. `--topics random20` runs the ablation and denoiser cells on a
fixed random 20 topics (`TOPIC_SUBSET_RANDOM_20`) instead, into `runs/`, as an
earlier version of the paper did; GIST's default cell, the headline run, is on
all 77 topics either way.

Finding a cell's gaps is CPU work, about a minute a topic, during which the GPU
would sit idle. Gaps are cached per topic and shared by every cell with the same
embedder, denoiser and radius, so they can be computed on the CPU before or
beside a chain:

```sh
uv run python -m bookgap.jobs run --prefix regen_gaps -- \
    uv run python -m artifacts_scripts.paper.regen_sweep --embedder gist --topics all --gaps-only 6
```

It competes with the chain's scoring, which is CPU-bound too: beside the paper's
sweep, even at idle priority (`chrt --idle`), it stretched each cell's scoring
from about 6 minutes to 20-40 while they overlapped. That cost the sweep under an
hour against roughly ten hours of gap detection saved; running it before the
chains avoids even that.

A run's directory is named by its cell and a hash of its configuration and the
model's digest, so a chain can be stopped (`uv run python -m bookgap.jobs stop
<job_id>`, which takes effect between proposals) and restarted with the same
command: finished cells are skipped and an unfinished one resumes where it
stopped. `--dry-run` lists each cell's directory and whether it is done. Each
run's `provenance.json` records the cell, the model's digest and, per session
that worked on it, the command, commit, every package version and the context
window. On the GPU above, the 23 all-topic cells not already run (the headline
and radius runs were) took about 17.5 hours, with their gaps computed beside
them.

No prompt is ever truncated. Ollama silently cuts over-long prompts to about
half its window by default; the generator instead asks it to refuse them
(an HTTP 400), and fails any reply whose prompt and answer run past the window.
That count leaves out the model's thinking, which Ollama does not report; estimated
generously from its text, no reply in the paper's runs came within 10% of its
window. The window is `llm.ollama_num_ctx`, 65,536 tokens, which 26 of the 27
runs used; the headline run used 16,384, which all its prompts fit. At radius
fraction 1/2 one prompt is 29,133 tokens (T13062's gap is ten near-duplicate
citation-list abstracts).

**4. Regenerate the paper's tables and figures.** The paper lives in a separate
repository. From there, `make artifacts` runs every generator that feeds it,
reading the runs named by the Makefile's `REGEN` and `RUN_DIR`:

| paper artifact | generator |
| --- | --- |
| Tables II, III and VII (`artifacts/latex_tables/`) | `artifacts_scripts.output_run_summary`, then `artifacts_scripts.paper.generate_latex_tables` |
| Table IV, `baseline_vs_pipeline_table.tex` | `artifacts_scripts.paper.baseline_vs_pipeline_table` |
| Table V, `radius_sweep_table.tex` | `artifacts_scripts.paper.radius_sweep_table` |
| Appendix B's intervals, `denoiser_difference_cis.md` | `artifacts_scripts.paper.denoiser_difference_cis` |
| Appendix B's Fig. 4, `denoiser_differences.tex` | `artifacts_scripts.paper.denoiser_difference_plot` |
| the embedder comparison's separability, `book_article_separability.md` | `artifacts_scripts.paper.book_article_separability` |
| Appendix F, `proposal_appendix.tex` | `artifacts_scripts.paper.generate_proposal_appendix` |
| the survey tables, `selected_books*_table.tex` | `artifacts_scripts.survey.verify_survey_alignment` (needs the survey data; see below) |

Two figures are generated by hand rather than by `make artifacts`:

- Fig. 1: `uv run python -m artifacts_scripts.pball_voronoi_example` (simulated
  data, seeded).
- Fig. 2: `artifacts_scripts/historical_gap_analysis/` (see its README) over the
  years 2000–2025 at radius fraction 3/4, GIST and LPP with k = 10, then
  `synthesize_results.py` and `plot_results.py --output-dir paper`. Its default
  cloud configuration is no longer the paper's, so pass the corpus explicitly:
  `experiment.cloud.filter_pair.pre.domain=null
  experiment.cloud.filter_pair.pre.subfield=1702
  'experiment.cloud.filter_pair.pre.work_types=[book,article]'`. About ten
  minutes per year as written.

Every generated file ends with the command and the bookgap commit that produced
it, naming any uncommitted change in the checkout it was produced from (the
README's "Generated files" says how one is excused).

---

## What is not reproducible here

### The human evaluation (Tables VI and VIII, Section VI-F)

**The survey responses are not distributed and never will be.** They contain
participant email addresses, IP addresses, geolocation, and panel
re-identification keys. Participants gave that data to receive their results,
not for publication. It has been removed from this repository's git history.

Consequently `artifacts_scripts/survey/verify_survey_alignment.py` cannot run
against a clone, and the human-evaluation tables cannot be regenerated. The
reported numbers stand on the paper alone.

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

The corpus is pinned to the OpenAlex snapshot as of 2025-11-06, but that does not
make it re-downloadable. OpenAlex partitions its snapshot by each record's *last*
change, so a later release no longer holds a record as it stood in November 2025,
and `--snapshot-date 2025-11-06` against a later release selects a different set.
The live API differs more: of 200 works sampled from the paper's corpus in October
2026, a fifth no longer resolved by ID, and of the rest about half had moved out
of subfield 1702 and over a third of the articles were retyped `conference-paper`.

OpenAlex keeps its pre-2026 layout frozen under a `legacy-data/` prefix, which may
still hold the November 2025 records; `bookgap.snapshot_ingest.run` takes the
prefix as an argument, but this has not been tried. Failing that, an exact
reproduction needs the per-topic Parquet files the runs read (OpenAlex data is
CC0, about 0.9 GB for subfield 1702), which are not distributed yet.

---

## Known gaps in this document

This section is honest bookkeeping rather than a disclaimer.

- The run outputs (`outputs/regen_2026-10/`: prompts, proposals, scores) and the
  caches are not distributed, so steps 3 and 4 cannot be checked against the
  paper's own runs without re-running them.
- Several `artifacts_scripts/` threads depend on the sibling `curriculum` repo,
  which is not public. The paper's pipeline does not.
