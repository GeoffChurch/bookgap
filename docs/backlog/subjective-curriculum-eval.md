# Subjective evaluation of the curriculum method on best-proposal gaps

> **Note — tooling removed.** The `artifacts_scripts/curriculum/` modules referenced in this doc were deleted in `0f2d0ad`; findings stand, commands do not. See `docs/curriculum-pipeline-removed.md`.

**Living / in-progress doc.** Goal: hand-pick a handful of topics whose textbook
*proposals* are especially strong, recover the exact in-gap articles behind each,
and generate a **curriculum** for each gap (scoped to the whole Physical Sciences
domain), then subjectively evaluate the curriculum method. This doc is the
self-contained recipe so a new user can reproduce every step cold.

## Status / working notes

- [x] Picked the production run and the candidate topics (see [Topics](#topics)).
- [x] Confirmed the full in-gap article set is recoverable (Recipe B).
- [x] Confirmed curriculum at domain scale is feasible — already run at 40.9M works
      (Recipe C).
- [x] Locked the 5 recommended topics and ran all of them.
- [x] Built the driver `artifacts_scripts/curriculum/seed_from_gap.py` (uniform seeds).
- [x] **Generated all 5 curricula (2026-05-29)** — see [Results](#results-2026-05-29)
      and `artifacts/curriculum/GAP_CURRICULA_INDEX.md`.
- [ ] **Your turn:** subjectively review the 5 curricula (the actual eval).
- [ ] (Backlogged) probabilistic seed weighting — see Recipe C → Future refinement.
- [ ] Seed breadth: used all in-gap articles (uniform). top-K is an open ablation.
- [ ] Define the subjective-eval rubric (what "good curriculum" means here).

## Results (2026-05-29)

All 5 curricula generated, uniform seeding over each gap's full in-gap article set,
`domain=3` (Physical Sciences, 40.9M works) scope, forward-only PageRank, clustering
the top 5,000 ranked works into 15 topics. ~40–50 min/run (dominated by the
15-batch frontier-adjacency PageRank). Outputs under
`artifacts/curriculum/gap_curriculum_<TID>/` (`summary.md`, `topic_keywords.json`,
`topic_work_ids.json`, `curriculum_plot_with_background.png`, `metadata.json`);
combined index at `artifacts/curriculum/GAP_CURRICULA_INDEX.md`.

**Qualitative finding:** the design worked as intended — each curriculum's
`Sim→Target` column *ascends* from foundational prerequisites (which live outside
the gap) to the gap topic at the frontier, e.g.:

| Topic | foundation (low sim) | frontier (high sim → the gap) |
|---|---|---|
| T11273 GNN | Parallel Distributed Processing, Laplacian Eigenmaps, Matrix Analysis (~0.86) | GraRep, LINE, ConvE, network embedding (~0.96) |
| T11902 ITS | Intelligent Tutoring Systems, AI: A Modern Approach (~0.84) | Deep Knowledge Tracing, Temporal Conv. KT (~0.96) |
| T10181 NLP | information theory / decoding (0.825) | Chinese word segmentation (0.974) |
| T13629 Text-Simpl. | WordNet (0.902) | lexical simplification / corpora (0.963) |

Minor cross-disciplinary noise appears in a few topics (e.g. a PCM/signal-processing
cluster in T11902) — expected when seeding PageRank over the whole domain.

**Decisions taken**

- **Gap → curriculum seeding (Option A):** pass the gap's in-gap article IDs as
  `target_work_ids`; PageRank + clustering run over the full cloud, so the
  curriculum *includes prerequisite/foundational works that lie outside the gap*
  (the gap is the under-covered frontier; the on-ramp lives in the older,
  well-covered literature). We did **not** constrain clustering to the gap set.
- **Cloud scope:** entire **Physical Sciences domain** (`domain=3`), which is
  already the curriculum pipeline's default and is proven at this scale.
- **Ranking method:** leave the **default `BIDIRECTIONAL_PAGE_RANK`**.
  `target_work_ids` populate the *forward* seed vector (forward: targets →
  prerequisites); with no `source_work_ids` the reverse pass is simply skipped and
  it degrades to forward-only personalized PageRank — mass flows to the gap's
  prerequisites, our curriculum on-ramp. The "missing source → forward-only"
  fallback is explicit in `build_curriculum.py:746-795` and
  `pagerank.py:two_direction_pagerank`, so **no override is needed** (`method=PAGE_RANK`
  is an exactly-equivalent forward-only alternative in the no-source case).
- **Seed weighting:** **uniform for the first run** — every in-gap article gets
  equal seed mass (no new metric code needed beyond the driver). Probabilistic
  weighting by *P(article is in the true gap)* is deferred (see Recipe C → Future
  refinement): it isn't readily available (this run used `raw_kball`, a hard count),
  so it's backlogged rather than blocking the first run.

---

## The run

```
outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10/
```

This is the paper's headline production run: `gemma4:31b` via Ollama, GIST-small
embedder, LPP denoiser dropping k=10 dims, a=10 prompt articles / b=5 prompt
books, on the OpenAlex AI subfield (`field=17, subfield=1702`). **77 topics**, one
gap each (`gap_idx=0`).

### Layout — what lives where

| Path | One row/record per | Holds |
|---|---|---|
| `gaps_analysis_results.csv` | topic (77 rows) | all metrics + **`proposal_title`, `proposal_abstract`** (cols 20–21). No input article/book text. |
| `contexts/gap_0_topic_T#####.json` | topic | **full gap analysis** (schema v2): `articles_by_centrality` (every in-gap article, `openalex_id`+`display_name`+`abstract`), `gap_stats`, `furthest_article`, `covering_books`, `proposal_coverage`, `local_projection_2d`. |
| `contexts/gap_0_<name>_id=T#####_analysis.txt` | topic | human-readable; **top-10 central articles only** (display truncation). |
| `prompts/gap_0_topic_T#####.json` | topic | exact LLM input: `sampled_articles` (10), `covering_books` (5), `backend`/`model`/`generation_params`. |
| `ollama_results.jsonl` | topic | the LLM output: `proposal.{title,abstract}`, `thinking`, `raw_content`, `metadata`. Keyed by `custom_id`. |

Titles + abstracts of the **proposal** are in *both* the CSV and `ollama_results.jsonl`.
Titles + abstracts of the **input articles/books** are in the per-topic
`prompts/`+`contexts/` JSON, **not** the CSV.

Code: context JSON schema is `bookgap/llm/prompt_record.py` (`ContextRecord`,
`articles_by_centrality`, schema v2). Written by `bookgap/llm/artifacts/context_export.py`.

---

## Recipe A — get titles/abstracts/metrics for a run and filter to useful ones

Works for *any* run dir, not just this one.

```python
import pandas as pd
RUN = "outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10"
df = pd.read_csv(f"{RUN}/gaps_analysis_results.csv")
df.columns.tolist()
# topic_id, gap_idx, proposal_book_rank, proposal_book_rank_fraction,
# proposal_article_rank, proposal_article_rank_fraction, gap_coverage_fraction,
# proposal_gap_size, proposal_gap_size_percentile, proposal_gap_zscore,
# topic_name, num_books, num_articles, gap_size, gap_size_percentile, gap_zscore,
# shapiro_log10_pval, actual_num_prompt_articles, actual_num_prompt_books,
# proposal_title, proposal_abstract
```

**Filter to the geometrically strong proposals.** The quality signals are:
`proposal_book_rank == 0` (the proposal is the single closest "book" to the gap
centroid), `gap_coverage_fraction` (how much of the gap the proposal covers), and
`proposal_gap_size_percentile == 1.0` (the gap is larger than every existing-book
gap on the topic).

```python
top = df[(df.proposal_book_rank == 0)
         & (df.gap_coverage_fraction >= 0.90)
         & (df.proposal_gap_size_percentile >= 1.0)]
top = top.sort_values(["gap_coverage_fraction", "proposal_article_rank_fraction"],
                      ascending=[False, True])
print(top[["topic_id", "topic_name", "proposal_title", "gap_coverage_fraction"]].to_string())
```

On this run that yields 22 candidates.

> **Caveat — metrics are necessary, not sufficient.** A high-coverage proposal can
> still be a **label/content mismatch**: the gap content is coherent but mislabeled
> by OpenAlex, so the proposal is "perfect for the gap, wrong for the topic name."
> Examples in this run: `T14335` ("Educational Robotics" → a *physical-therapy
> history* textbook), `T12026`/`T10126` (AI labels → *biology/arthropod taxonomy*).
> **Always read the proposal and check it against the topic name + the gap's
> central articles before trusting it.** See [Topics](#topics) for the curated set.

**Pull the input article/book titles + abstracts** (not in the CSV):

```python
import json
# exact LLM inputs (10 sampled articles, 5 selected books):
p = json.load(open(f"{RUN}/prompts/gap_0_topic_T11273.json"))
[a["display_name"] for a in p["sampled_articles"]]
[b["display_name"] for b in p["covering_books"]]   # + b["abstract"], b["fraction_covered"]
# OR the full ranked context (every in-gap article + every covering book):
c = json.load(open(f"{RUN}/contexts/gap_0_topic_T11273.json"))
len(c["articles_by_centrality"])                    # full in-gap set
[b["display_name"] for b in c["covering_books"]]
```

---

## Recipe B — recover a gap's full in-gap article set

The `.txt` files show only the top-10. The **full** set (titles + abstracts) is
available two ways.

### Fast path (read the JSON — preferred, zero compute)

```python
import json
ctx = json.load(open(f"{RUN}/contexts/gap_0_topic_T11273.json"))
arts = ctx["articles_by_centrality"]          # full set, sorted by centrality
ids   = [a["openalex_id"]   for a in arts]    # -> seeds for the curriculum step
print(len(ids), "in-gap articles")
arts[0]   # {"openalex_id": "...", "display_name": <title>, "abstract": <str>}

# dump IDs for Recipe C
json.dump(ids, open("artifacts/curriculum_eval/T11273_gap0_article_ids.json", "w"))
```

### Recompute path (deterministic — if the JSON is ever missing)

Gap detection is `@MEMORY.cache`d with fixed seeds (`bookgap/gaps.py:31`,
`compute_gaps_for_config` at `:243`); same config + same OpenAlex snapshot ⇒
identical membership. The joblib cache is present on disk
(`cache/joblib/bookgap/gaps/_compute_gaps_for_topic`).

```python
from omegaconf import OmegaConf
from bookgap.gaps import compute_gaps_for_config
# experiment_config.yaml stores enums as strings; coerce them back before use.
# Canonical loader (gap.cell_type, gap.deflation_strategy, llm.backend, llm.model):
#   see artifacts_scripts/lenient_recover_failed_proposals.py  (config-load section)
cfg  = OmegaConf.load(f"{RUN}/experiment_config.yaml")   # + enum coercion
gaps = compute_gaps_for_config(cfg)
g    = next(x for x in gaps if x.topic_id == "T11273" and x.gap_idx == 0)
di   = g.gap.get_dists_idxs()                 # DistsAndIdxs(sqdists, idxs)
works = g.articles.works()
ids  = [works[int(i)]["id"] for i in di.idxs] # full in-gap set, central-first
```

**Determinism caveats:** requires the *same* `snapshot_date` and embedder as the
run (`experiment_config.yaml` records both). HNSW is seeded/deterministic;
deflation KMeans uses `random_state=0`. The `.txt` truncation to top-10 is
display-only — full membership is always computed in memory at run time.

---

## Recipe C — generate a curriculum for a gap (Physical Sciences scope)

**Idea:** feed the gap's in-gap article IDs as `target_work_ids`; the curriculum
pipeline seeds PageRank from them over the **whole `domain=3` (Physical Sciences)
citation graph** (~41M works after dedup — 42.2M raw parquet rows across fields
15,16,17,19,21,22,23,25,26,31), ranks all works by relevance to the gap, clusters
the top `ranking_top_k`, and emits an
ordered topic structure (prerequisites → frontier).

**Scale is proven.** 26 existing runs under `artifacts/curriculum/` have
`cloud_name=(domain=3, ...)`, `n_works_total=40,914,141`, `n_works_clustered=5000`,
`rank_before_clustering=True`. Keep that invariant: PageRank is sparse over 42M
(cheap); only the top-5000 reach the O(n²) clustering step.

- **Cloud scope:** `domain=3` is the curriculum pipeline's default
  (`CurriculumConfig.experiment.cloud=CloudConfig()`), so **no scope override is
  needed**. The gap's AI articles (`field=17`) are a subset of the domain ⇒ they
  resolve as seeds.
- **Long-running + memory-heavy** (~60 GB+ to load the 42M×384 embedding matrix;
  `bookgap/cloud.py` warns this can OOM a 250 GB box). Run on a big-memory host
  and wrap in `bookgap.jobs`.

Entry point: `python -m artifacts_scripts.curriculum.main` (`build_curriculum.py`).
Input contract: `target_work_ids: List[str]` (required) + optional `source_work_ids`.
Output: `artifacts/curriculum/<output_dir>/` — `summary.md` (human-readable topics
+ keywords), `topic_work_ids.json`, `topic_keywords.json`, `curriculum_plot.png`,
`metadata.json`, plus the `.npy` dendrogram/mass artifacts.

### Option A (small seed set — CLI, runnable today)

Hundreds–thousands of full OpenAlex URLs overflow a shell command line, so the CLI
form only fits a **top-K most-central** seed subset:

```bash
RUN=outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10
# top-50 most-central in-gap articles as seeds:
IDS=$(conda run -p .conda python - <<'PY'
import json
c = json.load(open("outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10/contexts/gap_0_topic_T11273.json"))
print(",".join(a["openalex_id"] for a in c["articles_by_centrality"][:50]))
PY
)
conda run -p .conda python -m bookgap.jobs run --prefix curriculum_T11273 -- \
  conda run -p .conda python -m artifacts_scripts.curriculum.main \
    "target_work_ids=[$IDS]" \
    rank_before_clustering=true ranking_top_k=5000 \
    desired_n_topics=15 \
    output_dir=gap_curriculum_T11273
# method defaults to BIDIRECTIONAL_PAGE_RANK; with no source_work_ids it runs
# forward-only PR (mass → the gap's prerequisites). No method override needed.
```

### Option B (full seed set — small driver, **to build**)

For the full in-gap set, write a small programmatic driver and **upstream it into
`artifacts_scripts/curriculum/`** (e.g. `seed_from_gap.py`) — it's curriculum-specific
glue. Model it on `artifacts_scripts/radius_sweep.py`: read the in-gap IDs from the
context JSON (Recipe B), build a `CurriculumConfig` with `target_work_ids=<all ids>`
(uniform seed mass), and invoke the pipeline in-process (bypassing the CLI length
limit). Verify the in-process entry point in `artifacts_scripts/curriculum/main.py`
first. **This is the one piece of new code the first run needs.**

### Open questions for Recipe C (working notes)

- **Seed breadth:** all in-gap articles vs top-K most central. With the driver the
  CLI limit is moot, so this is purely about how widely PageRank mass is spread —
  default to the full set for the first run; top-K is a cheap ablation.
- **Seed resolution:** the `domain=3` cloud runs its own semantic dedup, so a few
  seed IDs may be merged/dropped vs the AI-subfield run. Check the resolved-seed
  count in `pipeline.log` (`_find_work_indices`); it errors only if *none* resolve.
- **Sanity of large uniform seeding:** prior domain runs used 3 targets; seeding
  50–500 uniform targets is heavier-tailed. Inspect the PageRank mass distribution
  and `summary.md` to confirm the curriculum isn't dominated by one cross-discipline
  basin.
- **`desired_n_topics`** (default 15) and `linkage_method` (default
  `weighted_bisecting_kmeans`) are unswept here.

### Future refinement (backlog): probabilistic seed weighting

The first run uses **uniform** seed mass. A principled refinement is to weight each
gap article by its **probability of being in the true gap** — the per-member `soft`
term in `kde_ratio` (`bookgap/gap_size_metrics.py:177`):
`sigmoid(log gap_density − log book_mixture_density)`, whose sum is the soft
effective size (n_eff). A book-covered-but-central article gets downweighted; a
deep-in-the-void article approaches 1.

**Not readily available — deferred, not blocking the first run:**

- This run used `size_metric: raw_kball` (hard count), so no soft memberships were
  computed or stored; `articles_by_centrality` carries no per-article score.
- `kde_ratio` currently returns only the *sum* and discards the per-member `soft`
  vector (`gap_size_metrics.py:247-252`).

**To implement (all small):**

1. Have `kde_ratio` optionally return the per-member `soft` array.
2. In the driver, recompute the gap (Recipe B) for members + denoised article/book
   embeddings, then compute `soft`.
3. Add an optional `target_work_weights` to `CurriculumConfig` /
   `build_curriculum.py` (today it hardcodes uniform `target_seeds[idx] = 1/len` at
   `build_curriculum.py:756`) and feed `soft`.

Membership is unchanged (same raw-kball member set, reweighted). Cheaper middle
option for comparison: distance-kernel weights `exp(-d²/2σ²)` from the recovered
`sqdists` (centrality only; ignores book coverage).

---

## Topics

From Recipe A's 22 candidates, filtered by reading each proposal and verifying it
against the gap's central articles. All metrics from `gaps_analysis_results.csv`
(`gap_idx=0`).

### Recommended 5

| Topic | OpenAlex label | Proposal title | cov | pct | book_rank | n_books |
|---|---|---|---|---|---|---|
| **T11273** | Advanced Graph Neural Networks | Advanced Network Embedding: Representation Learning for Attributed, Heterogeneous, and Hypergraphs | 0.996 | 1.00 | 0 | 20 |
| **T11902** | Intelligent Tutoring Systems and Adaptive Learning | Computational Knowledge Tracing: From Cognitive Foundations to Deep Learning | 0.996 | 1.00 | 0 | 177 |
| **T10181** | Natural Language Processing Techniques | Computational Chinese Word Segmentation: Theory, Adaptation, and Applications | 0.995 | 1.00 | 0 | 1231 |
| **T12761** | Data Stream Mining Techniques | Adaptive Machine Learning for Data Streams: Managing Concept Drift and Evolution | 0.991 | 1.00 | 0 | 25 |
| **T13629** | Text Readability and Simplification | Automatic Text Simplification: Principles, Methods, and Applications | 0.985 | 1.00 | 0 | 30 |

### Bench (swap freely — equally strong, adds domain breadth)

| Topic | OpenAlex label | Proposal title | cov | n_books |
|---|---|---|---|---|
| T10906 | AI-based Problem Solving and Planning | Autonomous Systems for Space Exploration: Planning, Execution, and Human-Robot Collaboration | 0.995 | 171 |
| T12805 | Cognitive Science and Mapping | Fuzzy Cognitive Maps: Theory, Extensions, and Applications in Complex System Modeling | 0.979 | 459 |
| T11424 | Security and Verification in Computing | Microarchitectural Security: Side-Channel Analysis and Hardware-Level Defenses | 0.970 | 52 |
| T12535 | Machine Learning and Data Classification | Robust Deep Learning: Navigating and Mitigating Label Noise | 0.962 | 57 |

### Avoid

- **Label/content mismatches** (high metrics, off-label proposals): `T14335`
  (physical-therapy history), `T12026` & `T10126` (biology/arthropod taxonomy),
  `T12260` (PE pedagogy, translationese).
- **The six §5.1 failure topics:** `T10215`, `T10639`, `T12157`, `T13623`,
  `T13674`, `T14381` (none reached the top tier anyway).

### Per-topic progress

| Topic | gap IDs recovered | curriculum generated | reviewed |
|---|---|---|---|
| T11273 | ✅ (722) | ✅ | ☐ |
| T11902 | ✅ | ✅ | ☐ |
| T10181 | ✅ | ✅ | ☐ |
| T12761 | ✅ | ✅ | ☐ |
| T13629 | ✅ | ✅ | ☐ |

---

## Source-file index

- Gap detection / membership: `bookgap/gaps.py` (`compute_gaps_for_config:243`,
  cached `_compute_gaps_for_topic:31`), `bookgap/llm/gap_algorithms/`.
- Context JSON schema: `bookgap/llm/prompt_record.py` (`ContextRecord`, schema v2);
  writer `bookgap/llm/artifacts/context_export.py`.
- Config reload pattern: `artifacts_scripts/lenient_recover_failed_proposals.py`.
- Curriculum pipeline: `artifacts_scripts/curriculum/main.py`,
  `build_curriculum.py`, `config.py` (`CurriculumConfig`), `pagerank.py`.
- Domain scope: `bookgap/util.py` (`Domain.PHYSICAL_SCIENCES="3"`, `Filter`),
  `bookgap/openalex_scopes.py` (`resolve_filter_upward`).
- Curriculum method background: `docs/curriculum-method-comparison.md`.
