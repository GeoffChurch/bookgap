# Expanding the bake-off to vary the gap-selection algorithm

**Living doc.** Iteratively tackled tier-by-tier; status section grows as work progresses.

## Goal

Extend the denoiser bake-off (currently sweeps `embedder × denoiser × d`) to also sweep the **gap-selection / gap-scoring algorithm**. We want to know whether the denoiser rankings (LPP/CITATION_ERROR_LDA on GIST, LIP/LPP on Gemma) are robust to the choice of gap algorithm, or whether different algorithms produce qualitatively different conclusions.

Self-contained context: the bake-off as of 2026-05-28 covers ~150 cells across the dimensions above. Per-cell metric outputs are `gap_coverage_fraction_mean`, `proposal_gap_size_percentile_mean`, plus per-topic versions used for paired-CI analysis. Gap selection is currently fixed at the production k-ball algorithm with `radius_fraction=0.75`. See `bookgap/gap_size_metrics.py` for the five production gap-sizing metrics already implemented.

## The interpretation challenge — what "apples-to-apples" means here

The current metrics depend on which gap algorithm defined "the gap":
- **Coverage** = "how well does the proposal cover articles in *this algorithm's* gap"
- **Proposal-percentile** = "how prominent is the proposal's location in *this algorithm's* gap-size distribution"

So direct cross-algorithm comparison is interpretation-fraught. Algorithm A getting higher coverage than B might just mean A picks *easier* gaps (smaller, topically tighter) on which any proposal does well.

What we can do, ranked by rigor:

1. **Per-algorithm curves separately.** Each algorithm's d-curve is internally meaningful. Cross-algorithm visual comparisons reveal *shape* differences (peak location, curve flattening, CI widths). Absolute numbers compare loosely.
2. **Random-baseline-normalized coverage.** For each (algorithm, topic), compute expected coverage of a RANDOM proposal. Report `(cov − random)` per cell. Controls for the "easy gap" effect. ~50 LOC.
3. **External metric: held-out book recovery.** See sibling backlog entry [[heldout-book-recovery-production-eval]]. Withhold N books per topic, run gap detection + proposal generation, check if the proposed book resembles any withheld book. Truly algorithm-independent. ~300 LOC.
4. **Cross-scoring matrix.** Generate ONE proposal per cell, then score it against EVERY algorithm's gap definition. Get `coverage_via_A`, `coverage_via_B`, … for the same proposal. Most informative; expensive in interpretation; ~150 LOC.

The Tier 1 plan below sidesteps the issue entirely by varying ONLY the scoring of fixed gaps. Tiers 2-3 progressively address it.

## Tier 1 — Vary gap *scoring* (same gap selection)

**Scope:** Add `gap.size_metric` as a sweep axis. The existing `bookgap/gap_size_metrics.py` already has 5 production-ready metrics that operate on the SAME k-ball-detected gaps:
- `raw_kball` (identity / member count)
- `virtual_centroid` (members preferring gap centroid over nearest book)
- `mutual_nn` (partition discount via 1-NN book + mutual-NN correction)
- `book_knn_discount` (top-k book claims globally)
- `kde_ratio` (gap KDE vs book mixture KDE, spherical Gaussian fits)

**Apples-to-apples by construction**: same gaps, different scoring functions. The actual gap detection is unchanged; only the post-hoc per-gap score differs.

**Impl outline** (~1-2 days work):
1. Confirm `bookgap/gap_size_metrics.py` exposes a clean dispatch by enum name.
2. Add `gap.size_metric: str` field to `GapConfig` (already exists in some form — check `bookgap/configs/gap/gap_config.py`).
3. Wire `gap.size_metric` through `workflow_functions.run_batch_workflow` so the chosen scoring function is applied at the right point.
4. Add `gap_size_metric` column to `output_run_summary.py` output CSV (so we don't conflate cells across metric choice).
5. Update `artifacts_scripts/gemma4_quantitative_sweep.py` (or the per-topic autonomous-search driver) to include `gap.size_metric` as a sweep axis.
6. Update `artifacts_scripts/autonomous_d_search.py` planner if we want gap_size_metric varied automatically. Otherwise keep it as a one-off sweep.

**Compute:** Headline cells per embedder × 5 metrics. If we limit to top-5 denoisers × 4 d values × 2 embedders × 5 metrics = 200 cells × ~17 min ≈ 56h. Could trim to top-3 denoisers × 3 d × 2 emb × 5 = 90 cells ≈ 25h.

**Deliverable:** Tier-1 plot (coverage vs d, one curve per metric per denoiser) + Tier-1 paired-CI table per metric. The cross-metric headline question: "do CITATION_ERROR_LDA d=382 and LPP d=374 stay tied under all 5 scoring metrics on GIST?"

**Cross-refs:**
- [[kball-residual-mass]] for the four `gap.size_metric` variants this builds on
- [[dual-landmark-eff-size-reporting]] for the cohesion-metric design context

## Tier 2 — Add alternative gap-selection algorithms

**Scope:** Promote 1-2 of the exploratory `artifacts_scripts/gap_*.py` algorithms to production and include as a sweep dimension. Most-promising candidates (per the scan in `nearest-book-radius.md`):
- **`gap_effective_size_local.py`** — local-bg posterior (CSLS-balanced K_B from 1-NN books). Density-adaptive; uses 1-NN book information for boundary definition. Production-adjacent.
- **`gap_soft_em_fullmix.py`** — full-data 2-component mixture posterior. Self-tuning Gaussian. Different operational model from k-ball.

These genuinely change the gap-selection algorithm, so direct coverage comparison across algorithms is fraught. **Apples-to-apples requires the random-baseline normalization from option 2 above.**

**Impl outline** (~3-5 days work):
1. Identify the minimal interface needed: gap-detection function that returns a `list[Gap]` given a `Cloud`.
2. Refactor selected exploratory scripts to expose this interface. Move the core algorithm logic to `bookgap/gap_algorithms/` (or similar).
3. Add `gap.algorithm: str` field to `GapConfig` (extending the existing `CellType` enum mechanism — see `bookgap/types.py`).
4. Implement random-baseline normalization: for each (algorithm, topic), compute expected `cov_random = E[similarity(random_book, gap_articles)]`. Store as a per-cell column. Report `cov - cov_random` as the apples-to-apples metric.
5. Add `gap.algorithm` as a sweep axis. Use random-baseline normalization in the headline plot.

**Compute:** Tier-2 cells = Tier-1 cells × ~2 new algorithms = ~50h additional. May need to start with fewer (denoiser, d) combinations to keep compute tractable.

**Deliverable:** Tier-2 normalized-coverage table per algorithm. Question: "does the headline ranking on GIST flip if we use `gap_soft_em_fullmix` instead of k-ball?"

**Cross-refs:**
- [[nearest-book-radius]] for the full punch-list of n_eff-based algorithms
- [[multi-start-union-pipeline]] for the idea of *combining* multiple gap selectors
- [[kball-vs-hdbscan-statistical-scoring]] for HDBSCAN as a fundamentally different alternative

## Tier 3 — External evaluation metric (held-out book recovery)

**Scope:** Replace the dependence on gap algorithms entirely. For each topic, withhold N books from the corpus, run gap detection + proposal generation on the held-out corpus, then check if the proposed book resembles (in some embedding-similarity sense) any of the withheld books.

This is the truly algorithm-independent metric: it doesn't depend on which gap algorithm defines "the gap"; it measures "did the pipeline recover books we already know exist."

**Impl outline** (~1-2 weeks work):
1. Implement held-out book sampling per topic (stratified by some criteria — by total citations? publication year? topic coverage?).
2. Wire the held-out set through the bake-off driver: each (embedder, denoiser, d, algorithm) cell runs with the held-out books removed.
3. Implement the recall@K metric: given the proposed book embedding, count how many of the K nearest works (in the original embedding space) are held-out books from this topic.
4. Replace coverage / percentile in the headline reporting with held-out recovery as the apples-to-apples metric.

**Compute:** ~2-3× current cell count (because every cell re-runs with a different held-out subset, and each subset is sampled multiple times to reduce variance). Possibly 100h+.

**Deliverable:** Cross-algorithm recall@K curves. **This is the only metric that lets us make rigorous claims like "algorithm A is better than algorithm B."**

**Cross-refs:**
- [[heldout-book-recovery-production-eval]] — existing backlog entry for the standalone version of this eval
- [[book-rediscovery-eval-candidates]] — living tracker of methods worth running through this eval

## Status / working notes

(Updated as work proceeds.)

- 2026-05-29: doc created. Tier 1 is the recommended starting point given highest information-per-hour ratio and apples-to-apples-by-construction.
- Open question before starting Tier 1: does the existing `gap.size_metric` config field already plumb through the workflow, or do we need to wire it up first? Check `bookgap/configs/gap/gap_config.py` and `workflow_functions.run_batch_workflow`.
- Open question for Tier 2: which 1-2 exploratory algorithms are most promising? Re-read `gap_effective_size_local.py` and `gap_soft_em_fullmix.py` in detail before committing.
- Compute budget consideration: with the autonomous d-search infrastructure already in place, we could potentially extend the planner to handle `gap.size_metric` as another search dimension. But adding a categorical dimension to the bisection search is a non-trivial generalization.

## Recommendation for next action

Start with **Tier 1 dry-run analysis** (option (b) from the original triage question): before committing to the full Tier 1 sweep, take ~3-5 existing cells' per-topic outputs and apply the 5 `gap_size_metrics.py` variants post-hoc. If the rankings shift dramatically, the full sweep is worth doing. If they don't, we can claim "the headline ranking is robust to scoring choice" with much less compute.

This is roughly:
1. Pick 3 GIST cells (LPP d=374, CITATION_ERROR_LDA d=382, ANTI_CSP d=128) and 3 Gemma cells (LIP d=374, LPP d=384, CCLIP d=374).
2. For each cell's existing per-topic gaps, recompute the 5 `gap_size_metric` variants.
3. See how the cells' rankings shift under each metric.
4. If rankings are stable → write a short note in the paper and skip Tier 1's full sweep. If they shift → proceed with Tier 1.

~half-day of work; resolves the major Tier 1 question before committing 25-56h of compute.

## Migration on completion

When Tier 1 completes (validated or refuted):
- If validated and adopted: production code lives at `bookgap/gap_size_metrics.py` (with the new dispatch infrastructure); results doc at `docs/gap_algorithm_sweep_tier1_results.md`; this living doc updates its Tier 1 section to "shipped" but keeps Tier 2/3 sections.
- If refuted: move conclusions to `docs/dead_ends/gap-detection-methods/tier1-gap-scoring-uniform.md` with failure-and-diagnosis writeup. Strike Tier 1 from this doc.

Same migration logic for Tier 2 and Tier 3.
