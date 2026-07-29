# Curriculum clustering method comparison

> **Note — tooling removed.** The `artifacts_scripts/curriculum/` modules referenced in this doc were deleted in `0f2d0ad`; findings stand, commands do not. See `docs/curriculum-pipeline-removed.md`.

(Migrated from `artifacts_scripts/curriculum/TODO.md` 2026-04-28.)

This doc records the cross-corpus comparison of clustering methods used by the curriculum pipeline. The full method-implementation history (shipped dates, code paths) is preserved in git log; this doc captures the empirical results that informed the default method choice.

## Current verdict (2026-04-28)

**Reframe**: seeds (target/source works) are *fair game* — they don't need to remain as singletons; getting clustered with related works is fine, getting dropped to -1 is fine. What matters is: cluster topical coherence, keyword distinctness, mass distribution, and whether earned non-seed landmarks (Shannon-MTC-class works) emerge.

## Cross-corpus rankings

From `compare_clusterings.py` (QC: quantum-crypto, 0 sources; Robotics: vision→robotics, 4 sources):

| Method | QC Q | Rob Q | QC tpc/kw | Rob tpc/kw | Singletons | Notes |
|---|---|---|---|---|---|---|
| leiden | **0.759** | **0.638** | 1.41 | 1.80 | **0** (refuses) | Citation-graph champion when cleanly producing ≥ K real communities. |
| **multi_view_spectral** (CSLS + KRD-2011, λ auto-tuned) | **0.691** | 0.484 | 1.42 | 1.68 | 0 (refuses) | **Best cross-corpus exact-K alternative to leiden.** No mega-cluster pathology (gini 0.27/0.32, max% 15%/17%); all 15 clusters substantive on both corpora. Beats wbk on robotics by Q +0.175. |
| **weighted_bisecting_kmeans** | 0.384 | 0.309 | **1.54** | **1.32** | 0–4 (knob) | **Best earned-singleton-supporting method.** Best keyword distinctness on both corpora; surfaces non-seed landmarks. |
| WU (citation_avg_linkage, weighted_unified) | 0.320 | 0.287 | 1.18 | 1.27 | 5 | Close to wbk on QC; behind on robotics. |
| binary (citation_avg_linkage) | 0.258 | 0.169 | 1.09 | 1.15 | 5–7 | Dominated by WU on QC, by wbk on robotics. |
| WPD (citation_avg_linkage, weighted_per_direction) | 0.248 | 0.176 | 1.18 | 1.12 | 5–8 | Most singletons but lowest Q among citation_avg variants. |
| average (avg-linkage on embeddings) | 0.378 | 0.131 | 1.24 | 1.09 | 5–8 | Cross-corpus unstable — worst Q on robotics. |
| spectral | 0.548 | 0.547 | 1.18 | 1.21 | 0 | Mega-cluster pathology on QC (max%=57%). |
| graph_bisect | 0.215 | 0.569 | 1.88 | 1.67 | 0 | Cross-corpus unstable — slices through real communities on QC. |

**Default**: `linkage_method = "weighted_bisecting_kmeans"` (`config.py:499`). Cross-corpus rationale confirmed by full sweep on 2026-04-28; retains the default because it's the only top-tier method that produces earned non-seed singletons (Shannon-MTC-class landmarks).

## When to deviate from wbk

- **leiden** if you want pure citation-driven communities, don't care about earned landmarks, and the corpus produces ≥ K real communities. Highest Q when it works.
- **multi_view_spectral** if you want exact-K with no mega-cluster pathology and balanced sizes, especially when leiden under-produces communities (it only found 9 real ones on robotics, padding with empties; multi_view_spectral produced 15 substantive ones). Best Q-vs-stability trade-off among non-singleton-allowing methods.
- **WU** if you specifically want citation-graph-driven clustering *with* the chance of earned non-seed singletons (it found 3 such on QC: EPR, Algebraic Alphabet, Bell-theorem QKD).

## Avoid

- **average**: cross-corpus instability
- **graph_bisect**: cross-corpus instability + slicing real communities
- **WPD**: no consistent advantage over WU
- **spectral**: mega-cluster pathology

## Multi-view spectral verdict (2026-04-28, complete)

Shipped with CSLS-corrected views and op-norm-based λ auto-tune (`λ = 1.5 · op_norm_per_K`). Did not dethrone leiden as the citation-graph champion (Q delta −0.07 on QC, −0.15 on robotics) and did not surface earned singletons (it's exact-K by construction). But: substantially beats wbk on robotics (Q +0.175) with cleaner cluster-size balance, and is the only method that produces 15 substantive clusters on both corpora without singletons or mega-cluster pathology.

**Net**: a compelling third option in the table above, not a default replacement.

### Cross-corpus probe (lambda auto-tune, 2026-04-28)

From `artifacts_scripts/curriculum/lambda_probe.py`:

| Corpus | n | K | Empirical optimum λ | Q at optimum | Q at λ=0 | Auto-tuned λ | Δ Q from optimum |
|---|---|---|---|---|---|---|---|
| QC quantum-crypto | 5000 | 15 | 0.100 | 0.693 | 0.206 | 0.097 | −0.019 |
| Robotics vision→robotics | 5000 | 15 | 0.100 | 0.485 | 0.049 | 0.095 | **−0.002** |

Both corpora hit the empirical optimum at λ=0.1 *exactly* and the sweet spot is broad (Q ≥ 95% of peak across λ ∈ [0.03, 0.3]). The op-norm-based auto-tune lands in the sweet spot on both corpora.

### Heuristic shootout (cross-corpus, |Q-delta| from empirical optimum, smaller is better)

| Heuristic | QC Δ Q | Robotics Δ Q | Status |
|---|---|---|---|
| **op_norm_per_K** | −0.019 | −0.002 | **Winner; shipped as auto-tune default** |
| trace_per_n | −0.023 | −0.016 | Solid runner-up |
| gap_K | −0.487 | −0.436 | Degenerate (collapses to ~0) |
| sqrt_gap_product | −0.487 | −0.436 | Degenerate |
| mean_bottom_K | invalid (negative) | invalid | Degenerate |
| Kth | invalid | invalid | Degenerate |

**Why bulk-spectrum heuristics dominate**: after CSLS+clip, both Laplacians are near-rank-deficient — the data decomposes into 15+ near-disconnected components, so the bottom-K eigenvalues are all at machine precision (~1e-16). Any heuristic indexed at the K-th eigenvalue or its gap collapses to numerical noise. Bulk quantities (trace, op_norm) are smooth in the matrix entries and stay informative.

## min_cluster_weight at scale (2026-04-29)

Run on robotics with `min_cluster_weight=0.033` = 0.5 × mean topic mass; numbers verified via `compare_clusterings.py`:

| metric | default wbk | mcw=0.033 | Δ |
|---|---|---|---|
| Q | 0.309 | **0.336** | +0.027 |
| flow_capture | 0.396 | 0.429 | +0.033 |
| gini | 0.57 | 0.54 | −0.03 |
| max% | 21% | 20% | −1pp |
| #fragments (size ≤ 20) | 6 | 5 | −1 |
| #singletons | 5 | 4 | −1 |
| #clusters ≥ 50 | 9 | 10 | +1 |
| tpc/kw | 1.32 | 1.35 | +0.03 (worse) |

Singletons are the expected mass-promoted seeds: Topic 6 (mass 0.336, projective_geometry/projection_matrices = Hartley & Zisserman), Topic 7 (mass 0.152, chapters/image_processing = Forsyth & Ponce), Topic 8 (mass 0.113), Topic 14 (mass 0.050). Other clusters topical-coherent (general CV / SLAM / automated driving / people tracking / visual SLAM).

**Recommended threshold**: `0.5 × (1 / desired_n_topics)` ≈ 0.033 for K=15. Available as `min_cluster_weight` on `CurriculumConfig`.

## Gap detection — partial sweep results (2026-04-29)

Ran probe sweeps on subfield 17/1702 = CS/AI subfield, 1.96M works, 77 topics. Aggregator at `artifacts_scripts/probe_sweep_summary.py`. Matrix: 5 denoisers × 4 drop_k = 20 configs per probe + 1 K-ball Cloud baseline = 41 runs. Total wall-clock: ~5 hours.

### HDBSCAN cluster-depletion (subfield 17/1702 with default `min_cluster_size=50`)

| rank | config | sig_eff | qual | #clust |
|---|---|---|---|---|
| 1 | **pca drop_k=100** | **0.092** | 3 | 4 |
| 2 | cclip drop_k=100 | 0.074 | 4 | 5 |
| 3 | wc_cclip drop_k=100 | 0.073 | 4 | 5 |
| 4 | multi_lda drop_k=100 | 0.071 | 2 | 2 |
| 5 | cclip drop_k=20 | 0.063 | 3 | 4 |
| ... | (cclip/wc_cclip drop_k=5/20 ≈ 0.056) | | | |
| bottom | rnojd drop_k=20/100 | 0.026-0.028 | 3 | 3-4 |

- **PCA drop_k=100 wins by a meaningful margin**. Class-aware denoisers (cclip family) preserve between-class structure that's exactly what we want HDBSCAN to find at this scope — counterproductive.
- **WC_CCLIP doesn't beat CCLIP** despite SNR advantage; statistically indistinguishable on this metric (the SNR ranking measures general embedding quality, not gap-detection specifically).
- **rNOJD aggressive (drop_k=100) is the weakest** — joint diagonalisation removes too many gap-relevant directions.
- **FDR signal is uniformly thin**: every config produces exactly 1 FDR-significant cluster at α=0.05 (1 of 20 produces 0). HDBSCAN at subfield scale finds "the one biggest book-depleted region" rather than many — wrong tool for LLM-pipeline gap candidates.

### K-ball gap detection (same scope)

- All 20 denoiser configs found 380 gaps each (5 × 76 topics with 1 degenerate).
- Mean gap_size_percentile ~0.98 across all configs (extreme outliers vs book null).
- Gap-size pattern: cclip/wc_cclip and pca-drop_k=100 produce tighter gaps (mean 237-245); pca-drop_k=1, multi_lda, rnojd produce wider (mean 317-326). Cloud baseline (no denoiser) at 284.
- **Denoiser choice has small but visible effects on gap structure but doesn't dramatically change which gaps are found**. Geometric outliers persist under most projections.

### Net practical verdict

For LLM-pipeline gap candidates, K-ball with any reasonable denoiser is the right tool. PCA at drop_k=100 is the most defensible HDBSCAN config but the signal is thin enough that it's mostly useful as a confirmatory diagnostic, not a primary gap source.

**Caveats**: results are for subfield 17/1702 only (CS/AI/NLP); cross-scope replication would strengthen claims. HDBSCAN's `min_cluster_size=50` default is high for this scope — tuning down may surface more clusters and change the picture.

## USearch evaluation as faiss-gpu replacement (2026-04-29)

Evaluated via `artifacts_scripts/usearch_smoke_test.py`. **Verdict: NOT a path to GPU range_search.**

- `pip install usearch` does NOT bundle GPU support. Their GPU path requires cuVS (RAPIDS-scale install with CUDA toolkit).
- **No native range_search API**. Index only has `search()` (kNN, with `exact=True` for brute-force) and `count()`. Range search must be emulated via kNN + post-filter — same constraint as faiss-gpu.
- CPU performance ~25% slower than FAISS HNSW on 200K × 384 (the advertised 10-100× speedup doesn't materialise at our scale).

USearch loses on every axis we cared about. The K-ball brute-force GPU refactor (see backlog) is the better path.
