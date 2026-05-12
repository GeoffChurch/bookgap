# Static-seed cohesion diagnostic — results (T11303)

**Run date:** 2026-05-12
**Scope:** 17 / 1702 / **T11303** (Bayesian Networks; `N_total_works = 20546`, `N_articles = 14126`, `N_books = 145`)
**Spec:** [`docs/superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-design.md`](superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-12-static-seed-cohesion-diagnostic.md`](superpowers/plans/2026-05-12-static-seed-cohesion-diagnostic.md)
**Driver:** `artifacts_scripts/gap_static_seed_diagnostic.py`
**Raw outputs:** `cache/static_seed_diagnostic/T11303/{decisions.json, section{1..5}/}`

T11303 was chosen as the clean small-topic diagnostic per the spec. Per-seed sweeps use the `k` grid `{5, 10, 20, 40, 80, 160, 320, 640, 1280}`. 219 seeds were stratified-sampled for sections 1–3 (145 book seeds, 50 random article seeds, 24 cohesive article seeds drawn from `artifacts/gap_article_as_seed/T11303__article_seed.csv`). Section 4 ran joint normalization on all 20546 seeded works. Section 5 sampled 200000 overlapping seed pairs.

---

## Headline recommendations

| Decision | Recommendation | Key number |
|---|---|---|
| §1 Surprise curve metric | **Use LLR.** Default to LLR-argmax for `k*`. | LLR floor-ratio > z-score floor-ratio in 145/219 = **66%** of seeds; argmax agrees in 67%; LLR wins 53/72 disagreements. |
| §2 Decay function | **DEFER.** Compare all three downstream; provisionally default to self-tuning Gaussian. | 0/219 seeds had all three pairwise WJ ≥ 0.85. Median WJ: Gauss↔Tri=0.36, Gauss↔Epa=0.48, Tri↔Epa=0.21. |
| §3 Static vs one re-centering | **Apply one re-centering step.** | Median Jaccard(static_topk, recentered_topk) = **0.527**, p25 = 0.431. About half of members shift. |
| §4 Joint-norm sufficiency | **Effective-size tail is sharp, but confluence-merge implementation does not scale.** Joint norm itself works; the per-article merge enumeration needs redesign. | 100% of top-50 clusters above 2× median `eff_size`; pair-cap (500k) **hit at article 2 of 14126** during confluence walk. Survivors: 20514 / 20546. |
| §5 Dedup metric | **Use Jaccard, not containment.** | **1111** nested-distinct candidates (J<0.5 ∧ C≥0.8) out of 200k pairs evaluated. Containment would merge tight-core into halo seeds. |

---

## Section 1 — Surprise curve shape (LLR vs z-score)

CSV: `cache/static_seed_diagnostic/T11303/section1/curves_summary.csv` ·
Figures: `section1/curves_{cohesive,random,book}.png`.

LLR and z-score argmax-`k` agree on 147/219 = 67% of seeds. Where they disagree, LLR produces the sharper peak-to-floor ratio in 53/72 cases. Overall LLR floor-ratio strictly exceeds z-score floor-ratio for 66% of seeds. At small `k` the null distribution of normalized kernel sums is right-skewed, which inflates the z-score's peak in a noisy way; LLR remains robust because it is invariant to the null's higher moments.

The `k_argmax_llr` distribution differs dramatically by seed kind, validating the stratification:

| kind | n | tight (k≤20) | diffuse (k≥320) |
|---|---|---|---|
| cohesive (known basins) | 24 | 18 (75%) | 5 (21%) |
| random article | 50 | 20 (40%) | 27 (54%) |
| book | 145 | 3 (2%) | 100 (69%) |

Books pile up at `k = 1280` (the grid's max — 72/145 saturate there) because most book seeds have no genuine tight neighborhood in article-space and the LLR curve continues to grow with `k` over the sampled range. This is a real signal: **most books are not natural cluster centers in T11303.** The few cohesive-seed saturations (4/24) are bookcase-sized cluster regions that the diagnostic's `k` grid is too small for; widening the grid to 2560 or 5120 would resolve that in production.

**Recommendation:** LLR-argmax for `k*`. Keep z-score available as a diagnostic overlay.

## Section 2 — Decay-function comparison

CSV: `cache/static_seed_diagnostic/T11303/section2/decay_summary.csv`.

Weighted Jaccard between the three decay families, computed at each seed's own `k*`:

| pair | mean | median | p25 | p75 | max |
|---|---|---|---|---|---|
| Gaussian ↔ triangular-in-rank | 0.434 | 0.358 | 0.328 | 0.446 | 0.921 |
| Gaussian ↔ Epanechnikov-in-dist | 0.495 | 0.483 | 0.429 | 0.549 | 0.869 |
| triangular ↔ Epanechnikov | 0.262 | 0.210 | 0.175 | 0.258 | 0.797 |

**0/219 seeds** had all three pairwise WJ ≥ 0.85 (the spec's pre-registered "agreement" threshold). Only 10/219 had any pair clear it. The three decay shapes produce meaningfully different weight vectors at the chosen `k*`. The Gaussian↔Epanechnikov pair (both distance-based, just different tail shapes) is the most agreeable; triangular-in-rank disagrees with both of the distance-based decays because its weights ignore distance scale entirely.

**Recommendation:** DEFER. Downstream prototype should sweep the decay family alongside other ablations rather than commit. Provisional default: self-tuning Gaussian, consistent with `docs/cohesion_design_axes.md` axis G ("kernel weighting"), which has held up across T10181 / T11303 / T13734 cohesion-analysis work.

## Section 3 — Static vs one-recentering

CSV: `cache/static_seed_diagnostic/T11303/section3/recenter_summary.csv`.

Jaccard between the static top-`k*` and the top-`k*` recomputed once around the kernel-weighted mean of the initial members:

| kind | n | mean | median | p25 | p75 |
|---|---|---|---|---|---|
| book | 145 | 0.497 | 0.495 | 0.426 | 0.576 |
| cohesive | 24 | 0.796 | **0.952** | 0.667 | 1.000 |
| random | 50 | 0.661 | 0.606 | 0.450 | 1.000 |
| **overall** | 219 | 0.555 | **0.527** | 0.431 | 0.667 |

About half of members change after one re-centering step for book seeds and random article seeds — exactly the populations where the static seed point is unlikely to coincide with the true local density center. Cohesive seeds (already at the centroid of a known basin) are largely stable, as expected. This is consistent with axis C in `docs/cohesion_design_axes.md`, where moving-centroid beat fixed-centroid on offset K-balls; the same drift correction is doing real work here.

**Recommendation:** Apply one re-centering step. One step is cheap (one extra `D_seed-to-articles` kernel row plus a top-`k*` selection) and the membership shift is substantial. Whether *iterated* re-centering pays for itself is left to the downstream prototype.

## Section 4 — Joint normalization and confluence merge

CSV: `cache/static_seed_diagnostic/T11303/section4/top20_by_eff.csv` ·
Figure: `section4/eff_size_hist.png`.

Joint normalization ran on all 20546 seed clusters (every article + every book + book-chapter + preprint in T11303 — `setup()` builds `D_art_NN_all` for every row of `state["sigma"]`, and `_compute_k_star` uses all of them, while membership only counts the 14126 articles).

**Effective-size tail is sharp.** Median effective size = 0.568. **100% of the top-50 clusters have `eff_size` ≥ 2× median.** All top-20 clusters are article-seeded at `k* = 1280` (`eff_size` ∈ [4.01, 5.15], LLR ≈ 5.3). The joint-normalization step is doing its job: redundant clusters self-penalize via shared mass; non-redundant high-LLR clusters retain their effective size.

**Confluence merge does NOT scale on this dataset.** The pair-enumeration cap (500000 candidate pairs) was **exhausted at article index 2 of 14126** during the per-article walk. Diagnosis: a small number of hyper-dense articles are claimed by ≈1000 seeds each, and a single article claimed by 1000 seeds contributes `C(1000, 2) ≈ 5 × 10⁵` candidate pairs by itself. The per-article inner-loop strategy is the wrong shape for this workload — most of the budget is consumed by a handful of articles before the algorithm ever reaches the long tail. Only 32 pairs were actually unioned before the cap fired, and 20514 of 20546 seeds survived (i.e. essentially nothing was merged).

**Recommendation for production:** replace per-article enumeration with a sparse matrix-multiplication-based pair finder. With `M` the (seeds × articles) membership matrix (rows are clusters, nonzero entries are kernel weights), the symmetric Jaccard numerator and denominators come straight from `M @ M.T` and the row L1 norms. SciPy sparse handles this in one pass and finds all globally-overlapping pairs at once, without re-enumerating the same pair multiple times via different articles. The joint-normalization recommendation itself is unchanged — the tail is exactly the shape we wanted.

## Section 5 — Jaccard vs containment

CSV: `cache/static_seed_diagnostic/T11303/section5/pair_metrics.csv` ·
Figure: `section5/jaccard_vs_containment.png`.

Across 200000 sampled overlapping seed pairs, **1111 pairs** satisfy `J < 0.5 ∧ C ≥ 0.8` (the spec's "nested-distinct" criterion: low symmetric overlap but one cluster is mostly contained in the other). Distribution by larger cluster's `k_max`:

| `k_max` | nested-pair count |
|---|---|
| 160 | 2 |
| 320 | 7 |
| 640 | 54 |
| 1280 | **1048** |

The smaller side spans the full `k` range (`k_min` ∈ {40, 80, 160, 320, 640}); the larger side is `k = 1280` in 1048 / 1111 = **94%** of nested pairs. Pattern: a tight `k ≤ 640` core is largely contained inside a diffuse `k = 1280` halo. Containment would merge these (since it ignores the size mismatch); Jaccard correctly identifies them as distinct because the halo carries 2–10× as many members. The pre-registered failure mode of containment is empirically real on this dataset.

For context: 93 pairs had `J ≥ 0.5 ∧ C ≥ 0.8` (true duplicate clusters that both metrics agree to merge). The Jaccard-only rule keeps the 1111 nested-distinct pairs separated; containment-only would over-merge them.

**Recommendation:** Jaccard for confluence-merge (no change from the pre-registered default).

---

## Implications for the static-seed pipeline design

1. **LLR surprise curve, self-tuning Gaussian decay, one re-centering step, Jaccard-merge.** Four of the five design choices are settled. Decay function is the only deferred choice; downstream prototype should compare all three but can start with Gaussian.
2. **Confluence merge needs `M.T @ M`, not per-article walks.** The current per-article inner-loop is fundamentally the wrong shape: pair-budget is consumed by hyper-dense articles long before the long tail is reached. A sparse matrix-multiplication-based pair finder (intersection sizes from `M @ M.T`, row-norms from `M.sum(axis=1)`) gets all globally-overlapping pairs in a single pass with no enumeration redundancy. This is a precondition for joint-normalization to operate at the right scale.
3. **The `k` grid needs to extend past 1280 for production.** 46% of section-1 seeds (almost entirely books and random articles, but also 4/24 cohesive) saturate at the current top. Most of these are spurious (books are not natural cluster centers) but at least some are genuine large basins. Extend to 2560 / 5120 and let the LLR curve actually peak.
4. **Books-as-seeds need a filter.** 69% of book seeds saturate at `k = 1280` with no real interior structure (median Jaccard 0.495 in section 3 — nearly half their members reshuffle on one re-centering step, indicating no stable local center). The pipeline can still anchor coverage readouts on books, but seeding from books for cluster-creation is mostly noise on T11303. A pre-filter (e.g. drop book seeds whose `k = 5` neighborhood contains fewer than `min_articles_near_book` articles) would be cheap and remove much of the diffuse-saturation noise.
5. **The effective-size readout itself is healthy.** Top-50 are all > 2× median, all article-seeded, all at `k* = 1280`. Joint normalization correctly concentrates mass on real clusters. The bottleneck for a production prototype is now (a) confluence-merge scaling and (b) deciding what to do with the diffuse-saturated seed tail.

## Open questions / follow-ups

- Does a wider `k` grid (≤ 5120) move the LLR argmax for the saturated cohesive seeds (the 4/24 at `k = 1280`)? If yes, the small-grid is silently truncating genuine large clusters.
- Does the sparse-matrix confluence merge change the survivor count meaningfully when fully run, or is T11303 genuinely already-merge-free post joint-norm? (32 unioned out of 20546 seeded is consistent with "almost nothing to merge"; we want this confirmed without the cap exhausted.)
- Is iterated re-centering (run to fixed point) worth it on top of the one-step re-centering recommendation? On cohesive seeds the gain is bounded above by p75 = 1.0 vs median 0.95 — small. On book seeds the static-vs-static-recentered diff is ~50%, so additional iterations might still move membership. Cheap to test in the prototype.
- Decay-function comparison should be measured in *downstream* terms (effective-size tail sharpness, book-coverage readout stability) rather than weight-vector WJ alone. WJ at fixed `k*` is sensitive to weight scaling but does not directly tell us whether the *clustering* differs.
