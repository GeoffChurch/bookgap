# Static-seed cohesion diagnostic — Phase 2 results (T11303 + T10181)

**Run date:** 2026-05-12
**Topics:** T11303 (Bayesian Networks; `N_total=20546`, `N_articles=14126`, `N_books=145`) and T10181 (NLP; `N_total=111649`, `N_articles=88460`, `N_books=1252`)
**Spec:** [`docs/superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-phase2-design.md`](superpowers/specs/2026-05-12-static-seed-cohesion-diagnostic-phase2-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-12-static-seed-cohesion-diagnostic-phase2.md`](superpowers/plans/2026-05-12-static-seed-cohesion-diagnostic-phase2.md)
**Phase 1 results:** [`docs/static_seed_cohesion_diagnostic.md`](static_seed_cohesion_diagnostic.md)
**Driver:** `artifacts_scripts/gap_static_seed_diagnostic.py`
**Raw outputs:** `cache/static_seed_diagnostic/{T11303,T10181}/{decisions.json, section{1..6}/}`

Phase 2 set out to (a) replace Phase 1's pair-cap-bound confluence merge with a real one and (b) ask whether dedup *survivors* still need recentering. Both runs completed. The implementation works at scale (T10181 has 1.67 × 10⁹ overlapping pairs from `M.T @ M`); the headline empirical finding is that the pre-registered Ruzicka ≥ 0.9 confluence threshold essentially never fires on this data structure, so §4's "real survivor count" is the seed count and §6's hypothesis is left unresolved by absence of dedup, not refuted.

---

## Headline answers

| Question | T11303 | T10181 |
|---|---|---|
| §4 joint-norm sufficient (real dedup this time)? | **No** — 19983 / 20546 survive (97.3%), 563 pairs unioned out of 1.07 × 10⁸ candidates | **No** — 109317 / 111649 survive (97.9%), 2332 pairs unioned out of 1.67 × 10⁹ candidates |
| §6 post-dedup recentering still needed? | **Inconclusive** — survivor set ≈ seed set, so §6 ≈ §3 (medians 0.546 vs 0.527) | **Inconclusive** — same reason (medians 0.486 vs 0.441) |
| §1 LLR vs z-score | LLR (66% win rate, unchanged from Phase 1) | LLR (76% win rate, new) |
| §2 decay-function choice | DEFER (0/219 all-pairs WJ ≥ 0.85) | DEFER (0/1326) |
| §3 static vs one-recentering | one-recentering (median Jaccard 0.527) | one-recentering (median 0.441) |
| §5 dedup metric | Jaccard (1111 nested-distinct pairs / 200k sampled) | Jaccard (155 / 200k — proportionally fewer at this scale) |

---

## What changed from Phase 1

**Algorithmic change.** `_confluence_components` is now a one-shot pair finder: `overlap = (M.T @ M)`, then iterate the upper-triangle nonzero pattern once. The Phase 1 per-article walk had `O(N_overlapping_pairs)` seen-set memory and was burning all of a 500k pair budget on the first two of 14126 articles. The new path has no seen-set, no cap, no enumeration redundancy.

**Two perf commits during execution** (not in the spec, but real prod code now on `main`):

1. `bcec753` (`perf: JIT confluence merge with numba`) — the per-pair Python loop ran at ~4k pairs/sec, which would take ~7 h on T11303 (107M pairs). The Ruzicka kernel and union-find are now compiled @njit over the CSC column buffers.
2. `a7f250f` (`perf: parallelize confluence Jaccard pass`) — T10181's 1.67 × 10⁹ pairs would still take ~2.5 h sequentially. Split into a parallel `prange` Ruzicka pass (`_jaccard_passes_threshold_numba(parallel=True)`, emits `bool[N_pairs]`) and a sequential path-compressed union-find restricted to the True indices (`_union_find_from_passing_pairs_numba`). T10181's §4 ran in ~25 min on 16 active cores.

Both commits are required to run the diagnostic at all on T10181's scale; without them the spec's "5–60 minute" estimate would have been a ~3 hour estimate. HNSW would not have helped §6 — the recenter step uses one FlatL2 search per survivor, BLAS-batched, and saturates compute already.

**Section 6** (new): same Jaccard(static_topk, recentered_topk) test as §3, restricted to the §4 survivor list. The hypothesis (spec §"New §6"): survivors are highest-LLR-per-confluence-component, so they should be near their local-density peaks ⇒ §6 median should be much higher than §3's.

---

## Section 4 — joint normalization and confluence merge

CSVs: `cache/static_seed_diagnostic/{T11303,T10181}/section4/top20_by_eff.csv` · Figures: `section4/eff_size_hist.png` per topic.

|  | T11303 (Phase 1) | T11303 (Phase 2) | T10181 (Phase 2) |
|---|---|---|---|
| `n_seeds_before_merge` | 20546 | 20546 | 111649 |
| `n_survivors_after_merge` | 20514 (cap exhausted) | **19983** | **109317** |
| Survival rate | 99.84% (artifact) | 97.26% | 97.91% |
| `median_effective_size` | 0.568 | 0.568 | 0.675 |
| `pct_top50_above_2x_median` | 1.0 | 1.0 | 1.0 |
| `M.T @ M` overlapping pairs | n/a | 1.07 × 10⁸ | 1.67 × 10⁹ |
| Pairs unioned at Ruzicka ≥ 0.9 | 32 (capped) | 563 | 2332 |
| Union rate | n/a | 5.3 × 10⁻⁶ | 1.4 × 10⁻⁶ |

**The Phase 1 estimate was qualitatively correct.** Phase 1 reported 20514 survivors with the cap exhausted and 32 unions; Phase 2 reports 19983 survivors with the merge run to completion — a difference of 531 additional unions across the long tail the cap had cut off. The Phase 1 verdict "joint_norm_sufficient: false (implementation-bound)" was actually data-bound, just not for the reason Phase 1 thought: the bound is the threshold, not the cap.

**Effective-size tail is unchanged.** Median `eff_size = 0.568` on T11303 (identical to Phase 1) and 0.675 on T10181, with 100% of top-50 above 2× median on both topics. All 40 top-20 entries (20 per topic) are article-seeded at `k* = 1280`. Joint normalization is doing the work it was supposed to; the rank order matches Phase 1 qualitatively. T10181's top-20 reaches `eff_size = 8.47` vs T11303's 5.15 — bigger topic, more room for sharp basins.

**Confluence dedup essentially doesn't fire.** This is the negative finding. At Ruzicka ≥ 0.9, almost no overlapping pair satisfies the threshold. The cause is structural: two seeds whose argmax-LLR `k*` falls at different grid points (e.g., one at `k=80`, one at `k=1280`) have weight vectors of very different L1 mass; their Ruzicka `sum(min)/sum(max)` is bounded above by `min(L1_a, L1_b)/max(L1_a, L1_b)`, which is small for size-mismatched pairs. Even when `k*` matches, the local-bandwidth `σ` from `D_art_NN_all` differs across seeds; in the joint-normalized matrix the per-article column-sum redistribution further drives weight vectors apart. §5's pair sample makes the same point in a different metric: median pairwise Jaccard between two same-topic seeds is 0.13 on T11303 and 0.04 on T10181.

**Recommendation:** the `M.T @ M` machinery is sound and the eff-size tail is healthy; the **Ruzicka ≥ 0.9 threshold is wrong for this data structure**. A production confluence merge needs either (a) a much lower threshold (0.5–0.7 — with §5 still vindicating Jaccard over containment at that range), (b) a different similarity (cosine between argmax-support indicator vectors; weighted-overlap Louvain on the `M.T @ M` graph), or (c) a different shape entirely — joint-norm + confluence-merge as a two-step pipeline may not be the right factorization, since the joint-norm step produces a sharp eff-size tail but the dedup at threshold ≥ 0.9 doesn't reduce the survivor population usefully.

---

## Section 6 — post-dedup recentering (new)

CSV: `cache/static_seed_diagnostic/{T11303,T10181}/section6/recenter_survivors_summary.csv`.

| | T11303 §3 (all seeds) | T11303 §6 (survivors) | T10181 §3 (all seeds) | T10181 §6 (survivors) |
|---|---|---|---|---|
| `n_seeds` | 219 | 19983 | 1326 | 109317 |
| median Jaccard | 0.527 | **0.546** | 0.441 | **0.486** |
| p25 Jaccard | 0.431 | 0.412 | 0.338 | 0.354 |
| p75 Jaccard | 0.667 | 0.712 | 0.539 | 0.739 |
| Decision | one-recentering | one-recentering | one-recentering | one-recentering |

**The hypothesis cannot be tested on these runs.** §6 was designed to compare the survivor distribution against the all-seed distribution. With survival rates of 97–98%, the §6 input set is essentially the §3 input set (modulo the kind imbalance: §3 stratifies across book/random/cohesive, §6 inherits the natural mix — 99.3% article, 0.7% book on T11303; 98.9% article, 1.1% book on T10181). The §6 → §3 median delta is small and *positive* on both topics (+0.018 on T11303, +0.044 on T10181) — which is consistent with the hypothesis direction (survivors marginally closer to centroids) but the magnitude is far below the test threshold (median ≥ 0.90 to skip recentering, p25 ≥ 0.80).

The slightly larger delta on T10181 is plausible mechanistically: T10181 had more merging in absolute terms (2332 unions vs 563), so a few more survivors are post-dedup "winners" relative to merged-away losers. But it's a 4-point delta on a population where the gap-to-threshold is 41 points. The dedup at this threshold is not selective enough to test the hypothesis.

**Per-kind §6 numbers** (median Jaccard) on T11303: article 0.546, book 0.495. On T10181: article 0.488, book 0.433. Books still recenter substantially, consistent with Phase 1's §3 per-kind breakdown — most book seeds are not at local-density peaks, and confluence dedup at Ruzicka ≥ 0.9 doesn't change which book seeds we're looking at.

**Recommendation:** §6 stays as a diagnostic but its hypothesis is **unresolved**, not refuted. A meaningful test requires a confluence criterion that materially shrinks the survivor population (target: ~10²–10³ survivors out of 10⁵ seeds). This is a Phase 3 question; see *Open questions* below.

---

## Sections 1 / 3 / 5 — sanity checks on both topics

**§1 (LLR vs z-score).** LLR wins both topics: 66% on T11303 (unchanged from Phase 1) and 76% on T10181 (new). The win rate is concentrated on book seeds on both topics (77%/78%), consistent with z-score's known sensitivity to the null's higher moments at small `k`. Cohesive-seed agreement is higher on T10181 (50%) than T11303 (29%) — likely because T10181's cohesive seeds saturate `k_argmax_llr = 1280` for only 5/24 of them (21%) and the curves have proper peaks. Saturation breakdown by kind: T11303 has 4/24 cohesive, 24/50 random, 72/145 book saturated at `k = 1280`; T10181 has 5/24 cohesive, 20/50 random, **774/1252 (62%) book** saturated. T10181's book-seed population is even less center-like than T11303's, reinforcing Phase 1 §"Books-as-seeds need a filter."

**§3 (one-recentering).** Per-kind median Jaccard on T10181: cohesive 0.667, random 0.616, book 0.433. Cohesive seeds recenter less than on T11303 (0.952 → 0.667) — bigger topic, more spread-out neighborhoods, less stability under one recentering. Books shift more on T10181 (0.433 vs T11303's 0.495). Recommendation stands on both topics.

**§5 (Jaccard vs containment).** Per the pre-registered "nested-distinct" criterion (`J < 0.5 ∧ C ≥ 0.8`): T11303 has 1111 / 200000 = 0.56%; T10181 has 155 / 200000 = 0.08%. Nested-distinct pairs are proportionally less common at T10181 scale, plausibly because the median pairwise Jaccard is 0.04 on T10181 vs 0.13 on T11303 — most sampled pairs are too far apart in raw overlap for *either* metric to call them similar. The containment-vs-Jaccard distinction remains the right call on both topics; the pre-registered failure mode of containment (merging tight cores into diffuse halos) is still concretely visible in T11303's 1048 / 1111 nested-distinct pairs where `k_max = 1280`. Recommendation unchanged.

---

## Implications for the static-seed pipeline design

1. **The `M.T @ M` machinery is production-ready.** 107M pairs (T11303) and 1.67 × 10⁹ pairs (T10181) both fit in memory and finish in minutes (parallel numba). The Phase 1 implementation concern about confluence-merge scaling is fully resolved. Production code can use this exact path.
2. **Ruzicka ≥ 0.9 confluence is *not* a useful dedup criterion on jointly-normalized weight vectors.** Survival rates of 97–98% on both topics mean the threshold passes ~no pairs; the few hundred unions that do happen don't change the eff-size tail or the §6 distribution materially. **The threshold question (or the metric question) is the gating issue for any "joint-norm → confluence-merge" pipeline.**
3. **Four Phase 1 recommendations stand on both topics.** LLR for surprise (§1), one-recentering for centers (§3), Jaccard not containment for dedup (§5), DEFER on decay-function comparison (§2). T10181 reproduces all four. These are settled.
4. **The §6 hypothesis ("dedup survivors are well-centered, so recentering is redundant") is unresolved.** Phase 2 cannot answer it at the chosen Ruzicka threshold because dedup is essentially inactive. A real test needs a confluence criterion that brings survivor count down by 2+ orders of magnitude. Until then, **assume the production pipeline includes the recentering step** (§3's verdict, not §6's).
5. **Books-as-seeds remain noisy on both topics.** T10181 saturates 62% of book seeds at `k = 1280` (vs 50% on T11303). The Phase 1 recommendation — pre-filter book seeds by article-density in their `k = 5` neighborhood — is still the right move and is now reinforced by T10181 data.

---

## Open questions / Phase 3 candidates

- **Threshold/metric sweep for confluence merge.** What does the survivor distribution look like at Ruzicka ∈ {0.3, 0.5, 0.7}? Does Jaccard on argmax-support indicator vectors (binarized membership) produce a more useful dedup signal than Ruzicka on weighted membership? The §5 sample already gives a free preview: at J ≥ 0.5 on the same M.T @ M structure, 1111 + ~few-thousand pairs cross the bar on T11303 — i.e., dedup at J ≥ 0.5 would merge thousands, not 563, of pairs. That's the right scale for a meaningful §6 test.
- **Alternative confluence formulations.** Louvain or label-propagation on the weighted `M.T @ M` graph treats this as a clustering problem rather than a thresholded-pair problem. Survivor count comes from the community count, not from a per-pair decision. Worth a side-by-side with thresholded Ruzicka.
- **Section 6 is the test, the threshold is the variable.** Phase 3 should restructure: sweep the confluence threshold/criterion as the primary axis, report §4 survivor count *and* §6 median Jaccard *and* §6 → §3 delta at each setting. The "right" criterion is the one where §6 median crosses 0.85 with survivor count still in a useful range (say, ≥ 10² survivors).
- **Iterated recentering.** Open since Phase 1; Phase 2 has nothing new to add on it. Defer until the confluence-criterion question is settled.
- **T10181 templated-leak vs §1 cohesive saturation.** 5 / 24 cohesive seeds saturate `k = 1280`; on T11303 it was 4 / 24. Without a leak filter we can't tell whether T10181's cohesive saturations are real large basins or templated-leak basins masquerading as cohesive. Same diagnostic-vs-leak boundary as Phase 1.

---

*Naming note:* the diagnostic implements `sum(min)/sum(max)` over jointly-normalized non-negative weight vectors; this is **Ruzicka similarity**, the continuous-data generalization of Jaccard. Phase 1 and the spec call it "weighted Jaccard" or "generalized Jaccard" interchangeably; the script's function name is `_sparse_weighted_jaccard`. Same operation either way.
