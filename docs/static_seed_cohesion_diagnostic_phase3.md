# K-ball + LLR refinement — Phase 3 results (T11303 + T10181)

**Run date:** 2026-05-12
**Topics:** T11303 (Bayesian Networks, ~14k articles, 145 books) and T10181 (NLP, ~88k articles, 1252 books)
**Spec:** [`docs/superpowers/specs/2026-05-12-kball-llr-refinement-design.md`](superpowers/specs/2026-05-12-kball-llr-refinement-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-12-kball-llr-refinement.md`](superpowers/plans/2026-05-12-kball-llr-refinement.md)
**Phase 1 / 2:** [`static_seed_cohesion_diagnostic.md`](static_seed_cohesion_diagnostic.md), [`static_seed_cohesion_diagnostic_phase2.md`](static_seed_cohesion_diagnostic_phase2.md)
**Driver:** `artifacts_scripts/gap_kball_llr_refine.py`
**Outputs:** `artifacts/gap_kball_llr_refine/{T11303,T10181}__llr_refined.csv`; figures in `cache/gap_kball_llr_refine/{T11303,T10181}/`

## Headline numbers

| | T11303 | T10181 |
|---|---|---|
| Corpus size (N_total / N_articles / N_books) | 20,546 / 14,126 / 145 | 111,649 / 88,460 / 1,252 |
| Input K-ball gaps | 200 | 200 |
| Refined survivors after dedup (J ≥ 0.5) | 142 | 180 |
| Median `kball_size_delta` (all 200) | +33.5 | +29.0 |
| Mean `kball_size_delta` | +285.7 | +138.1 |
| `kball_size_delta` sign split (grow / shrink / same) | 175 / 19 / 6 | 149 / 48 / 3 |
| Median `kball_jaccard` (K-ball ↔ refined membership) | 0.255 | 0.379 |
| Q1 / Q3 `kball_jaccard` | 0.047 / 0.542 | 0.182 / 0.560 |
| Refined `k*` saturating at 1280 (all 200) | 33 | 10 |
| Refined `k*` saturating at 1280 (survivors only) | 4 / 142 | 9 / 180 |
| Wall time (setup + 200 gaps) | ~61 s | ~9 min 23 s |

## How K-ball gaps change under LLR refinement

LLR refinement is **not** a minor adjustment: across both topics the median refined cluster is **larger** than the K-ball seed (median delta +33 on T11303, +29 on T10181), and the membership overlap is **modest** (median Jaccard 0.255 / 0.379). The size-delta histograms (`cache/gap_kball_llr_refine/{topic}/size_delta_hist.png`) are right-skewed: most gaps grow by tens-to-hundreds of articles, but a long tail saturates the K_GRID at 1280. On T11303 the mean is dragged way up by 33 gaps that saturate at k\*=1280 (these are mostly the matrioshka-eating losers that get merged out in dedup — only 4 survive). On T10181 the corresponding tail is much shorter (10 of 200, 9 of 180 survivors), consistent with the larger corpus having more genuine internal structure for LLR to lock onto at smaller k\*.

This matches Phase 1-2: LLR-vs-k consistently picks **larger** clusters than the K-ball radius-fraction heuristic, because the corpus null log-mass drops faster than the local kernel mass for almost any seed in a topically-coherent region. K-ball's `radius_fraction=0.75` is, in retrospect, conservative compared to what the surprise framework says is the "right" size.

## Survivor count and dedup behavior

Survivor counts are **142/200 (71%)** on T11303 and **180/200 (90%)** on T10181. This is the inverse of what we saw in Phase 2's all-articles-as-seeds run (98% survival on T10181 because there were so many static seeds spread over the corpus that few overlapped meaningfully after refinement). Here, K-ball's greedy max-coverage already chose 200 seeds to be far apart, but the **LLR refinement then re-centers and grows each seed**, so distinct K-ball seeds can drift into the same neighborhood. T11303's smaller corpus has fewer distinct topical neighborhoods, so the LLR-grown clusters collide more frequently — hence 29% merge rate.

This is sanity-positive: the survivors are exactly the gaps whose refined membership doesn't collapse onto a larger neighbor. The dropped 58 gaps on T11303 are mostly the saturating-at-1280 ones (only 4 of 33 saturators survive), which is exactly the matrioshka-collapse story the spec called out — the LLR machinery's bias toward large k\* on noisy seeds gets cleaned up by dedup rather than poisoning the output.

## Top-5 survivors per topic

T11303 (Bayesian Networks):

| rank | kball_gap_id | k\* | refined_n | logit_sum | llr_peak |
|---|---|---|---|---|---|
| 1 | 66  | 1280 | 1280 | 2522.4 | 0.543 |
| 2 | 195 | 1280 | 1280 | 1391.4 | 0.537 |
| 3 | 149 | 1280 | 1280 | 1377.5 | 0.556 |
| 4 | 129 | 1280 | 1280 | 1221.3 | 0.596 |
| 5 | 190 | 640  | 640  | 886.3  | 0.605 |

T10181 (NLP):

| rank | kball_gap_id | k\* | refined_n | logit_sum | llr_peak |
|---|---|---|---|---|---|
| 1 | 195 | 1280 | 1280 | 2680.4 | 0.573 |
| 2 | 16  | 1280 | 1280 | 2321.9 | 0.541 |
| 3 | 126 | 1280 | 1280 | 2020.1 | 0.658 |
| 4 | 169 | 1280 | 1280 | 1875.4 | 0.429 |
| 5 | 7   | 1280 | 1280 | 1870.4 | 0.577 |

The top survivors on both topics all saturate at k\*=1280 — they're the largest, most-cohesive book-distant blobs in each corpus, ranked by `logit_sum` (which scales roughly with cluster size at fixed mean posterior). For deployment we'll want to either extend K_GRID past 1280 or report a separate "small-cluster" leaderboard ranked by `mean_logit` instead.

## Implications for the production pipeline

- **K-ball + LLR refinement is the production path.** End-to-end runtime is ~1 minute for a 20k-work topic and ~9 minutes for a 110k-work topic, dominated by FAISS KNN search in `setup()`. This scales to the physical-sciences domain because the LLR refinement is O(200) per topic — independent of corpus size beyond the up-front KNN.
- **The refined output is meaningfully different from the K-ball seed.** Median Jaccard of 0.26-0.38 (matching the static-seed Phase 1-2 results) means roughly 60-75% of each K-ball seed's members are replaced by LLR re-centering. Anyone consuming K-ball gap CSVs directly is using a different cluster than they think they are.
- **K_GRID saturation matters for the leaderboard.** Top-ranked survivors all hit 1280 on both topics. Reranking by `mean_logit` (or any size-normalized score) and/or extending K_GRID to 2560 or 5120 would let LLR distinguish "big and tight" from "big and bursting".
- **Dedup at J ≥ 0.5 collapses ~10-30% of K-ball seeds.** This is the right scale: it cleans up the matrioshka-eating overgrowers without flattening the natural ~200 K-ball gap structure (cf. Phase 2's static-seed dedup which had to handle ~10⁵ raw candidates).
- **Templated-leak tagging is still external.** The Phase 2 softmax-enrich pipeline's keyword tagger isn't plumbed into the refined CSV. For now, the LLR machinery happily refines and ranks tight artifact clusters alongside real subspecialties; downstream LLM proposal generation will need to filter, or we plumb the tagger as a follow-on.

## Open questions

- **K_GRID extension.** All five top survivors on T10181 saturate at k\*=1280 (33/200 on T11303 too). Without extending the grid, we can't distinguish "the right k\* for this seed is 1280" from "the right k\* is 5000 but we capped it." A follow-on diagnostic on the most-saturated handful (T10181 gap 195, T11303 gap 66) with K_GRID = (..., 1280, 2560, 5120) would settle this.
- **Templated-leak filter.** Should the refined CSV inherit a `templated_leak` column from the softmax-enrich tagger, or stay a pure geometric output and let the downstream proposal generator filter? The former simplifies the consumer side but couples this script to keyword heuristics that don't yet generalize past physical sciences.
- **Sensitivity to K-ball `radius_fraction`.** This run used `radius_fraction=0.75` K-ball outputs. With `rf=0.5`, K-ball gaps would be smaller and tighter; the LLR refinement should still re-center to roughly the same k\*-bounded set, but the **dedup merge rate** is likely to be higher (tighter seeds = more candidates lying inside the LLR-grown radius). Worth a quick check before locking radius_fraction in production config.
- **`mean_logit` vs `logit_sum` as the ranking metric.** `logit_sum` rewards size; `mean_logit` rewards purity. For a "Books to write" recommendation list, `mean_logit` is probably more useful for the long tail; `logit_sum` is fine for the top-of-list.
