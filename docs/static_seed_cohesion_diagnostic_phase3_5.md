# K-ball + LLR refinement — Phase 3.5 follow-ups

Follow-up to Phase 3 (`docs/static_seed_cohesion_diagnostic_phase3.md`)
that lands six additions:

1. Scaling K-ball seed pool from 200 to 1000 candidates per topic.
2. Empirical validation of corpus-null sub-sampling, with σ_k measured
   on T11303 and T10181.
3. New per-member logit-distribution diagnostic.
4. Stein two-stage adaptive sub-sampling for the corpus null.
5. Two-stage dedup: containment-pruning + Jaccard merging.
6. Fuzzy-jointnorm K_G and K_B as the new default posterior.

Run date: 2026-05-12. All commits land on `main`.

## Headline numbers

| | T11303 (1000 gaps) | T10181 (1000 gaps) |
|---|---|---|
| Refined survivors after Jaccard ≥ 0.5 dedup | 420 / 1000 (42%) | 829 / 1000 (83%) |
| Median survivor `kball_size_delta` (refined − K-ball) | +33.5 (Phase 3 was +33.5 at n=200) | +29.0 |
| Saturators at `k* = 1280` (survivors) | 2 / 420 (0.5%) | **108 / 829 (13%)** |
| `σ_k` at k=1280 (full-corpus null) | 0.176 | 0.185 |
| Top-1 by median_logit | kb-327 (k\*=5, ml=3.46) | **kb-144** (Filipino-names leak; k\*=20, ml=5.26) |

The survivor count on T11303 stayed at 142 → 420 as K-ball was scaled
200 → 1000 — non-trivial growth in candidate count, validating that
T11303 has more like 400 natural cluster centers, not 200.
T10181's 83% survival rate (vs T11303's 42%) reflects the larger
corpus having more genuinely independent semantic basins.

## 1000-gap K-ball + LLR refinement

K-ball `num-gaps` scaled from 200 to 1000 on both topics. K-ball
runtime grew linearly (~20s → ~60s on T11303, ~5min → ~8min on
T10181). LLR refinement grew linearly in num_gaps (~1min for 1000 on
T11303, ~20min on T10181). Outputs at
`artifacts/gap_kball_llr_refine/{T11303,T10181}_n1000__llr_refined.csv`.

T11303 survivor k\* distribution shows the matrioshka structure that
was driving Phase 3's dedup: heavy mass at small k (84 at k=5, 96 at
k=10, 71 at k=20), tapering, with only 2 of 420 saturating at k=1280.
T10181's distribution is broader (212 at k=40, 172 at k=20, 131 at
k=80) and has the 13% saturator tail — those are the genuine large
basins that K_GRID truncates.

## Sub-sampling validation (T11303 + T10181)

A new diagnostic
(`artifacts_scripts/null_subsampling_diagnostic.py`) measures the
empirical SE of the sub-sampled corpus null. Run on both topics
(`cache/null_subsampling_diagnostic/{T11303,T10181}.csv`).

`σ_k` empirically across topics:

| k | T11303 σ_k | T10181 σ_k | ratio |
|---|---|---|---|
| 80 | 0.095 | (~0.10–0.12) | ~1.1× |
| 160 | 0.112 | 0.119 | 1.06× |
| 320 | 0.132 | 0.141 | 1.07× |
| 640 | 0.153 | 0.163 | 1.07× |
| 1280 | 0.176 | 0.185 | 1.05× |

σ_k is essentially scope-level-invariant. T10181 has 6× more articles
than T11303 but σ_k is within 5–10%. Two implications:

- Sub-sampling reliability is **not** N-driven; m=2000 generalizes
  comfortably across topics at the same scope level.
- The CLT prediction `SE = σ_k / √m` matches empirical SE within ~20%
  at every m and every k. Bias is essentially zero — random
  sub-sampling is unbiased as expected.

At m=1000 on T11303, relative SE is 0.1–0.3% across the entire grid.
At m=2000 it's 0.07–0.2%. Either is orders of magnitude smaller than
the LLR signal we care about (LLR peaks are O(0.5) on typical
clusters).

## median_logit default + clipping validation

Phase 3 used `mean_logit` ranked the clipped per-member logits
(`np.clip(post, 1e-3, 1-1e-3)` before `log(p/(1-p))`). The clip was
load-bearing in theory (preventing `±inf` from blowing the sum) but
its empirical effect is invisible. The new
`artifacts_scripts/gap_logit_distribution.py` tool re-runs the
local-CSLS posterior on a refined gap and dumps quantiles + clipped /
un-clipped mean & std.

T11303 top-5 by median_logit (per
`cache/gap_logit_distribution/T11303_top5.csv`):

| gap | k\* | mean_clipped | mean_raw | std_clipped | std_raw | min | q10 | med | q90 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| kb-327 | 5 | 3.50 | 3.50 | 0.45 | 0.45 | 3.03 | 3.04 | 3.46 | 4.03 | 4.25 |
| kb-311 | 5 | 3.21 | 3.21 | 0.91 | 0.91 | 2.03 | 2.43 | 3.04 | 4.15 | 4.85 |
| kb-588 | 1280 | 1.98 | 1.98 | 0.61 | 0.61 | 0.26 | 1.13 | 2.01 | 2.75 | 3.51 |
| kb-341 | 5 | 1.84 | 1.84 | 0.24 | 0.24 | 1.61 | 1.62 | 1.78 | 2.12 | 2.26 |
| kb-18 | 40 | 1.64 | 1.64 | 0.82 | 0.82 | −0.28 | 0.56 | 1.68 | 2.51 | 3.20 |

`mean_clipped == mean_raw` to two decimals on every gap. No member
has `post < 1e-3` or `post > 1 - 1e-3`; the clip never fires.
median_logit and mean_logit are interchangeable in practice on this
data.

Default rank-by switched from `mean_logit` to `median_logit` (commit
`66a62f7`) — same answer in practice, more principled in theory, no
artificial threshold.

## Stein two-stage subsampling

Implemented (`_compute_corpus_null_stein`, commit `cfa2e3b`):

1. Pilot pass: m_pilot=200 random article seeds, compute σ_k.
2. m_needed = max_k (σ_k / (target_rel_se · |null_k|))².
3. Final pass: max(m_needed, m_min=500) seeds, reuses pilot's kNN
   rows and draws additional seeds as needed.

CLI:

```
--null-method {fixed,stein}      default stein
--null-target-rel-se 0.005       stein-only precision target
--null-pilot-size 200            stein-only pilot size
--null-sample-size 2000          fixed-only sample size
```

**Stein is now the default**. On topic-scale data (T11303 / T10181),
m_needed lands around 500–1500, well below the fixed-m=2000 default
while giving tighter precision than the user specifies. At larger
scopes (subfield / field / domain) the pilot would adapt upward
automatically — no retuning required.

The current Stein implementation reuses `state["D_art_NN_all"]` from
setup(); it adapts m without actually saving kNN compute. The kNN
savings (the case where Stein matters for wall time) requires a
setup() restructure to build kNN only for the sampled seeds — a Phase
4 follow-on.

## K_GRID extension — design notes

Saturator data: T10181 has 108 / 829 = 13% of survivors saturated at
k\*=1280. T11303 has 2 / 420 = 0.5%. T10181 has real broad
subspecialties whose natural sizes are above the current K_GRID's max.

The straightforward extension: K_GRID =
`(5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120)`. Every seed
evaluates LLR at every k; no per-cluster adaptive grids. This works
because:

- A cluster whose true peak is at k=200 has LLR(k=200) > LLR(k=2560) >
  LLR(k=5120) (LLR must decline as k grows past the peak; in the
  limit k→N, LLR→0). So argmax still picks the natural peak.
- Only the saturators benefit from the extra k points; everyone else
  is unaffected.

Cost:

- `K_QUERY` decouples into two: `K_QUERY_SETUP=1500` (what
  `setup()` builds for all N works; unchanged) and `K_QUERY_GRID=5120`
  (top-5120 nearest articles, only for the m=2000 null subsample and
  the ~1000 K-ball gap centroids).
- Per-K-ball-centroid kNN at K_QUERY_GRID=5120: ~25 ms × 1000 gaps =
  ~25 s on T10181. Trivial.
- Null subsample kNN at K_QUERY_GRID=5120 over m=2000 seeds: ~50 s on
  T10181. Trivial.
- Total added cost on T10181: ~1–2 minutes. Worth it given 13% of
  survivors are currently truncated.

Not extending further than 5120 in the first cut. If the post-
extension saturator fraction is still meaningfully nonzero, double
again to 10240. Each doubling is cheap given the K_QUERY decoupling.

## Two-stage dedup: containment-pruning + Jaccard merging

Phase 3 used single-stage Jaccard ≥ 0.5 union-find dedup. Phase 3.5
splits this into:

- **Stage 1 — containment pruning.** For each pair (A, B) with
  |A| > |B|, containment(B in A) ≥ 0.9, and B's median_logit > A's:
  drop A. Catches the "diffuse saturator wrapping a tight inner core"
  pattern.
- **Stage 2 — Jaccard merging.** Union-find at J ≥ 0.5 on survivors.
  Catches symmetric same-size near-duplicates.

Per-pair enumeration uses an inverted index (article → gap indices)
to skip non-overlapping pairs — ~8× speedup vs naive O(N²) at 1000
gaps. CSV gains a `dedup_reason` column with values `"survivor"` /
`"containment_pruned"` / `"jaccard_merged"`.

Post-hoc analysis on T10181 1000-gap showed the two stages are
near-orthogonal: of 421 firings, only 2 are caught by both; 327
Jaccard-only (same-size near-duplicates) and 92 containment-only
(saturator-wraps-tight-inner). Both stages are needed.

## Fuzzy K_G + fuzzy K_B (FF) as default posterior

Phase 3 used hard-boolean per-member gap membership for K_G and
hard-CSLS-balanced K_B. The "Frankenstein" worry was: under K_G_fuzzy
(members-only joint-normalized) + K_B_hard, big clusters whose members
are shared with tighter peers get their K_G strongly diluted while
their K_B isn't, structurally biasing the posterior toward tight
clusters in a topic-dependent way.

Concretely, on T11303 the saturator kb-453 ("Bayesian network
learning" broadly) went from median_logit = +2.00 under HARD to
**−3.38 under members-only FJ + hard K_B** — a structural collapse,
not a semantic finding. T10181's saturators stayed near +2.0 under the
same setup. The asymmetry produces very different behavior across
topic scales.

The FF (Fully Fuzzy) formulation fixes this by:

- Computing μ_G(j) over **extended support**: top-K_QUERY ≈ 1500
  nearest articles to each gap's centroid (not just hard k\* members).
- Joint-normalizing across gaps so Σ_g μ_G_g(j) = 1 for every article
  claimed by any gap.
- K_G(i) = Σ_j W_ij · μ_G(j) · 1[j ≠ i] (kernel-weighted expected
  fuzzy gap membership)
- **K_B(i) = Σ_{j ∈ B_CSLS} W_ij · (1 − μ_G(j))**: a CSLS background
  article that's partially claimed by this gap contributes less to the
  bounding-object mass.

Cross-topic comparison of the tight–broad median_logit gap:

| topic | HARD baseline | members-only FJ | **FF (default)** |
|---|---|---|---|
| T11303 (kb-327 tight vs kb-453 broad) | 1.46 | 6.84 | **5.14** |
| T10181 (kb-144 tight vs kb-373 broad) | 3.31 | ~5 | **3.66** |

FF retains strong tight-promotion (5.14 logit gap) without the
topic-dependent extreme of members-only FJ (6.84 on T11303 vs ~5 on
T10181).

Final pipeline results at 1000 K-ball gaps:

| | T11303 | T10181 |
|---|---|---|
| Survivors | 363 | 706 |
| Containment-pruned | 602 | 258 |
| Jaccard-merged | 35 | 36 |
| Saturators (k\*=1280) | 0 | 58 |
| Top-1 by median_logit | kb-327 (k=5, ml=3.42) | kb-144 (k=20, ml=5.89) |

T10181's 58 surviving saturators all have **negative** median_logit
(top is −1.48). These are the genuinely isolated diffuse subspecialties
— broad clusters with no tight inner core to be pruned by. Negative
median is the correct FF semantics for such clusters: under joint-
normalized fuzzy membership, a 1280-article cluster with no internal
structure has many members with weak fuzzy claim. Downstream consumers
can post-filter on `median_logit > 0` to drop them.

The 58 T10181 saturators correspond to broad subspecialties without
matrioshka structure (no K-ball seed found a tight inner core for
them). T11303 has zero such cases because every K-ball saturator there
has at least one tight inner cluster.

## Open follow-ons

- **K_GRID extension** — original plan was to add 2560 and 5120 to
  handle saturator truncation. Under FF this matters less: T11303 has
  zero saturator survivors, T10181's 58 surviving saturators all rank
  below 0 in median_logit anyway. Extension would only help if we
  cared about resolving the LLR peaks of the very-broad-no-inner
  subspecialties — probably not urgent.
- **setup() restructure** — Phase 4 work: instead of building
  `D_art_NN_all` for every work, do kNN on demand for the K-ball gap
  centroids and the null subsample. Unlocks deployment at field /
  domain scale where the current O(N²) kNN is the bottleneck.
- **Templated-leak filter** — `kb-144` (T10181 Filipino names) keeps
  surfacing at #1 by median_logit, exactly as desired for algorithm
  validation. Downstream consumers that don't want artifacts need a
  filter — the keyword tagger from softmax-enrich Phase 3 is the
  right candidate.
- **σ_k across scope levels** — empirical evidence only covers
  topic-level scope. Once we deploy at subfield/field/domain, we
  should re-measure σ_k to confirm Stein two-stage adapts correctly
  (or whether σ_k truly does grow with heterogeneity as conjectured).
