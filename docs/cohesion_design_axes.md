# Cohesion-aware gap detection: design axes

Working notes summarizing all the design dimensions we explored while
building cohesion-aware gap selection on top of K-ball / max-bookless.
Each axis lists its options, marks which are **competitive**, and
recommends a **default** when one is a clear winner or vastly simpler
than its close competitors.

Topic studied: T10181 (NLP), with cross-checks on T11303 (Bayesian
Networks) and T13734 (Applied/Engineering AI).

---

## A. Gap-source algorithm
*Where do candidate gap centers come from?*

- **K-ball** (`kball_gap_probe.py`) — geometric sphere around each seed
  bounded by `radius_fraction × dist-to-1-NN-book`. ✅ Competitive.
- Max-bookless (`maximal_bookless_extractor.py`) — dendrogram walk-up
  to first book-containing ancestor. Useful as a structural reference
  but produces sprawling gaps in book-poor regimes (T13734's largest
  mb gap was 5,325 articles, post 0.49). Less suitable as a refinement
  seed source.
- HDBSCAN clusters / cohesion-cut on SLT — discussed but not
  implemented.

**Default: K-ball** as the seed source for cohesion-aware refinement.
K-ball's greedy max-coverage selection produces tight, well-centered
seeds; max-bookless's dendrogram walk produces gap candidates of widely
varying quality.

---

## B. Initialization for cohesion refinement

- **K-ball gap members** (used in hill-climb, moving-centroid) — start
  G with the K-ball gap's actual articles. ✅ Competitive.
- Singleton centroid-derived (used in `gap_kball_cohesion.py`) — start
  G empty, sweep candidates by distance from K-ball mean. Works fine
  if scores are computed counterfactually (`G_test = G ∪ {candidate}`,
  always ≥ 2 members at evaluation time).

**Default: K-ball members.** Immediately gives a non-trivial G and
matches the actual K-ball signal we trust. Singleton init only differs
in regimes where the K-ball-mean ranking diverges from K-ball
membership, which is rare for clusters larger than ~20.

---

## C. Frontier construction
*Which candidates are considered at each expansion step?*

- **Fixed-centroid radial** — ranked by distance from frozen K-ball
  mean. Simple, deterministic. ✅ Competitive for tight symmetric
  clusters.
- **Moving-centroid radial** — ranked by distance from currently-
  recomputed centroid of G. Captures drift toward cluster's true
  center as G grows. ✅ Competitive, especially for offset K-balls.
- **Aspherical (union K-NN of G)** — ranked by min distance to any G
  member. Allows non-spherical cluster shapes. ✅ Competitive for
  elongated subspecialties.

The three find genuinely different things on asymmetric or large
clusters. They mostly agree on tight symmetric ones.

**Default: moving-centroid.** Best balance of complexity and
adaptiveness — captures offset-seed correction (drift-finds-tight-
interior on tiny K-balls) without admitting the extra failure modes
of aspherical expansion. Aspherical only wins for clusters with
genuinely non-spherical shape; for general use, moving-centroid is
strictly more flexible than fixed without much added complexity.

---

## D. Acceptance policy

- **Strict** — single candidate per iter (the nearest), accept iff
  min_post stays ≥ τ; stop on first violation. ✅ Competitive.
- **Lenient** — top-K candidates per iter, accept the one most
  improving logit_sum (subject to min_post ≥ τ). ✅ Competitive.

Strict gives tighter cores (matches the natural cluster boundary
exactly). Lenient lets the gap "skip past" boundary-violating candidates
and find slightly more articles further out, at the cost of dropping
min_post to ~0.5.

**Default: strict** for tight-core *discovery*. The boundary the strict
policy finds is interpretable as "the cluster's true edge." Lenient
gaps end up at a calibration cliff (min_post ≈ 0.51 across the board)
that isn't really informative.

---

## E. Shrinkage

- **Forced shrinkage if min_post < τ** (hill-climb, moving-centroid).
  Necessary when initialization can produce a G with sub-threshold
  members. ✅ Required.
- No shrinkage (gap_kball_cohesion). Only valid if init guarantees
  min_post ≥ τ.

**Default: forced shrinkage on.** No warmup parameter needed if scores
are computed *counterfactually* — i.e., always evaluate
`G_test = G ∪ {candidate}` rather than scoring the current G in
isolation. Counterfactual framing means |G_test| ≥ 2 at every decision
point, so K_G is always > 0 and the posterior is well-defined. Both
hill-climb and moving-centroid already use this framing.

---

## F. Boundary criterion
*Per-gap stopping rule — defines where each gap ends.*

- **min_post ≥ τ (τ = 0.5)** ✅ Clear winner.
- effective_size monotonicity — fails (the score is monotonic in size
  so there's no peak).
- logit_sum monotonicity — also mostly monotone within reasonable
  K_max, so it doesn't define a useful boundary.
- K-ball's intrinsic radius rule — too conservative; cohesion analysis
  showed K-ball under-sizes most clusters.
- Max-bookless's first-book-ancestor — too generous in book-poor
  regimes; sprawls far past natural cluster.

**Default: `min_post ≥ 0.5`.** Theoretically grounded (the worst-case
member's confidence is the right "is this gap intact?" signal), and
the only stopping rule we tried that produces meaningful peaks rather
than running to K_max. τ = 0.5 is the natural threshold (above:
foreground; below: background); raising to 0.7 would give tighter
cores at the cost of dropping more articles.

---

## G. Ranking score
*Cross-gap, sorts surviving gaps for display.*

- **logit_sum** = `Σ log(p/(1-p))` ✅ Canonical Bayes evidence; the
  theoretically grounded choice. Rewards both size and tightness.
- **mean_logit** = `logit_sum / |G|` ✅ Size-invariant, surfaces tight
  cores at the top.
- **min_post** ✅ Strictest tightness criterion; surfaces the most
  confidently-tight gaps.
- effective_size = Σ post — close to logit_sum but less interpretable.
- mean_post / mean_purity, median_post, q10_post, geometric mean,
  harmonic mean — variants emphasizing robustness or balance.

**Default: `logit_sum` for primary display ranking** (canonical), with
**`mean_logit` as a secondary view** for tight cores. Use `min_post`
as a *filter* (e.g., `min_post ≥ 0.7`) rather than as a ranker, since
its values cluster near the boundary threshold and don't separate
gaps well.

---

## H. Background construction
*How "out-of-gap" is defined for the local-bg classifier.*

- Global corpus prior — fails in 384-dim due to Gaussian normalization
  (degenerate posterior ≈ 1 for any moderately-tight gap).
- **Distance-ratio centroid** — likelihood-free, uses
  `‖x-μ_C‖² / (‖x-μ_G‖² + ‖x-μ_C‖²)`. Cheap, stable, but coarse
  (doesn't capture local cohesion structure).
- **Local CSLS-balanced** ✅ Clear winner. For each gap article, take
  its 1-NN book. For each unique boundary book b with f(b) gap
  articles claiming it, take b's f(b) nearest non-gap articles. Total
  background size ≈ |G|, balanced per book.

**Default: local CSLS-balanced.** The 1-NN-per-article book set is
the canonical "bounding object" definition (matches both K-ball's
radius-defining book and max-bookless's dendrogram-stopping book).
The CSLS-style `f(b)` sizing automatically discounts hub-books — a
boundary book that's nearest to many gap articles also pulls many
non-gap competitors into the background, so its bounding strength is
correctly normalized.

---

## I. Kernel form
*Similarity function for kernel-density-style cohesion.*

- **Self-tuning Gaussian**: `K(i,j) = exp(-d²/(σ_i σ_j))` ✅ Clear
  winner. Heat-kernel motivation (canonical in any-d Euclidean space),
  no singularity, works in high-dim.
- Inverse-square `1/d²` — has singularity at d → 0 (near-duplicates
  blow up); dimension-inappropriate (Newtonian Green's function is
  3D-specific; in d-dim it's `1/r^(d-2)`).
- Top-hat / K-NN purity — discrete, parameter-free but loses density
  structure within the K-NN ball.
- Distance-ratio (no kernel) — likelihood-free alternative; coarser
  than Gaussian KDE.

**Default: self-tuning Gaussian.** The heat kernel is the right
canonical choice in high-dim, and self-tuning bandwidth removes the
need to calibrate σ globally.

---

## J. Bandwidth source for σ

- **Self-tuning local** (σ_i = √(median squared K_sigma-NN distance))
  ✅ Clear winner. Adapts to local density (sharp in dense regions,
  broad in sparse). Standard from Zelnik-Manor & Perona /
  diffusion-maps / t-SNE.
- Global single σ — needs calibration; doesn't adapt to varying
  density.
- σ fit from gap data (parametric Gaussian fit) — fails in
  high-dim (unstable variance estimates for |G| < d).

**Default: self-tuning local with K_sigma = 7** (Zelnik-Manor's
convention, robust empirical default).

---

## K. Distance / aggregation

- **L2 squared distance** (with Gaussian kernel transform) ✅ Clear
  winner. Rotation-invariant; on the unit sphere where embeddings live,
  monotone in cosine angle.
- L1 — not rotation-invariant, coordinate-system-dependent.
- For per-gap aggregation: mean (logit, post), min/max (boundary),
  median (robust), sum (evidence). Each captures different aspects;
  no single winner.

**Default: L2 squared distance for kernel; L2 for centroid/RMS-based
summaries.** Use median (q50) or q10 for robust aggregation when
outliers are a worry.

---

## L. Boundary vs ranking distinction
*The key conceptual insight: separate per-gap stopping from cross-gap
ordering.*

✅ **Critical to keep separate.** Conflating them causes failure modes:
- Using ranking score (logit_sum) as boundary → gap engulfs corpus
  because logit_sum is monotone-ish.
- Using boundary score (min_post) as ranking → values cluster near
  threshold and don't separate gaps.

**Default: min_post as boundary, logit_sum as primary ranker, mean_logit
as secondary tight-core view.** The separation is the load-bearing
design move.

---

## M. Operational filters

- **Size filter** (e.g., `30 ≤ |G| ≤ 200`) — operational constraint
  for downstream proposal generation. Discussion: ❌ filter masks
  metric design issues. Now removed; size constraints applied at
  the consumer level instead.
- **Jaccard dedup** with threshold τ — current default τ = 0.5 is
  fairly lax (only ~2 of 200 gaps eliminated; 17 pairs with J ∈
  [0.3, 0.5) survive). ✅ Mostly fine.
- **Containment-based dedup** — alternative for "tight core nested in
  halo" cases (drop smaller gap if `|small ∩ large| / |small| ≥ 0.5`).
  Worth trying if Jaccard misses these.
- **Dedup priority** — currently `logit_sum` (larger wins). Could be
  `mean_logit` (tighter wins). Empirically minor effect at τ = 0.5.

**Default: Jaccard dedup with τ = 0.5, sorted by logit_sum.** Lower
to τ = 0.3 if quasi-duplicates are suspected. Switch to containment-
based dedup if "tight cores absorbed by halos" becomes the dominant
failure mode.

---

## N. K-ball / max-bookless internal knobs

K-ball:
- `radius_fraction = 0.75` — working default. Cohesion analysis
  shows K-ball is *systematically conservative* (under-sizes most
  clusters); raising to 0.95 would expand baseline gaps but the
  cohesion-aware refinement makes this less important.
- `num_gaps = 200` — working default; gives enough seed candidates.

HDBSCAN (for max-bookless extraction):
- `min_cluster_size = 10`, `min_samples = 15` — working defaults from
  the existing pipeline; affects the SLT but max-bookless uses the
  raw single-linkage-tree, not the HDBSCAN cluster labels, so this is
  somewhat orthogonal.

---

## Summary: recommended default pipeline

For a single topic, the recommended cohesion-aware gap pipeline is:

1. **K-ball seed source** with `radius_fraction = 0.75, num_gaps = 200`
2. **K-ball-init**, **strict policy**, **moving centroid** for refinement
3. Boundary: **`min_post ≥ 0.5`** with self-tuning Gaussian KDE +
   CSLS-balanced local background, K_sigma = 7
4. Forced shrinkage on; counterfactual scoring (no warmup needed)
5. Display ranking by **logit_sum**; secondary view by **mean_logit**
6. Jaccard dedup at τ = 0.5

Out of 14 axes, ~6 have a clear-winner default and the rest are
largely consistent across reasonable choices, modulo the
boundary-vs-ranking separation (axis L) which is the load-bearing
conceptual move.

---

## Empirical findings

### 1. K-ball is systematically conservative, not generous

The original "Oort cloud around a tight core" hypothesis was inverted by
the cohesion analysis. K-ball's `radius_fraction × dist-to-1-NN-book`
rule under-sizes most clusters; cohesion-aware expansion almost
universally *grows* gaps past their K-ball size. For T10181, the
strict-moving-centroid expansion grew kb-2 from 291 → 446 articles,
kb-46 from 76 → 490, kb-102 from 19 → 50. The cohesion measure says
these regions are still tight at the larger sizes.

### 2. Confluence makes article-as-seed tractable

Moving-centroid expansion is highly deterministic — two seeds in the
same cluster compute centroids that converge within a few iterations
and then trace identical expansion paths. This means most seeds in
already-discovered basins can be cheaply early-aborted via a claimed-
set check. Empirically (with HNSW K-NN, see below):

| Topic | Articles | Books | Wall | Discovered | Discovery rate |
|-------|----------|-------|------|------------|----------------|
| T11303 (Bayesian Networks) | 14K | 145 (1.0%) | 42s | 254 | 1.80% |
| T10181 (NLP) | 88K | 1,252 (1.4%) | 640s (10.7m) | 1,556 | 1.76% |
| T13734 (Applied AI) | 142K | 353 (0.25%) | 3913s (65m) | 1,064 | 0.75% |

~82% of seeds reject via early-abort, ~16% via instant claimed-set skip.

T13734's lower discovery rate is the predicted consequence of book
sparsity: fewer books → larger natural basin sizes → each claims more
articles → fewer remaining for new basins. Cross-topic discovery rate
correlates closely with book density.

**HNSW K-NN gives a 5–9× speedup over FlatL2** at T10181/T13734 scale
(with efSearch ≥ K_query), with identical gap counts. The
`gap_article_as_seed` pipeline uses HNSW for the full-corpus index;
book and article indices stay Flat (smaller).

**Note on max_iters:** the inner loop is capped at 500 iterations,
which translates to a max gap size of 501. T13734's top gaps all hit
this cap because the underlying cohesive regions are genuinely larger;
T10181 and T11303 rarely hit it. Raising max_iters quadratically
increases compute cost (each iteration's compute_state cost scales with
|G|), so the cap is a sensible default — 500-article gaps are already
at the upper bound of useful proposal-generation size.

### 3. Tight-core detection finds real subspecialties + corpus QC issues

The cohesion + min_post pipeline correctly identifies the tightest
clusters in embedding space — and those include both:
- **Real research subspecialties with no textbook**: LLM inference
  efficiency (kb-102), text/image style transfer (kb-172), Chinese
  microblog word segmentation (kb-195), MAP inference (T11303 seed 60),
  Markov-network structure learning (T11303 seed 874), variational
  message passing (T11303 seed 326).
- **Corpus-QC artifacts (templated data leaks)**: Filipino-name records
  (kb-144 in T10181), N3C clinical concept sets (kb-30 in T10181),
  Tibetan folklore transcriptions (mb-194479 in T10181), French-language
  Bayesian-network papers (T11303 seed 6929).

The metric is doing its job; data-noise is a separate corpus-QC
concern. Templated/structured records are intrinsically tighter in
embedding space than organic prose, so they will always rank high under
any tightness measure. Filtering them needs a *semantic* signal
(boundary-book topical relatedness, OpenAlex topic_id purity, or
explicit data-quality QC), not a tighter cohesion threshold.

### 4. logit_sum and min_post surface largely different gaps

Top-N by logit_sum favors larger cohesive regions (size 80-500); top-N
by min_post favors small tight cores (size 10-30). On T11303, only
1 of 4 top gaps is shared between the two rankings (MAP inference,
seed 60). For proposal generation, both rankings are useful and surface
distinct candidates.

### 5. Topic-level book-richness shapes the gap landscape

Cross-topic comparison on three single-topic scopes:

| Topic | Books | Singletons | Top mb size | Top mb post | Character |
|-------|-------|------------|-------------|-------------|-----------|
| T10181 (NLP) | 1,252 (1.4%) | 97.8% | 98 | 0.95 | small clean gaps + 3 data-noise clusters |
| T11303 (Bayesian Networks) | 145 (1.0%) | 98.0% | 75 | 0.91 | small clean gaps, no visible noise |
| T13734 (Applied AI) | 353 (0.25%) | 92.6% | 5,325 | 0.49 | one sprawling mb top gap (correctly demoted by cohesion); real engineering subdomains below |

Book-poor topics produce huge sprawling top max-bookless gaps (the
dendrogram walk runs unchecked) that the local-bg post correctly
dampens. The cohesion metric works in all three regimes without
parameter tuning.

### 6. Centroid drift matters most for tiny K-balls; symmetric clusters are unaffected

Comparing fixed-centroid (singleton-init) vs strict-moving-centroid
(K-ball-init) on T10181:
- **kb-144 (Filipino, 40 articles, symmetric)**: both produce identical
  output (40, 0.97). Drift doesn't matter.
- **kb-102 (LLM inference, 19 articles, K-ball seed offset from cluster
  center)**: fixed produces (19, 0.89); moving produces (50, 0.60). The
  centroid drift correctly identifies the 50-article cluster.
- **kb-13 (Code LLMs, 142 articles, large)**: fixed produces (180, 0.54);
  moving stays at (142, 0.71). Different paths, different boundaries —
  neither is strictly "right."

Drift's biggest practical effect is on tiny K-balls where the seed is
non-representative of the cluster center.
