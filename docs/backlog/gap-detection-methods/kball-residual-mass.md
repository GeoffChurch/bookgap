# K-ball residual-mass gap sizing

## Idea

The current K-ball gap size is `|members|` — every article inside the
maximal book-free ball around the seed counts equally. A more
informative measure discounts articles based on **how much they're
already claimed by existing books**, giving the gap's *residual* mass
rather than its raw mass.

Four variants below, all in the family of "discount book-claimed
articles inside the k-ball." They differ in (a) how a book's claim is
defined and (b) whether discounting is hard or soft.

All four address the gap-*sizing* (and therefore ranking/selection)
question. They do **not** fix the data-debris artifact — debris is
geometrically distant from any AI book, so book-claim signals don't
shrink debris gaps. The debris fix is binomial-p filtering (uses
topic-wide base rate, separate axis from geometry).

## Variants

| # | Name | Book's claim | Discount | Cost | Status |
|---|---|---|---|---|---|
| 4 | **Virtual-centroid book** | The gap's centroid is added as a virtual book; real boundary books compete. | Hard — drop articles whose nearest book (among real + virtual) is a real one. | Cheapest | TODO |
| 3 | **Per-partition mutual-NN** | Partition gap by nearest boundary book → for partition `p_b` of size `|p_b|`, query book b's `|p_b|` nearest articles. Book NN query is NOT restricted to non-gap articles — in-gap articles can be claimed back. | Hard — drop articles in the mutual intersection. | Moderate | TODO |
| 1 | **Global book k-NN discount** | Each book's natural neighborhood = its top-k nearest articles (fixed global k). | Hard — gap articles inside their nearest book's k-NN are dropped. | Moderate | TODO |
| 2 | **KDE-on-KDE soft** | Each book is a Gaussian; gap is a Gaussian. | Soft — articles where `gap_KDE > book_mixture_KDE` are counted, weighted by ratio. | Heaviest | TODO |

## Variant details

### Variant 4: Virtual-centroid book (CHEAPEST)

```
centroid = mean(emb[m] for m in gap.members)
gap_size_residual = sum(
    1 for m in gap.members
    if dist(emb[m], centroid) <= dist(emb[m], nearest_real_book[m])
)
```

A meet of K-ball (envelope) and Voronoi (cell of the virtual book).
Reuses production-computed `nearest_real_book` and a single mean over
gap members. ~30 lines.

### Variant 3: Per-partition mutual-NN

```
for each boundary book b:
    p_b = {m ∈ gap.members : nearest_book[m] == b}
    if not p_b: continue
    nn_b = book_index.knn(b, k=|p_b|)   # book b's |p_b| nearest articles, NO restriction on gap membership
    claimed_by_b = p_b ∩ nn_b
    discount += |claimed_by_b|
gap_size_residual = |gap.members| - discount
```

Variable-k mutual-NN. Symmetric: each book gets to "claim back" the
same number of articles that named it as nearest. Similar to CSLS
hubness correction (cf. `artifacts_scripts/gap_effective_size_local.py`)
but with variable k tied to partition size.

### Variant 1: Global book k-NN discount

```
for each book b:
    nn_b = book_index.knn(b, k=K_GLOBAL)
    for m in gap.members:
        if nearest_book[m] == b and m in nn_b:
            discount += 1
gap_size_residual = |gap.members| - discount
```

Same idea as variant 3 but with a global K_GLOBAL instead of per-book
partition size. Simpler but loses the per-book mass-conservation
property of variant 3.

### Variant 2: KDE-on-KDE soft

```
book_mixture_density(x) = sum_b Gaussian(x; emb[b], sigma_b) / n_books
gap_density(x) = Gaussian(x; centroid, sigma_gap)
gap_size_residual = sum(
    sigmoid(log(gap_density(emb[m])) - log(book_mixture_density(emb[m])))
    for m in gap.members
)
```

Soft probabilistic version. Each member contributes between 0 and 1
based on the log-density ratio. Spiritual sibling of
`compute_gap_stats` in `gap_kball_llr_refine.py` (the bake-off-canonical
`effective_size`). Heaviest because of the book mixture evaluation
across all gap members.

## Connections to existing methods

- **Variant 4 ↔ Voronoi ∩ K-ball** (existing `VoronoiAlgorithm` + the
  current K-ball envelope, just intersected).
- **Variant 3 ↔ CSLS hubness correction** (cf.
  `gap_effective_size_local.py`'s boundary-book equally-sized outside
  neighborhood).
- **Variant 2 ↔ `compute_gap_stats`'s effective_size** (the
  Local-CSLS-balanced posterior the bake-off uses as `n_eff`).
- **All four ↔ `FL_BOOKDEMAND_RESID`** (which puts the residual-mass
  intuition at the gap-*selection* layer; these put it at the
  gap-*sizing* layer).

## Experiments

All runs: LPP (NORM_LPP) + GIST embedder + drop_k=10 + greedy_hard +
radius_fraction=0.75 + num_gaps=1 + candidate_pool_size=10 + gemma4:31b
proposals. AI subfield (1702), book+article work types.

| Variant | Status | Coverage mean / median | Proposal pct mean / median | gap median |
|---|---|---|---|---|
| (baseline) raw_kball LPP rf=0.75 | done | **82.3 / 95.6** | 70.0 / 93.6 | 366 |
| (baseline) raw_kball CCLIP rf=0.75 | done | **84.6** / 95.9 | 73.7 / 94.9 | 353 |
| **3 mutual_nn LPP** | **done** | **84.7 / 94.7** | **75.3 / 94.1** | 358 |
| 4 virtual_centroid LPP | done | 81.8 / 94.2 | 73.7 / 95.2 | 369 |
| 1 book_knn_discount LPP (k=50) | done | 82.7 / 94.9 | 70.7 / 92.7 | 370 |
| 2 kde_ratio LPP | done | 83.2 / 93.7 | 70.3 / 94.4 | 372 |

**Headline finding:** **LPP + variant 3 (mutual_nn)** matches CCLIP on
coverage mean (84.7 vs 84.6) and exceeds it on proposal gap percentile
mean (75.3 vs 73.7). For the paper-simplification claim, this is the
strongest result: not only does plain LPP suffice — LPP + a tweaked
gap-sizing metric *ties or beats* the published CCLIP headline.

All variants change which gap is selected on ~50/76 topics (vs the raw-KBALL
baseline) — substantial reordering despite small aggregate moves.
Most variants *raise the mean* by lifting low-coverage topics while
slightly *lowering the median* (which is already near 95%). The
exception is variant 4, which essentially ties baseline because its
top-VC equals top-raw-size on most well-clustered gaps (top-largest
candidate dominates by raw size; variant 4 has VC ≤ raw_size, so the
largest is unbeatable for greedy_hard candidates).

**Adjacent radius_fraction sweep** (LPP raw_kball, all else equal):

| rf | Coverage mean / median | Proposal pct mean / median | gap median |
|---|---|---|---|
| 0.5 | 80.0 / **100.0** | 57.1 / 93.1 | 80 |
| 9/16 = 0.5625 | 80.4 / **100.0** | 62.7 / 91.9 | 128 |
| 0.75 (default) | 82.3 / 95.6 | 70.0 / 93.6 | 366 |
| 1.0 | 47.5 / 48.0 | 52.7 / 54.2 | 1112 |

rf < 0.75 hits **median coverage 100%** because the tighter gaps are
small enough for gemma4 to fully centroid. But the *mean* drops: the
distribution is bimodal — some topics get perfect coverage, others get
0% because the gap shrank below "publishable book" threshold and
gemma4's proposal lands somewhere off-target (e.g., T10181 drifts from
"Chinese Word Segmentation" to "Clinical Phenotyping" at rf=1/2 because
the 93-article gap is dominated by clinical-NLP papers).

rf=1.0 is dramatically worse on every axis: the over-broad gaps produce
broad-umbrella proposals that don't centroid effectively. **rf=0.75 is
the right paper default.**

## Implementation plan

1. Add `gap.size_metric: str = "raw_kball"` to `GapConfig`, with
   enum or string values: `raw_kball | virtual_centroid | mutual_nn |
   book_knn_discount | kde_ratio`.
2. In `_compute_gaps_for_topic`, generate `num_gaps >= 10` candidates
   (so we have enough to re-rank), score each by the selected metric,
   sort, take top-1 (or top-num_gaps).
3. Each metric is a function `(gap_members, gap_seed, book_emb,
   article_emb, nearest_book_per_article) -> float`. Pure function,
   no state.
4. Variants 4 → 3 → 1 → 2 in order of implementation cost.

## Caveats

- These improve **gap-ranking signal**, not debris filtering. Combining
  with binomial-p (separate backlog entry) gives the cleanest debris
  story.
- Re-ranking changes the proposal context as well — gaps with smaller
  raw size but higher residual mass will win, so per-topic article
  composition changes. Topic-by-topic comparisons should be done
  per-variant, not against the raw-KBALL baseline directly.
