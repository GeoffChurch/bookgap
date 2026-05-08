# Fuzzy / soft-EM membership in the symmetric loop, preserving the local boundary

Replace G's hard set membership with continuous weights w(x) ∈ [0, 1]
in the symmetric add+remove inner loop of `gap_article_as_seed.py`,
keeping everything else (local CSLS-balanced boundary, self-tuning
Gaussian kernel, per-basin local computation) intact. Equivalent to
upgrading the current Viterbi-style hard EM to soft EM, only at the
within-basin level.

## The reframing that makes this obvious

The current symmetric loop is **Viterbi-style hard EM** on a 2-component
mixture (G vs B):

1. **E-step (hard):** for each candidate, compute posterior; threshold
   to get binary membership.
2. **M-step (implicit):** the boundary B reconstructs from G's hard
   support, and the kernel sums use only points where w(x) ∈ {0, 1}.
3. **Iterate to fixed point.**

Soft EM keeps the posteriors instead of thresholding:

1. **E-step (soft):** w(x) ← post(x | current weights).
2. **M-step:** B reconstructs from G's *thresholded* support (or
   weighted boundary; see below). Kernel sums use w(x) directly.
3. **Iterate to weight convergence.**

Both converge to fixed points. Viterbi is faster per iter, locks in
early decisions; soft is smoother, more invariant across initial
conditions, captures multi-scale membership naturally.

## Why this fixes failure modes we've seen

Validating Phase 3 on T10181 (Task 10) showed that ~5 K-ball topics
that have textbooks (Dependency Parsing, Coreference, Morphology, MT
Evaluation, Question Answering) ended up tagged `book_anchored=0`
because the relevant book never got pulled into the basin during
symmetric expansion. Two specific Viterbi artifacts:

- **Threshold brittleness.** A book whose individual `min_post(G ∪ {book})`
  test sits just below τ never enters G. Under soft EM it'd have
  w=0.4, contribute partially to K_G for nearby members, and either
  rise on the next iteration (if its posterior under the updated G
  goes up) or fall.
- **Path-dependence between seeds.** Two seeds in the same basin
  produce different fixed-point member sets because early-iteration
  hard additions diverge. Soft updates accumulate more gradually and
  reach more invariant weight distributions.

Both also explain the 48.8% confluence we got with ordered cumulative
ADD on T11303. Soft EM should push that meaningfully higher — likely
toward weight-distribution-cosine ≥ 0.95 across same-basin seeds, even
if exact-set equality is rarer.

## What's preserved

- **Local CSLS-balanced boundary** (`cohesion_design_axes.md` §H — the
  clear winner against global-corpus priors that fail in 384-dim).
  Boundary is still defined per-basin; books in G's thresholded support
  are skipped from B's lookup; CSLS-balanced f(b)-weighting carries
  over (with f(b) becoming a weighted vote: `f(b) = Σ_{i: 1-NN(i)=b} w(i)`).
- **Self-tuning Gaussian kernel** with K_sigma=7 (Zelnik-Manor convention).
- **Per-basin local computation.** No N²-scale global mixture.
- **HNSW neighborhood infrastructure.** Same K-NN queries.
- **The two-pass discovery shape** (book seeds first, then article seeds)
  and the claimed-overlap fast path. Fuzzy membership is an inner-loop
  change, not a discovery-pass change.

## What changes

```python
# Before (Phase 3 inner loop, sketched):
G: set[int] = {seed}
for it in range(max_iters):
    cands = find_frontier(G)
    G_candidate = G.copy()
    for c in cands:                      # ordered cumulative ADD
        if min_post(G_candidate | {c}) >= tau:
            G_candidate.add(c)
        else:
            break
    G_new = G_candidate - {i for i in G_candidate if post(i | G_candidate) < tau}  # REMOVE
    if G_new == G: break
    G = G_new

# After (soft EM, sketched):
w = np.zeros(n)
w[seed] = 1.0
for it in range(max_iters):
    G_supp = np.where(w > tau_supp)[0]    # thresholded support for boundary lookup
    if len(G_supp) < 2:
        # Bootstrap: pick nearest unweighted neighbor and set w=1
        ...
        continue
    B_books, B_articles, B_mass = construct_boundary(G_supp, w)  # f(b) = Σ_i w(i)·1{1NN(i)=b}
    # Update weights for every point in the neighborhood of weighted centroid
    centroid = weighted_mean(emb, w)
    candidates = knn_around(centroid, top_K)
    new_w = w.copy()
    for y in candidates:
        K_G = sum(w[x] * kernel(y, x) for x in supp(w))
        K_B = sum(B_mass[b] * kernel(y, b) for b in B)
        new_w[y] = K_G / (K_G + K_B)      # soft posterior — directly the new weight
    if np.linalg.norm(new_w - w) < eps: break
    w = new_w
```

The convergence criterion shifts from set-equality to weight-vector
distance. The output shifts from a `frozenset` to a sparse weight
vector (only points with w > some display threshold are surfaced).

## What it doesn't fix

- **Cross-basin competition.** Two basins still don't talk to each
  other. A book that's a member of basin H doesn't pull articles
  toward H when basin G is expanding. That requires a global-mixture
  or lockstep-style approach. See `lockstep-expansion` backlog item.
- **Geometrically-far books.** A book that's not in any high-weight
  article's K-NN never enters the math. The local-boundary trick by
  construction ignores far-field books. Fuzzy-vs-hard doesn't change
  this.

## Coverage detection under fuzzy membership

Today: `book_anchored = (n_books_in_G > 0)`.

Under fuzzy: `expected_books_in_basin = Σ_b w(b)` over books b in G's
neighborhood. A natural threshold `expected_books_in_basin ≥ 0.5`
gives "at least one book is more-likely-in-than-out."

Then "is article x covered?" becomes a soft score:
`coverage(x) = max over discovered basins of (w_k(x) · expected_books_in_basin_k)`.

This handles the multi-scale case naturally — an article in a tight
specialty and a broad mega-cluster gets coverage from whichever has
books, weighted by membership strength.

## Test plan

1. **Implement** as a `--fuzzy-membership` flag on `gap_article_as_seed.py`.
   Inner loop replaces symmetric add/remove; everything else stays.
2. **T11303 confluence diagnostic:** weight-vector cosine between
   same-basin seeds. Target: cosine ≥ 0.95 even for hard-set-non-identical
   pairs.
3. **T10181 K-ball false-gap validation:** rerun the K-ball-top-20 →
   Phase-3-basin mapping (the inspection from Task 10's results). Did
   the ~5 missed topics (Dep Parsing, Coreference, Morphology, MT
   Eval, QA) now have `expected_books ≥ 0.5` in their corresponding
   fuzzy basins?
4. **Wall time:** soft EM is more expensive per iter (every neighborhood
   point gets weight updates, vs. only top-W passing candidates).
   Measure on T11303; if the slowdown is < 3× the current ordered-cumulative
   variant, ship.

## Pairs with

- **Cross-basin / multi-membership coverage scoring** (the `coverage(x)`
  formula above). Not a separate backlog item yet; would emerge naturally
  as the comparison-table consumer of the fuzzy weights.
- **`lockstep-expansion`** (existing backlog) — the across-basin sibling
  of this within-basin change. They're complementary; lockstep is a much
  bigger restructure that this doesn't depend on.
- **`boundary-centroid-cosine`** (existing backlog) — once we have soft
  weights on books, "boundary book centroid" becomes weighted, which
  gives a cleaner version of that heuristic.

## Cost vs. value

~1-2 days of focused implementation work. The change is localized
(one inner loop in `gap_article_as_seed.py`) but the data shape is
different enough (sparse weights instead of frozensets) that
downstream consumers (`gap_book_existence_filter.py`, the comparison
table, the CSV schema) need adaptation.

The qualitative gain is large for a small algorithmic change: the
specific failure mode that Task 10's validation surfaced
(book-anchoring missed for ~5 K-ball topics due to threshold
brittleness) is exactly what soft EM with the same boundary would fix.
