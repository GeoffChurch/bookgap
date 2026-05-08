# Full-data soft-mixture posterior (proper EM, no separate boundary)

Replace the current `K_G(y) / (K_G(y) + K_B(y))` posterior — where B is
a size-matched local CSLS-balanced sample of `¬G` — with the strict
mixture-model posterior:

```
P(z=G | y) = K_G(y) / K_total(y) = Σ_x w(x) · K(y,x) / Σ_x K(y,x)
```

over **all** points x (not a subsample), where `w(x) ∈ [0,1]` is the
membership weight in G. Each point contributes `w(x) · K(y,x)` to K_G
and `(1 − w(x)) · K(y,x)` to K_¬G; their sum is `K(y,x)`, so
`K_G + K_¬G = K_total`. No separate boundary set is constructed; the
"background" emerges naturally as the complement of membership across
the entire dataset.

## What's wrong with the current local-CSLS-balanced approximation

`docs/cohesion_design_axes.md §H` settled on local CSLS-balanced
boundary as the empirical winner. Two design choices baked in:

1. **B is a local sample** drawn from points spatially near G's
   boundary, not uniformly from `¬G`. Tractable and gives intuitive
   "boundary" semantics.
2. **B is size-matched to G** (`|B| ≈ |G|`). Empirically calibrated so
   `τ = 0.5` is the cluster boundary.

Both choices are tractability/calibration moves, not theory-guided.
Specifically, neither matches the strict 2-component mixture posterior
`P(z=G | y) = π_G · p(y|G) / p(y)`, which (with KDE estimates and
`π_G = |G|/N`) reduces to `K_G / K_total` over the full data.

The Task-7 "skip in-G books" rule was added to make the local-sample
approximation behave correctly under Phase 3 (when books may enter G).
Under proper soft-mixture semantics, this rule is unnecessary —
in-G points are handled correctly by the membership weighting.

## Mechanics

In a soft-EM context (`gap_soft_em.py` already provides the weight
infrastructure):

```
K_G(y)     = Σ_{x ∈ neighborhood(y)} w(x) · kernel(y, x)
K_total(y) = Σ_{x ∈ neighborhood(y)}        kernel(y, x)
post(y)    = K_G(y) / K_total(y)
```

Where `neighborhood(y)` is a K-NN set around y (not the full corpus —
sparsification is required for tractability). For weight updates:
`w_new(y) ← post(y)`, identical to the soft-EM iteration but with the
new posterior formula.

The `K_¬G(y) = K_total(y) − K_G(y)` decomposition makes it explicit
that "everything not in G" appears in the denominator, weighted by
how much each point doesn't belong to G. Points fully in G (w=1)
contribute only to K_G. Points fully out (w=0) contribute only to
K_total − K_G = K_¬G. Borderline points contribute to both.

## Why this might fix things the local-sample model can't

- **No "should we skip in-G books" question.** A book in G with w=0.9
  contributes 0.9·K(y,b) to K_G and 0.1·K(y,b) to K_¬G. No special-case
  filtering rule needed.
- **No size-matching calibration.** The threshold τ=0.5 is theoretically
  motivated ("more likely than not to be in G"), not empirically tuned.
- **Cleaner book-as-anchor semantics.** A book that's geometrically
  near a basin but not pulled in by expansion still contributes to
  K_¬G with full weight — and articles near it have lower posteriors
  in G as a result. The "Dependency Parsing book sits outside the
  basin's high-weight region" case becomes self-correcting: articles
  near the book have their G-posterior pulled down by K_¬G's mass
  there, until either the book gets pulled in or the basin shifts away
  from it.
- **No separate `compute_state` boundary construction.** Removes ~50
  lines of fairly subtle code (the 1-NN-book-outside-G + CSLS-balanced
  f(b) sampling). Replaces with one K-NN query per candidate.

## Relationship to other proposals

- **`article-as-seed-lockstep-expansion`** is a discrete special case:
  P(z=G_i | x) = K(x | G_i) / Σ_j K(x | G_j) competes among same-cardinality
  active basins. Full-data soft-mixture is more general — it competes
  G against the full data marginal, not against other basins. Lockstep
  is a discovery-dynamics restructure; this is a metric-formulation
  change. They're complementary: full-data soft-mixture defines the
  per-basin posterior; lockstep defines the discovery scheme.
- **Existing `gap_soft_em.py`** already has the weight infrastructure;
  the change is in the inner posterior computation. The size-matched
  local boundary in `compute_state` would be replaced with a K-NN
  neighborhood-around-y kernel sum.

## Cost and the K-NN sparsification trick

`K_total(y)` over all N points is `O(N)` per candidate evaluation,
prohibitive at T10181 scale (88K points). The standard fix: use a
K-NN graph and sum only over y's K-NN. For our self-tuning Gaussian
kernel, contributions from points beyond K-NN distance decay
exponentially and are negligible. K=200 (the existing soft-EM frontier
width) is probably enough.

This is the same K-NN-sparsification trick that makes mean-shift and
spectral-clustering tractable on large data. The bookgap repo already
has the HNSW K-NN infrastructure — applying it to the posterior
computation is straightforward.

## Test plan

1. Modify `gap_soft_em.py` (or fork as `gap_soft_em_fullmix.py`):
   replace `compute_state`'s size-matched-local-boundary construction
   with K_total over y's K-NN (size = K-NN budget).
2. **T11303 first.** Compare:
   - Basin count vs. current soft EM (130 basins).
   - Top-N basin overlap vs. current soft EM (Jaccard-on-frozensets).
   - Wall time per seed (expected: similar; K-NN cost replaces
     boundary-construction cost).
3. **Threshold reinterpretation.** With proper mixture semantics,
   τ=0.5 means "more likely than not to be in G." Likely produces
   different basin shapes than the current empirical-τ=0.5
   (which is "more in-cluster than out among nearby points").
   Calibrate empirically and document.
4. **Books-as-members behavior.** Run with books participating in
   weighting: confirm a book pulled into G with w=0.7 contributes
   appropriately to K_G and K_¬G of nearby articles.
5. **Cross-check with multi-scale labeling.** Multi-scale post-hoc
   uses `compute_state` posteriors. If we change the metric, multi-scale
   needs to use the new posterior too. Verify drop-in compatibility.

## Cost vs. value

Cost: ~1-2 days. The change is localized to `compute_state` and the
inner-loop convergence check. Test plan is straightforward (T11303
runs in 5-15 min per variant).

Value: theoretical foundation; eliminates the calibration burden and
the Task-7 in-G-skip rule; potentially cleaner basin shapes; required
for any future work that wants strict EM semantics (e.g., DP-mixture
with automatic K determination, Bayesian posterior over basins).

The original `cohesion_design_axes.md §H` declared local CSLS-balanced
the winner because the alternatives tested at the time (global corpus
prior, distance-ratio centroid) failed for unrelated reasons. The
"weighted full-data marginal via K-NN sparsification" alternative
wasn't tested. This backlog item is to test it.
