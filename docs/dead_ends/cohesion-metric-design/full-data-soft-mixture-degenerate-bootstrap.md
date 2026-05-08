# Full-data soft-mixture posterior — degenerate from single-seed init

The strict 2-component mixture posterior `P(z=G | y) = K_G / K_total`
over y's K-NN neighbourhood is mathematically equivalent to the proper
mixture-model posterior. Implemented as
`artifacts_scripts/gap_soft_em_fullmix.py` (since reverted to dead
end). **Fundamentally degenerate** from single-seed initialization: no
positive feedback to bootstrap cluster mass.

## What was tested

T11303 (14k works, 145 books). Single-seed run from article 0:

```
Running fullmix from seed 0...
Weight vector stats:
  nonzero count:    243
  max weight:       0.0002
  p50 of nonzero:   0.0001
  p99 of nonzero:   0.0002
  |w > 0.5| = 0    |w > 0.3| = 0    |w > 0.1| = 0
  |w > 0.05| = 0   |w > 0.02| = 0   |w > 0.005| = 0
```

The seed starts at `w=1`; after 1 iteration max weight drops to
~0.0002 across ~243 candidates. Subsequent iterations don't recover.
Whole-corpus run (T11303 with τ_supp=0.5): 4000 seeds processed, 0
basins discovered, all classified `too-small`.

## Why it fails (the math)

At iter 1 with seed initialization (`w[seed]=1`, others `w=0`) and K=200:

```
K_G(seed)     = 1 · kern(seed, seed) + Σ_{x ∈ KNN(seed)} 0 · kern(seed, x) = 1
K_total(seed) = 1 + Σ_{x ∈ KNN(seed)} kern(seed, x) ≈ 1 + 30
post(seed)   = 1 / 31 ≈ 0.032
```

For non-seed candidates `y`:

```
K_G(y)     = w(seed) · kern(y, seed) ≈ 0.95   (nearest neighbours)
              ≈ 0  (far neighbours)
K_total(y) ≈ 1 (self) + Σ kern over y's K-NN ≈ 1 + 30
post(y)   ≈ 0.030  (nearest)  to  ~0  (far)
```

So **every point's posterior becomes ~0.03 in one iteration** —
roughly `1/K_total` regardless of how close it is to seed (because
single seed has only one kernel-mass-of-1 contribution to spread
around). The cluster's "total weight" (Σ w) actually *increases* from
1 to ~243 × 0.0001 ≈ 0.024 after the iteration — the seed's mass
diffuses outward then evaporates further.

Subsequent iterations: `K_G(y)` is dominated by `w(x) · kern(y, x)`
where `w(x) ≈ 0.0001` for all x — orders of magnitude smaller than
`K_total(y) ≈ 30`. Posteriors collapse to ~10⁻⁵ and effectively zero.

## Why size-matched local-CSLS works where this doesn't

The current `gap_soft_em.py` (CSLS-balanced boundary):

```
K_G(y)  = Σ_x w(x) · kern(y, x)        # over support, |support|≈|G|
K_B(y)  = Σ_b f(b) · kern(y, b)        # over boundary, |boundary|≈|G|
post(y) = K_G / (K_G + K_B)
```

Crucial: **|B| is size-matched to |G|**. For a tight cluster with
|G|=2 (after bootstrap), |B|≈2. Both K_G and K_B are sums over ~2
points each, so they're comparable in absolute scale. The post=0.5
threshold has a stable interpretation regardless of cluster size.

In contrast, the full-mix variant has K_total summed over a fixed
K=200 K-NN window. K_total is always ~30 in absolute scale, but K_G
scales with |G|·avg_kernel — tiny for small |G|.

So the size-matching in CSLS-balanced isn't just a calibration
convenience — it's the mechanism that gives the EM iteration positive
feedback to concentrate weight. Without it, the EM diffuses to a
uniform-prior fixed point (every point with w ≈ |G|/N).

## What would be required to make full-mix work

Plausible fixes, none of which were tested here:

1. **Adaptive K**: `K_dynamic = max(2 · |support|, K_min)`. As cluster
   grows, K-NN window grows with it, keeping cluster fraction
   `|G|/K ≈ 0.5` at the meaningful threshold. But: requires careful
   handling of the growth dynamics; may oscillate.
2. **Bootstrap with multiple initial members**: seed K-NN of size
   ~10-50 all with `w=1` initially. Provides cluster mass to start.
   Fundamentally an initialization fix; doesn't address the underlying
   diffusion-to-uniform fixed point.
3. **Parameterized cluster density**: replace `p(y | G)` (KDE on G)
   with `Gaussian(centroid_G, σ_G)`. M-step updates centroid and σ from
   weighted members. E-step computes Gaussian likelihood, not KDE
   over weighted members. This is standard GMM EM; cluster shape is
   parametric. Different model entirely.
4. **Temperature annealing on the kernel**: start with very narrow σ
   (concentrated kernel) and gradually broaden. Concentrates initial
   mass, then expands.
5. **Posterior renormalization**: `post(y) = K_G(y) / |G|·(K_total(y)/N)`
   — divide by expected per-cluster mass. Restores the "more likely
   than random" interpretation independent of |G|.

(5) is theoretically interesting but wasn't tested.

## Conclusion

The size-matched local CSLS-balanced boundary in
`docs/cohesion_design_axes.md §H` isn't just a tractability dodge —
it's a **necessary calibration trick** for kernel-density EM with
small cluster sizes. Replacing it with the strict mixture posterior
`K_G/K_total` is mathematically correct but loses the positive
feedback that makes EM converge from a single seed.

The backlog item was theoretically appealing but underestimated the
bootstrapping problem. Keeping the size-matched local-CSLS approach
as the production design is justified by both empirical performance
and this dynamic-stability analysis.

If a future attempt wants to revisit this, the most promising
direction is probably **(1) adaptive K** combined with **(3)
parameterized cluster density** — which would essentially be a
Gaussian mixture model with K-NN sparsification. That's a much
larger restructure than just swapping the posterior formula.

## Files

- `/home/gchurchill/src/bookgap/artifacts_scripts/gap_soft_em_fullmix.py`
  remains in tree as the implementation reference (it works
  mechanically, just produces degenerate output). Could be removed if
  cluttering, or kept as a "this is what you'd build if pursuing the
  alternative."
- Diagnostic at `/tmp/fullmix_diagnostic.py` (single-seed inspection)
  produced the converged weight stats above.
