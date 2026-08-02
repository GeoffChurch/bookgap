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

## Addendum: the deeper structural reason (no initialization fixes it)

A natural follow-up question: can clever initialization salvage the
formulation? E.g. seeding with the seed's K-NN at uniform weight, or
with a Gaussian-shaped initial bump `w(x) ← kern(seed, x)`?

**No, and here's why.** The iteration

```
new_w(y) = (Σ_x kern(y, x) · w(x)) / (Σ_x kern(y, x))
         = (K · w)(y) / (K · 1)(y)
```

is exactly **power iteration on the stochastic kernel-random-walk
matrix `D⁻¹ K`** (where `D` is the kernel-row-sum diagonal). For a
connected kernel graph this iteration's fixed point is the
**stationary distribution of the random walk** — approximately the
uniform / data-marginal distribution. The cluster around the seed is
*not* this distribution. The seed's identity is forgotten in O(1/λ₂)
iterations, where λ₂ is the spectral gap.

Said differently: clustering structure lives in the *lower*
eigenvectors of the random-walk Laplacian (this is the spectral
clustering insight). The dominant eigenvector — which is what plain
power iteration converges to — is uniform up to local density. **No
amount of cleverness in the initial `w` changes the fixed point.**

Quick analytical check, kernel-init `w(x) ← kern(seed, x)`:
- iter 2: `w(y) ← Σ_x kern(seed,x)·kern(y,x) / Σ_x kern(y,x)` — a
  smooth bump around seed.
- subsequent iters: the bump broadens. After enough iterations, `w`
  approaches the stationary degree distribution.

This means **size-matched local-CSLS is the regularizer that breaks
the random-walk drift**, not just a calibration trick. By making
`K_B` grow with `G` (size-matched to support), the iteration becomes:

```
post(y) = K_G(y) / (K_G(y) + K_B(y))
```

which is *not* power iteration — it has an anchor that prevents
diffusion. Interior points (`K_B` small there because the boundary
isn't local to them) stay high; exterior points (`K_B` large) drop.
The size-matching of `B` to `|G|` is what makes the iteration
anti-diffusive.

### Salvage paths (each is a structural fix, not just initialization)

In rough order of how much they preserve "kernel-only soft EM" vs.
how much they restructure:

1. **Anchor the seed**: lock `w[seed] = 1` throughout. Prevents
   diffusion-to-uniform; converges to a smooth bump around seed.
   But: privileges the seed, threshold still needs recalibration,
   doesn't handle "seed on cluster boundary."
2. **Signed iteration with anchor `K_B`** — size-matched local-CSLS,
   i.e. `gap_soft_em.py`. The validated production design.
3. **Parameterized cluster density (proper GMM EM)**: `p(y|G) =
   N(y; μ_G, σ_G)`, M-step updates parameters. Cluster has explicit
   shape; doesn't degenerate. Standard machinery; bigger restructure.
4. **Multi-cluster competition (lockstep / global EM)**: the
   constraint `Σ_k post_k = 1` across clusters prevents any single
   cluster from collapsing to uniform. See
   [multicluster-softmax-membership](../../backlog/cohesion-metric-design/multicluster-softmax-membership.md)
   — the worked-out version of this salvage path, with per-member
   normalization providing the anti-diffusion regularizer that's
   missing from the naïve `K_G/K_total` formulation diagnosed above.
   Generalizes the older `lockstep-expansion` backlog entry.
5. **Spectral cluster discovery**: find clusters as low eigenvectors
   of the kernel Laplacian. Different algorithm entirely; well-studied
   but not in this codebase.

(2) is what we have. (1) and (3) are the natural EM-flavored
alternatives if we want to revisit. The fundamental observation
generalizes: **kernel-only iterations need anti-diffusion to find
clusters; you can't bootstrap your way around it.**

## Files

- `artifacts_scripts/gap_soft_em_fullmix.py`
  remains in tree as the implementation reference (it works
  mechanically, just produces degenerate output). Could be removed if
  cluttering, or kept as a "this is what you'd build if pursuing the
  alternative."
- Diagnostic at `/tmp/fullmix_diagnostic.py` (single-seed inspection)
  produced the converged weight stats above.
