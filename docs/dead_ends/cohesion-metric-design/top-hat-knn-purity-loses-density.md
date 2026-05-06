# Top-hat K-NN purity loses local density structure

**Idea**: for each gap article i, look at its K nearest neighbors (top-hat / unweighted) and count how many are in G. Mean over G articles gives a "purity" score; sum gives effective size. Parameter-free if K = |G|.

**What happened**: the top-hat treatment loses information that *weighted* K-NN purity preserves. A gap with a tight inner core and a leaky boundary scores the same as one that's uniformly half-mixed because top-hat weights all K neighbors equally regardless of distance.

**Diagnosis**: kernel-density estimation theory — top-hat is the simplest kernel but has discontinuities; smoother kernels (Gaussian, Epanechnikov) are typically preferred for stable density estimates.

**Resolution**: **inverse-distance-weighted K-NN purity** within the K-NN truncation:
```
purity_w(i) = Σ_{j ∈ K-NN(i), j ∈ G} (1/d_ij²) / Σ_{j ∈ K-NN(i)} (1/d_ij²)
```
Or, with the dimension-aware fix (see [inverse-square-kernel-dimension-inappropriate](inverse-square-kernel-dimension-inappropriate.md)), the **self-tuning Gaussian** form `exp(−d²_{ij} / (σ_i σ_j))`. Compute both top-hat and weighted; agreement is reassuring, divergence is informative (top-hat-high but weighted-low ⇒ uniform leakage; weighted-high but top-hat-low ⇒ tight core with frayed boundary).

**Lesson**: weighted variants of K-NN-style measures generally subsume the unweighted top-hat; the cost is negligible (you already have the K-NN distances cached) and the gain is real local-density signal.
