# Inverse-square kernel is dimension-inappropriate

**Idea**: weight neighbors by `1/d²` (inverse-square) for kernel-density-style cohesion. Has a singularity at d → 0 but is parameter-free (no bandwidth to tune).

**What happened**: the `1/d²` form is the Newtonian Green's function of the Laplacian in **3D** Euclidean space. In d-dimensional space the Green's function is `1/r^(d-2)`. For 384-dim GIST embeddings, the canonical "Newtonian" kernel would be `1/r^382` — essentially a delta function.

**Diagnosis**: pointwise inverse-power kernels are dimension-specific, and `1/d²` is masquerading as the right kernel in high-dim when it's actually a 3D-specific form. The singularity at d → 0 is the model's complaint that the dimensionality doesn't match.

**Resolution**: **Gaussian kernel with self-tuning bandwidth**. The Gaussian (heat kernel) is the dimension-independent canonical kernel — Green's function of the heat equation in any-dimensional Euclidean space. Self-tuning bandwidth (Zelnik-Manor & Perona 2004; t-SNE perplexity) sets `σ_i` from i's local k-NN distance, removing the global-bandwidth calibration problem.

The product form `K(i, j) = exp(−d²_{ij} / (σ_i σ_j))` is symmetric, scale-invariant under bandwidth normalization, and high-dim-stable.

**Lesson**: when picking kernels for high-dim data, prefer the heat-kernel family (Gaussian) over Newtonian (`1/r^k`) forms. The latter are dimension-specific.

**Related**: harmonic mean of squared distances `HM(d²) = K / Σ(1/d²)` is just a renormalization of inverse-square — same singularity, same dimensionality issue.
