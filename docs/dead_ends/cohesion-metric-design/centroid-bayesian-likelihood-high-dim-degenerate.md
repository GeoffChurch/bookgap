# Centroid-Bayesian likelihood degenerates in high dimension

**Idea**: fit a single isotropic Gaussian `N(μ_G, σ_G²)` to gap members and use a Bayes-mixture posterior `post(i) = π_G·p_gap(x_i) / [π_G·p_gap(x_i) + (1−π_G)·p_corpus(x_i)]` as soft-membership for effective-size scoring. (See `gap_effective_size_centroid.py`'s original Bayes formulation, since replaced by distance-ratio.)

**What happened**: in 384-dim, every gap got `posterior ≈ 1.0` regardless of cohesion. Top-20 by effective size was identical to top-20 by raw size — no discrimination.

**Diagnosis**: the Gaussian log-density has a normalization term `−(d/2) log(2π σ²)`. For 384-dim and gap σ_G ≈ 0.37 vs corpus σ_C ≈ 0.53:
```
log p_gap − log p_corpus  ≈  −192 × (log(σ_G²) − log(σ_C²))  ≈  +135 nats
```
That's a factor of `e^135` advantage for the gap Gaussian over the corpus Gaussian — *just from the dimensionality coefficient* — before we even look at where the article actually sits. Any moderately-tight gap (σ_G < σ_C) gets the full normalizer benefit; differences in actual cohesion get drowned out by `d/2 · log(σ_G/σ_C)`.

**Salvage paths**:
- Bayesian shrinkage of σ_G toward σ_C (regularize the variance estimator). Adds a hyperparameter.
- Skip absolute densities; work with **standardized distance ratios** instead: `m_gap(i) = ‖x_i − μ_G‖ / σ_G`, `m_corp(i) = ‖x_i − μ_C‖ / σ_C`, then `post(i) ≈ sigmoid(m_corp(i) − m_gap(i))`. **This is what we shipped in `gap_effective_size_centroid.py`** — cheap, stable, but coarser than the per-article KDE.
- Per-article kernel density (KDE) with self-tuning bandwidth — what `gap_effective_size_local.py` does. Robust because `σ_i` is set by local 7-NN geometry, not by fitting to small gap samples; the σ_i × σ_j product in the Gaussian kernel cancels the dimension-d normalizer in numerator vs denominator.

**Lesson (more general)**: in high-dim, parametric isotropic-Gaussian likelihoods don't produce discriminative scores. Need either non-parametric (KDE), distance-ratio, or regularized variants.

**Forward reference**: the [multicluster-softmax-membership](../../backlog/cohesion-metric-design/multicluster-softmax-membership.md) backlog item proposes a global multi-cluster softmax built on KDE likelihoods (not parametric Gaussians) for exactly this reason — the KDE+ratio formulation sidesteps the dimensional normalizer that doomed this approach.
