# Global corpus background degenerates in high dimension

**Idea**: define soft membership for each gap article relative to the *whole corpus* (e.g., `K_G(i) = mass of in-gap neighbors at i`, `K_corpus(i) = mass of any-corpus neighbors at i`, `post(i) = K_G / (K_G + K_corpus)`).

**What happened**: in 384-dim, posteriors saturate near 1.0 for any moderately-tight gap. Same root cause as [centroid-Bayesian likelihood](centroid-bayesian-likelihood-high-dim-degenerate.md): high-dim Gaussian normalizers swamp local cohesion differences when comparing against a global prior.

**Resolution**: **local CSLS-balanced background** — for each gap article i, compute its 1-NN book b_i, count `f(b) = #(gap articles claiming b)`, take b's f(b) nearest non-gap articles. Total background ≈ |G|, balanced per book. The σ_i × σ_j product in the self-tuning Gaussian kernel cancels the dimension-d normalizer cleanly.

**Salvage paths**:
- Bayesian shrinkage of `K_corpus` toward `K_local`
- Subset the corpus to a "neighborhood of relevance" — but this just becomes the local CSLS-balanced background by another name

**Lesson**: in high-dim, "background" must be local and commensurate with the foreground in size (`|B| ≈ |G|`); global corpus priors are too unbalanced to produce discriminative scores.
