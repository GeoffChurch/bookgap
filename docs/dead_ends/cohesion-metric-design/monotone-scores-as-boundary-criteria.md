# Monotone scores (effective_size, logit_sum) as boundary criteria

**Idea**: use a per-gap quality score (effective_size = Σ post, or logit_sum = Σ log(p/(1-p))) as the stopping criterion for cohesion-aware expansion: keep adding articles while the score increases; stop when it stops improving.

**What happened**:
- **effective_size**: monotonically non-decreasing in |G| because each new article contributes a non-negative posterior. No peak ever forms within practical K_max. Result: gaps grow until hitting K_max=500 cap, regardless of their natural cohesive size.
- **logit_sum**: has theoretical peaks for tight clusters (kb-144 Filipino peaked at 180.5 at k=40, then declined) but for clusters whose natural size exceeds K_max (kb-2 with natural size > 500), it stays monotonic within range. For practical purposes, *also* fails as a boundary criterion.

**Diagnosis**: the per-article posterior is a *probability of belonging* — adding a marginal article with post ≈ 0.5 contributes 0.5 to `effective_size` and ≈ 0 to `logit_sum`. Neither score dramatically *decreases* until the gap engulfs nearby noise *en masse*, which only happens at huge sizes.

**Resolution**: separate the two roles cleanly:
- **Boundary criterion** (per-gap, where to stop): `min_post ≥ τ` (default τ=0.5). Catches the cluster's *worst* member dropping below threshold — fires before total scores would have peaked. This is what we shipped.
- **Ranking score** (cross-gap, sort surviving gaps): logit_sum (canonical Bayes evidence, monotone in size — fine for ranking, just not for stopping).

**Lesson**: don't use the same score for "where to stop adding articles" and "how to rank gaps." They're structurally different questions with different mathematical requirements (a stopping criterion needs a peak; a ranker doesn't).
