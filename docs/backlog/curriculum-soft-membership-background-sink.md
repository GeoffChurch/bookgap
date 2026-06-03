# Soft topic membership: the boundary-background (∅-sink) and its family

**Origin (conversation 2026-06-03).** Design review of the validated embedding-soft keyword
membership ([[curriculum-fuzzy-soft-topic-membership]]) asked whether the *boundary/local-background*
principle that won elsewhere (gap-selection's `log p/g_local`; the curriculum-marginal keyword lift
instead of the global 6.8M corpus) is applied inside `soft_responsibilities`. It is **not** — and
adding it, in its canonical form, is the highest-leverage improvement. This doc is the living plan for
that work plus the same-family refinements.

## The gap: the softmax has no ∅/background sink

`gap_fuzzy_topics.soft_responsibilities` returns `softmax(sims/τ, axis=1)` — rows sum to 1 **over the N
chapters**, no "belongs to none" outlet. So a work far from *every* chapter (a foundation-wander
intrusion — IoT, profilometry; or Color Image's catalog-blurb) still dumps its full unit of mass into
its least-far chapters and pollutes them. `soft_floor` (0.05, applied downstream as `resp >= floor`)
does **not** substitute: a *diffuse* work whose responsibilities spread ~uniformly (≈1/13 = 0.077
each) **exceeds the floor on many chapters at once → contributes to all of them** (max pollution),
while a peaked work contributes to one. This is precisely the increment-1 Color Image failure (soft
pulled in *more* boilerplate) and the soft analog of what gap-selection already does right: commit
only above the local ambient.

---

## Method 1 (PRIMARY, in progress) — canonical ∅-sink = vMF mixture with an ambient-null component

**Not a quantile, not a scalar threshold — a *direction*.** `softmax(κ·cosine)` already *is* a von
Mises–Fisher mixture posterior (κ = 1/τ); it's just missing a component. Add one more vMF component
whose mean direction is the **pool centroid** μ₀ (the data's own mean direction), same concentration κ,
uniform mixing prior.

### Derivation

Unit embeddings `xᵢ ∈ Sᵈ⁻¹`. vMF density `f_k(x) = C_d(κ)·exp(κ μ_kᵀx)`; the normalizing constant
`C_d(κ)` is identical across all components (same κ) so it **cancels**. Posterior responsibility:

```
            e^{κ c_ik}
r_ik = ───────────────────────────,      c_ik = μ_kᵀ xᵢ   (chapter cosine)
        e^{κ c_i0} + Σ_j e^{κ c_ij}       c_i0 = μ_0ᵀ xᵢ   (ambient cosine)
```

Divide numerator and denominator by `e^{κ c_i0}`:

```
            e^{κ (c_ik − c_i0)}
r_ik = ─────────────────────────────
        1  +  Σ_j e^{κ (c_ij − c_i0)}
```

So the ∅-sink **is just `softmax` over the ambient-subtracted similarities `(c_ik − c_i0)`, with ∅ the
implicit logit-0 baseline** (the `1` in the denominator). A work leaks to ∅ exactly when **no chapter
beats the ambient direction** (`c_ik < c_i0 ∀k`). This is the boundary-background move (commit only
above the local null) and *literally the probabilistic version of the `--mean-center` already used for
Track B assignment* (subtract the common direction so chapters separate) — one unifies the other.

### Why it's the right first move

- **Zero new knobs.** μ₀ is the pool centroid; uniform priors cancel; κ comes from the existing
  calibration; the vMF constants cancel. The textbook generative model for "points on a sphere, some
  belonging to no cluster" is exactly a vMF mixture *with a uniform/ambient null* — we simply omitted
  the null. ("What a statistician does without fear.")
- **Strict generalization → de-risked.** For a distinctively-placed work one chapter dominates and
  ∅≈0, so `argmax` is preserved (the "reduces to today" property [[curriculum-fuzzy-soft-topic-membership]]
  wanted). Only *diffuse* works change — the point.
- **Fixes the real failure.** Works closer to the ambient than to any chapter (boilerplate,
  foundation-wander) drain to ∅ instead of polluting chapters. Inverts the increment-1 Color Image
  result (blurb is ambient → leaks out instead of being re-attracted).
- **Already local.** The pool *is* the gap neighborhood (the ~5000 selected works), so its centroid is
  the local ambient — not the global-6.8M centroid. The locality we want is built in.

### Design choices

- **Ambient direction μ₀ = mass-weighted pool centroid**, normalized: `μ₀ = normalize(Σᵢ massᵢ xᵢ)`
  — consistent with the rest of the pipeline (everything is mass·r weighted); the ambient is then "where
  the typical *important* work points." **(Chosen for the first cut.)**
  *Alternative to revisit:* plain (unweighted) mean `normalize(Σᵢ xᵢ)` — the pure geometric ambient,
  importance-agnostic. Worth A/B-ing if mass-weighting makes μ₀ chase a few high-mass works; cheap to flip.
- **κ stays calibrated on the K chapters only** (the existing `_calibrate_tau` over `sims`), then the
  ambient column is appended at that κ. This preserves the "~1.5 chapters per *committed* work"
  semantics; ∅ siphons only the below-ambient mass. (Alternative: calibrate over K+1 — shifts κ, treats
  ∅ as a "chapter"; less clean. Note and skip unless #1 misbehaves.)
- **Downstream unchanged:** `w_ik = massᵢ · r_ik`; the leaked `r_i∅` mass is simply discarded (the
  "belongs to no chapter" outlet). Diffuse works now fall below `soft_floor` on every chapter and
  contribute nowhere — which is the fix.

### Implementation sketch

`soft_responsibilities(E_unit, C_unit, perplexity, ambient=None)`: compute `sims = E_unit @ C_unitᵀ`,
calibrate τ on `sims`; if `ambient` (unit μ₀) given, append `amb = E_unit @ ambient` as a (K+1)th
column, softmax the augmented matrix at τ, return `R = R_aug[:, :K]` (chapter responsibilities, rows
sum to 1−r_∅). Callers (`summarize_topics`, `book_toc_reconstruction._materialize_track_b`) compute
`μ₀ = normalize(Σ massᵢ Eᵢ)` from the filtered embeddings + masses and pass it. Default `ambient=None`
⟹ today's behavior (back-compat for the bake-off).

---

## Method 2 — per-work concentration (κ per element)

κ is still global. Two ways to make it per-element; A/B them:

- **Per-work τ (t-SNE bandwidth).** `_calibrate_tau` finds a *single* τ for the *mean* perplexity ≈
  1.5; the "per-work adaptivity" the backlog credits for embedding-soft beating citation-soft is only
  approximate. Per-work τ (each work hits ~1.5, like t-SNE's per-point β) makes dense- and sparse-region
  works commit by *relative* geometry, not absolute scale. Cheap at N=13 (bisect per row).
- **Per-chapter κ by vMF MLE (more canonical — no perplexity knob).** Banerjee et al. (2005) closed
  form `κ̂ ≈ R̄(d − R̄²)/(1 − R̄²)`, `R̄` = mean resultant length of the chapter's members. A tight
  chapter ⟹ high κ (peaked), a diffuse one ⟹ low κ — read off the data, no target perplexity.
  **Tradeoff:** drops the knob, but the backlog credits *perplexity*-adaptivity for the win, so MLE-κ is
  a behavior change to validate, not a free swap. Test MLE-κ vs per-work-τ head to head.

---

## Method 3 — robust thin-chapter anchors

Responsibility centroids are hard-cluster *means*; a 1-work cluster's centroid *is* that one (often
boilerplate) work — so soft re-attraction chases the blurb. Use a top-mass **core set / medoid** per
chapter (the absorbing-walk design's `S_k`), or shrink thin-cluster centroids toward μ₀. Complementary
to #1 (∅ helps a work *leave* a bad chapter; robust anchors fix the bad *centroid*), but lower urgency
once #1 drains diffuse mass — fewer degenerate centroids survive.

## Method 4 — per-member normalization (mirror the shipped gap-side)

The gap-side `gap_softmax_enrich.py` uses **per-member normalization** (size-fairness) rather than a raw
softmax. Orthogonal to the ∅ and can ride along; mirror it for size-fair chapter pools.

---

## Where the same idea helps other aspects

- **dynamic-k floor → per-topic ESS cut.** The keep-threshold is a global relative floor (0.05).
  Replace with the topic's **effective size** `ESS = (Σ mass·r)² / Σ(mass·r)²` so a genuinely thin
  topic honestly shows few keywords and a rich one many — no global knob (participation-ratio, already
  surfaced as `summary.md`'s "Effective # of Works"). Flagged in [[curriculum-fuzzy-soft-topic-membership]].
- **Seeded bipartite-PPR edges are already degree-normalized** (keyword degree = document frequency),
  so generic-keyword hubness is damped to first order; **lift-weighting the edges** (`B[d,t] = log-lift`
  not binary) would bind works to their *distinctive* keywords — folding the boundary-background lift
  into the graph. Lower priority (PPR refuted on silhouette, `docs/dead_ends/curriculum/keyword-coclustering.md`).

## Alternatives considered (do NOT do first)

- **Absorbing-walk arm gets ∅ "for free"** (the (1−α) teleport leak is a background sink) and is
  citation-aware — *but it was refuted*: no global α (even per-node adaptive + class-balance) separates
  "dissolve the degenerate chapter" from "genericize the rest." Its ∅ comes **bundled with the citation
  diffusion that genericizes.** The vMF-null gets the same ∅ benefit on the **embedding axis alone** —
  the axis the keyword-coclustering dead-end confirmed is the good one. This is the argument for #1 over
  reviving the walk. ([[curriculum-fuzzy-soft-topic-membership]])
- **Archetypal analysis** (each work = convex combination of N "archetype" works) — "the one principled
  alternative", very interpretable soft memberships, but embedding-only, a heavier optimization, and no
  natural ∅. Fallback, not first.

## Eval

The ∅ sink should: (a) leave peaked works' responsibilities ~unchanged (`argmax(r) ≈ current hard
labels` → the generalization still holds — **validate this first**); (b) move diffuse/off-topic works'
mass to ∅ (mean chapters/work drops for the intrusions, not the core); (c) on Color Image, dissolve the
blurb chapter's boilerplate *without* re-attracting more (the increment-1 failure inverts). Judge via
the ToC-reconstruction panel + `gap_intrinsic_metrics.py` (watch kw_gap_pmi / kw_distinct rise without
silhouette dropping). Bake-off: ambient=None (today) vs mass-weighted μ₀ vs plain-mean μ₀.

## Status / working notes

- **2026-06-03:** doc written; canonical derivation settled (softmax over ambient-subtracted sims, ∅ =
  implicit logit-0). Starting Method 1 with **mass-weighted** μ₀; plain-mean noted as the A/B alternative.
- **2026-06-03:** Method 1 IMPLEMENTED (`soft_responsibilities(..., ambient)`, `KeywordConfig.soft_background
  ∈ {none,mass,mean}`, threaded through summarize_topics / run_keyword_extraction /
  `reextract_keywords_topk --soft-background` / `book_toc_reconstruction --soft-background`; commit 165d64d).
  **Mechanism validated on real trackA_sem (5000 works, mass μ₀):**
  - `argmax(soft+∅) == argmax(soft,none)` = **100%** — ∅ never changes the winning chapter (strict
    generalization holds exactly). (`argmax == hard-citation-label` = 51.8%, the pre-existing
    soft-embedding-vs-hard-citation gap, unchanged by ∅.)
  - **Surgical leak:** median ≈ 0, only **4.7%** of works leak >0.5; mass-weighted leak 0.063 < unweighted
    0.076 (core retained, periphery drained).
  - **Drains the right works:** the heaviest-∅ works are off-topic (mechatronics, hadron calorimeters),
    paratext (special-issue editorial, NEUROPULS press release), foreign-language (the Korean/Spanish
    foundation-wander), and pseudoscience ("Photonic–Gravitational Transduction"). Lowest-∅ = core
    ML/NN papers. Exactly the intrusions that shouldn't define chapter keywords.
  - **Keyword payoff on the RELEVANCE+soft stack: NEGLIGIBLE.** Full re-extract `neuromorphic_trackA_sem_bg`
    (soft+∅, mass μ₀) vs `trackA_sem_eval` (soft, no ∅): silhouette 0.192=0.192 (unchanged — argmax is
    identical), prereq 0.710=0.710, **kw_distinct 0.876 vs 0.867 (+0.009), kw_gap_pmi 3.770 vs 3.782
    (−0.012)** — both within noise; tail keywords just reshuffle. **Diagnosis: the ∅ is largely
    REDUNDANT with RELEVANCE doc-weighting on this pipeline.** The intrusions the ∅ drains are mostly
    *low-mass* (the mass-weighted leak 0.063 < unweighted 0.076 confirms it), and RELEVANCE already gives
    them ≈0 weight — so removing them changes the mass·r-weighted keyword statistics almost not at all.
    Put positively: **the pipeline's mass-weighting was already an approximate boundary-background**
    (down-weight the periphery); the explicit vMF-∅ makes it canonical/direction-aware but adds little on
    top. The ∅'s *unique* contribution — draining a **high-mass** but off-topic work (kept by RELEVANCE,
    drained by ∅) — is rare on this corpus (the high-∅ works here are all low-mass).
  - **So: keep the ∅ as a principled, harmless OPT-IN** (it's the correct generative model, a safety net
    for EQUAL-weighted / relevance-absent contexts, and theoretically cleaner) — but it is **not a quality
    win on the default RELEVANCE+soft stack**, so not default. Don't bother with the mass-vs-mean A/B
    (mass is already marginal; mean would be too).
  - **Residual soft weakness is elsewhere.** Color Image's blurb chapter is a degenerate-**centroid**
    problem (the centroid *is* the blurb → blurb-neighbors have low ∅-leak, so ∅ won't fix it) → that's
    **Method 3 (robust/core-set anchors)**, the better-targeted next lever. Method 2 (per-work κ / MLE-κ)
    is the other open one. Both untested.
