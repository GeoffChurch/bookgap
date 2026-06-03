# Keywords for cluster DEFINITION (bipartite co-clustering), not just evaluation

**Idea (from conversation 2026-06-02).** Make keywords first-class in clustering. Let `M` be the
work×keyword matrix (`M[w,k]` = keyword k's weight in work w — tf, or the log-odds score). The
bipartite power iteration `u → Mᵀu → M(Mᵀu) → …` (works → keyword-weights → works → …) converges to
the leading singular vector of `M` = **bipartite spectral co-clustering** (Dhillon; LSA): a cluster
is *jointly* a set of works AND their characteristic keywords, co-defined. Keywords stop being a
post-hoc label and become part of the cluster definition — which could fix the recurring
**vague/generic/overlapping chapter** defect, because a co-cluster is keyword-coherent by construction.

**The catch (recurring anti-diffusion lesson, 3rd time).** Plain power iteration on `MMᵀ` converges
to the GLOBAL dominant singular vector regardless of the seed — the seed washes out in
`O(1/spectral-gap)` steps, so there's no per-seed "basin of attraction." To make a seed define its
own co-cluster you need the anchor: **personalized PageRank on the bipartite work-keyword graph
(teleport to the seed)**, or deflation, or early-stop. Cf.
[[full-data-soft-mixture-degenerate-bootstrap]] and the citation-soft / absorbing-walk results in
[[curriculum-fuzzy-soft-topic-membership]].

## Increment 1 — nonlinear keyword↔work iteration VALIDATED (2026-06-02)

`gap_keyword_iteration.py`: the loose/nonlinear power iteration — run the real keyword extraction
(log-lift vs the LOCAL pool background) in the loop, `works → keywords → works (top-m by match) → …`
from a seed. Pool = the run-dir's selected works (the gap neighborhood = the boundary-local
background; the better ambient than the global corpus). Phrases via one shared unigram+bigram
vocabulary over the pool (the vocabulary/frequency split that resolves quantum_hall/hall_effect).

Neuromorphic, **unanchored**, seeds = chapters 5 and 11:
- **Converges** fast + stably (step-to-step Jaccard → 1.00 by iter ~5–7; no oscillation).
- **Seed-dependent basins exist:** seed 5 → a tight VCSEL cluster (`vertical_cavity, surface_emitting,
  vcsel`); seed 11 → a distinct photonic-NN cluster (`neuromorphic_photonics, photonic_neurons`).
  **Jaccard between the two fixed points = 0.05** — essentially disjoint. The nonlinearity beat
  linear mode-collapse with NO anchor, as predicted.
- Fixed points are **coherent + keyword-clean** (no debris) and *purer* than the seed (drift-from-seed
  ≈ 0.06–0.11 — it refines the seed into its keyword-coherent core).

So the premise holds: keyword-coherent fixed points exist, are seed-dependent, coherent. The bigger
builds below are de-risked. `K2W` (keyword→work scoring) is built and reusable. Next: multi-seed /
deflation to make a full curriculum of these basins, then the term-view, then the full co-clustering.

## Increment 2 — keyword-DEFINED curriculum (deflation) tested; motivates multi-view (2026-06-02)

`gap_keyword_curriculum.py`: deflate the pool into N keyword-coherent basins (highest-mass seed →
iterate → record → remove → repeat), order by citation balance, write a judge/intrinsic/
correspondence-ready run-dir. Neuromorphic (13 basins): several excellent + distinct (VCSEL,
reservoir computing, diffractive NN, CNN, optical-comm, activation functions). Head-to-head intrinsic:

| metric | keyword-curric | book ToC | dendrogram |
|---|---|---|---|
| silhouette | **0.078** | 0.154 | 0.192 |
| kw_distinct | **1.000** | 0.952 | 0.908 |
| kw_gap_pmi | 2.713 | 4.210 | 3.635 |
| prereq_consist | 0.678 | 0.534 | 0.710 |

So pure keyword-clustering **maxes keyword-distinctiveness (1.0, by construction — the target defect)
but tanks embedding silhouette (0.078) and gap-fidelity (2.7)**; judge still book_toc. **Two lessons:**
(1) keyword-coherence ≠ embedding-coherence (keyword-coherent clusters are embedding-spread), so
optimizing keywords ALONE sacrifices cohesion → **use multi-view (keyword + embedding + citation
together)**, not keywords alone; (2) keyword-clustering **organizes but doesn't filter** — the pool's
foundation-wander surfaced as off-topic-but-coherent basins (intrusion-detection, QKD) that drag
gap-PMI down → gap-selection (product-of-experts) is still needed upstream. This directly motivates
the term-view-in-multi_view_spectral as the *balanced* version.

## Increment 3 — naive multi-view(+term) FAILS; views conflict (2026-06-02)

`gap_multiview_term.py`: build the three CSLS-corrected views (embedding + citation + **term**) over
the pool, combine by SUM, cluster. `SpectralClustering` collapsed to one giant cluster (4975 + 12
tiny) — spectral-on-dense-affinity failure. Recursive-Fiedler `graph_bisect` gave *balanced* clusters
(275–606) but **silhouette = −0.024** (worse than random!), kw_distinct 1.0, kw_gap_pmi 2.8. So a
**naive sum of the three views wins NEITHER cohesion (negative silhouette) nor gap-fidelity** — only
the trivial kw_distinct. **The views genuinely conflict** (keyword-coherence vs embedding-coherence vs
citation-coherence pull apart); summing them and cutting the combined graph is incoherent.

**Conclusion of the keyword-coclustering thread (3 increments):** the nonlinear iteration is a nice
*primitive* (validated, seed-dependent coherent basins), but turning it into a full curriculum —
keyword-deflation (inc 2) or naive multi-view (inc 3) — **does NOT beat the validated dendrogram +
product-of-experts + dynamic-k stack** on the structural axes, because keyword/embedding/citation
coherence conflict and naive combinations sacrifice cohesion. **The one remaining proper test is the
co-regularized 3-view `multi_view_spectral`** (generalize the existing 2-view Kumar-Rai-Daumé solver
to add the term view — it *balances* views via the co-reg term `L_v − λ Σ_{w≠v} U_w U_wᵀ` instead of
summing, which is designed to avoid exactly this conflict). That's the "full unification" and the only
version not yet refuted; needs generalizing the solver + a pipeline rebuild. Until then the dendrogram
stays.

**Concrete directions (cheapest first):**
1. **Add a keyword/term VIEW to the existing `multi_view_spectral` clustering** (it already
   co-regularizes an embedding view + a citation view; the term-document matrix is a natural third
   view). Brings keywords into cluster definition with NO new pipeline; directly judgeable.
2. **NMF on `M`** (non-negative → interpretable parts-based co-clusters; non-negativity matches
   "membership"). = topic-modeling the term-document matrix.
3. **Seeded bipartite PPR**: gap-seed → personalized diffusion on works×keywords → the co-cluster
   (works + keywords) it defines. Unifies gap-seeding + clustering + keywords in one step.
4. **The deep unification**: replace gap-seed → citation-PR → rank → embedding-cluster →
   extract-keywords (5 stages) with gap-seed → seeded multi-view co-clustering over
   works × {keywords, citations, embedding-neighbors} → joint co-clusters with keywords built in.

**Caveats:** (a) inherits keyword hygiene — debris keywords pollute `M`, so the keyword view wants the
dynamic-k / keyword-gap-PMI cleaning already built; (b) must beat the validated stack (semantic-rerank
+ dynamic-k) — its distinctive bet is "keyword-describable clusters fix vagueness," a real unaddressed
defect, so test it on that. Eval: the ToC-reconstruction panel + the intrinsic metrics
(`gap_intrinsic_metrics.py`: silhouette, kw_distinct, kw_gap_pmi) — does it raise keyword
distinctiveness/fidelity without hurting silhouette/prereq-order?
