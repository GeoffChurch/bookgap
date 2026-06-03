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
