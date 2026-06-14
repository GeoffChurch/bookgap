# Curriculum applications — the full map

*Status: living doc / brainstorm capture (2026-06-10). Source: the curriculum_lean matrix-experiment
session. Ranked by leverage of machinery this repo already has; includes the unglamorous entries on
purpose. Companion empirics: `docs/handoffs/2026-06-10-usage-difficulty.md` (premise-surprisal
difficulty model), `docs/handoffs/2026-06-10-fewshot-arms.md` (worked-example arms).*

## The unifying frame

A curriculum is **a policy for spending a budget over a dependency graph**. The budget varies by
application — context tokens, training steps, eval items, human attention — but the artifact is the
same: an *ordering* plus a *weighting* of a dependency closure. Two session findings shape every
application below:

1. **Value is capability-relative.** Curriculum-as-context helped a competent specialist
   (Goedel-8B) and hurt/was useless to a weak generalist (gemma) — reference material requires
   capability to exploit. Every deployment needs a consumer-capability gate.
2. **The difficulty model may be worth more than the curriculum.** Ordering is cheap once you can
   measure "what is at this consumer's edge" (premise surprisal under the usage distribution ≈
   training-prior surprisal). Every application below *consumes* the difficulty oracle; only some
   consume the ordering. If the oracle validates, it is the product; curricula are one output.

## Applications, ranked

### 1. Frontier selection for expert iteration / RLVR (highest leverage)
Self-improvement loops live or die on problem selection: too hard → no reward signal (all-fail
batches), too easy → no gradient. That *is* a curriculum problem, and the stack here is
purpose-built: the verifier gives ground-truth reward; premise-surprisal estimates difficulty
relative to the current model; **CircuitRank's known→goal voltage formulation is literally "find
the next provable rung."** The loop: fail a hard target → mine its prerequisite cone for the
nearest-provable rung (cumulative-difficulty spacing) → prove → add to the known set → re-solve the
voltage problem → climb. The staged-decomposition eval arms, reframed as a data engine.

### 2. Curriculum-structured synthetic data / distillation
phi-style results showed textbook-*quality* data punches above its weight; this machinery generates
textbook-*structure*: prerequisite-ordered, centrality-weighted. Concretely: distillation sets where
the teacher solves items in dependency order (student never trains on an item whose prerequisites it
hasn't seen) and **sampling mass follows the cover-80 core** — train deeply on the ~27 lemmas that
*are* matroid theory, lightly on the tail. The few-shot finding sharpens the unit: for weak
consumers the right curriculum element is the **worked pair (statement, proof)**, not the statement
list. The book pipeline is the prose twin of this application.

### 3. Inference-time orientation (cheap, shippable, capability-gated)
The cover-80 set is an **optimal cheat-sheet constructor**: "this domain's API in K items" for agent
system prompts; repo maps (Aider's PageRank repo-map is production prior art); RAG reranking by
load-bearingness rather than similarity. Carries finding (1) as a hard caveat: gate on consumer
capability, do not deploy blanket context.

### 4. Adaptive, contamination-resistant evaluation
A structured item bank with graph-grounded difficulty enables IRT-style adaptive testing (place a
model on the ladder in ~20 items, not 2,000) and fresh-item generation along the frontier for
contamination resistance. Low glamour; every lab needs it; the difficulty oracle is the hard part
and falls out of this work. The matrix50 anchors/predicted-similar/fresh blocks are this
application in miniature.

### 5. Pretraining data **selection** (the bg-norm idea) — and, weakly, ordering
Evidence says large-scale pretraining is fairly order-insensitive; the live version is *selection*
for mid-training/domain adaptation: **bg-norm over a citation/usage graph carves "the N documents
that ARE this field"** — `p^a · log(p/g)` with `p` = seeded centrality and `g` = uniform-teleport
background demotes the hubs every raw-centrality method drowns in (DOI pages, Eq/congrArg, generic
survey papers) and surfaces the *surprisingly central* core. This is the same Fightin'-Words move
the repo already validated twice (Wikipedia topics, mathlib lemmas), pointed at corpus curation.
Ordering remains the weak bet at scale; keep it for curriculum-RL and domain adaptation only.

### 6. Human-facing curricula (the original application)
Textbook ToCs, reading lists, onboarding paths. Banked honesty: against real book ToCs the
generated curricula currently tie at best — so position as **prior/grounding for a generative
ToC phase** (feed the toc-tournament a centrality-ranked concept list), not as replacement.
Cross-prediction from finding (1): priming a *weak* ToC-generator with curriculum context may hurt;
test before adopting.

### 7. Agent skill libraries
Voyager-style skill acquisition ordered by the dependency graph: which skills to consolidate first,
which to attempt next (= application 1 with "skill" for "lemma"). Speculative; listed for
completeness.

### Unglamorous but real
- **Doc onboarding order**: introduce-before-use ordering over an internal wiki (see bipartite
  section below).
- **Context packing for RAG**: cover-X greedy selection over a retrieved set = maximal coverage per
  token — the cover-80 curve as a packing algorithm.
- **Dedup/diversity measurement**: working-set perplexity (effective number of distinct items) as a
  corpus-diversity dashboard metric; the size-controlled z-score makes it comparable across corpora.
- **Review/maintenance prioritization**: in a codebase, surprisingly-central functions (bg-norm over
  the call graph) are where review and test effort should concentrate.

## The bipartite sentence–word construction (raw text corpora)

Setup: incidence matrix `B[s, w]` over sentences × words (tf or binary; sentences could be
paragraphs/chunks). The **stochastic Gramian** is the two-step walk: row-normalize each hop,
`P = RowNorm(B) · RowNorm(Bᵀ)` — sentence → word → sentence. Stationary/PageRank mass on `P` is
sentence centrality; the transpose walk gives word centrality.

**Prior art to be honest about:** PageRank on a sentence-similarity graph is **LexRank/TextRank**
(2004) — classic cosine-over-tf-idf similarity is a normalized Gramian. `BᵀB`'s spectrum is **LSA**.
And the repo's hub-suppression `1/log(indeg)` applied to edges *into* common words **is IDF** at the
edge level. So the construction is well-trodden — which is good news: the *walk* is commodity, and
everything novel this repo adds sits on top:

1. **bg-norm selection on sentences** — LexRank's known failure is generic-central sentences
   (connectives, boilerplate, thesis restatements) dominating. `p^0.5 · log(p/g)` with `g` = the
   uniform-teleport walk on the same Gramian (or on a global/background corpus) extracts
   *surprisingly central* sentences: the domain core, not the generic summary. Nobody runs
   Fightin'-Words-corrected LexRank; this is the immediately publishable/usable delta.
2. **Two-direction bridging** — seed forward from known/query sentences, reverse from target
   sentences, combine by product: the *bridging* passages between what the reader knows and what
   they're trying to reach. (The bidirectional-PR machinery verbatim, on the Gramian walk.)
3. **Cover-curriculum ordering** — the Gramian is symmetric, so it has **no intrinsic prerequisite
   direction** (the key limitation vs. the proof-term graph). Inject direction cheaply:
   **introduce-before-use** — order sentences so each new sentence minimizes newly-introduced
   vocabulary mass (greedy cover over the word side = the cover-80 machinery as a *sorting*
   criterion). Reading order that never uses a term before the corpus has introduced it; a
   vocabulary-prerequisite curriculum with zero NLP beyond tokenization.
4. **Word-side unification** — word centrality under the transpose walk + Fightin'-Words lift =
   the existing keyword layer, derived from the same object. One bipartite graph yields keywords
   (word marginals), topics (sentence clusters / LSA-or-Laplacian spectrum — the existing
   spectral-from-graph layer), curriculum (ordering), and summary-core (bg-norm selection). The
   GCCA multi-view default already treats keyword-LSA as a view; this says why: same Gramian.

**Numerics:** never materialize `B Bᵀ` (the repo's never-densify discipline): power-iterate
`v ← RowNorm(B)(RowNorm(Bᵀ) v)` on CSR. Stationary mass of the raw two-step walk is degree-biased
(≈ sentence length) — exactly why bg-norm (rank level) and/or IDF damping (edge level) must
compose with it; both exist in `curriculum/`.

**Applications instanced to text:** continued-pretraining corpus distillation (5); book/long-doc
"distinctive summary" (bg-norm vs the book's own background = chapter cores; vs corpus background =
what makes this book different); onboarding-doc reading order (introduce-before-use); RAG context
packing (cover-X). Evaluation can reuse the intrinsic quality basket + the de-randomized pairwise
judge already in `curriculum/layers/`.

**Open questions:** chunk granularity (sentence vs paragraph — degree bias worsens with variance in
length); whether introduce-before-use ordering beats document order on real onboarding corpora
(testable with the judge battery).

**First empirical verdict (2026-06-11, see `docs/handoffs/2026-06-11-lexrank-bgnorm.md`): bg-norm
LexRank is REFUTED for generic summarization** — −3.6pp R1 vs plain LexRank (n=246 wiki articles,
CI excludes 0), failure locked at the objective level by a post-hoc control. Mechanism: a summary
is the *generic core* of a document; bg-norm extracts the *distinctive periphery* by construction
(qualitative: it picks the silica-sand-jetty minutiae over the lead-like overview). The directions
of the likelihood-ratio family are not interchangeable: **p/g = distinctive (lift), g/p = gaps,
plain p / cover-mass = representative (summaries)** — summarization belongs to the cover family.
Two positive byproducts: tf-idf *degree* (+4.5pp over LexRank) is the validated cheap summarizer —
i.e., the edge-level IDF ≡ hub-suppression correspondence is what summarization actually needs
(the raw walk's true pathology was length bias, ρ=0.93); and bg-norm-LexRank remains the right
tool for the *distinctive-summary* applications (what-makes-this-book-different, distillation
dedup), which should be evaluated against distinctiveness references, not leads.

## Gap detection, revisited (the homecoming application)

*The repo's original problem, restated with the matured toolkit (2026-06-11 session). Setting:
foreground cloud F (the field's works) and background cloud B (the broader corpus), in any subset
of {content embeddings, keyword space, citation graph}.*

**The unification: a gap is bg-norm reflected.** bg-norm scores *surprisingly occupied* regions
(`p^a · log(p/g)`); a gap is *surprisingly empty*:

    gap(x) = g(x)^a · z( log g(x)/p(x) )

with `g` = **background density (habitability — where works CAN live)** and `p` = **foreground
density (occupancy)**. The bg term kills the everything-outside-the-cloud pathology; the
Fightin'-Words variance normalization `z = Δ/√(1/n_F,loc + 1/n_B,loc)` kills sparse-sampling
mirages (emptiness must be statistically earned). One likelihood-ratio family, three directions,
each its own task — empirically priced by the LexRank dead-end above:
**p/g = distinctive · g/p = gaps · cover-mass = representative.**

**Spherical gaps: essentially one canonical answer.** The largest empty ball's center sits at a
foreground **Voronoi vertex** (largest-empty-circle theorem); the high-dimensional estimator is to
use background points as candidate centers with the **kNN density-ratio score**

    score(x) = r_F,k(x) / r_B,k(x),   x ∈ B

(distance to k-th fg neighbor over k-th bg neighbor; its log is the nonparametric density log-ratio
— continuous Fightin' Words, `k` as the Dirichlet prior). Argmax over B = the spherical gap; ~30
lines on a kNN index.

**General gaps: ill-posed until you pick the shape regularizer.** Three natural ones, finding
different gap species:
1. **Level-set gaps** (recommended): gap = connected component of `{p_F ≤ ε} ∩ {p_B ≥ δ}` — mined
   on the background sample: keep bg points whose spherical score clears threshold, build their
   neighbor graph, take components. **Size = n_eff of the component's bg mass** (perplexity
   machinery), never volume — "effective number of distinct background works living where the
   foreground has none." Facility location then gets its proper job (cf. the FL/DPP anchor-selection
   dead end): **k-center on the gap component picks the representative missing works** — the
   proposal generator.
2. **Bridging gaps** = Burt's structural holes: two fg clusters with high bg-mediated but low
   fg-mediated connectivity. **CircuitRank computes this directly**: clamp the clusters at 0V/1V,
   solve on the background graph; current density concentrates in the fg-empty corridor. Interior
   holes (1) and bridging holes (2) are different topologies; level sets cannot see the second.
3. **Lattice gaps** (FCA / most interpretable): an **unrealized concept** — a closed keyword set
   with substantial background extent and empty (or FW-surprisingly small) foreground extent:
   "works combining {X, Y, Z} exist at large but not here." This is contrast-set / emerging-pattern
   mining (Dong–Li) between the two formal contexts, and it gives every geometric gap a *name* —
   fixing embedding-gap detection's chronic interpretability problem.

**Multi-view discipline:** detect geometrically (1–2, ideally in the GCCA joint space), describe
lattice-wise (3), and demand cross-view agreement — a gap visible in embeddings with no
keyword/citation signature is probably an embedding artifact.

**The fully-general definition** (for completeness): a gap = a long-persistence void (H₁/H₂ class)
of the foreground cloud whose interior carries background mass — persistent homology gives a
parameter-free size; the level-set construction is its computable shadow.

**Empirical verdicts (2026-06-11, plant-and-recover on 30k wiki docs — see
`docs/handoffs/2026-06-11-gap-plant-recover.md`):**
- **Species↔detector matching CONFIRMED** — the taxonomy is real: concept gaps are geometrically
  invisible (0 orphaned supporters) yet lattice-mined at top 0.4% (16/16); corridors invisible to
  the ratio score yet found by voltage bridging at median rank **12**. No single detector covers
  the species.
- **Habitability matters decisively — but as the candidate UNIVERSE, not a soft weight.** Unscoped,
  off-cloud probes take 50/50 of the top-50 (the classic pathology, demonstrated); field-scoping
  improves rank ~23×. But the soft `g^a` weight *hurts* within scope (distance-only: rank 10 vs
  full statistic 386). Corrected form: `gap(x) = 1[x ∈ habitable] · emptiness(x)` — hard gate, not
  soft weight.
- **The z-form was refuted on measurement** — `Δ/√(1/n_F+1/n_B)` inverts rankings (the 1/n_F term
  explodes exactly on the signal). The working form is an LCB (`Δ − 0.5σ`) at the adaptive max-ball
  scale; mechanism real, field effect marginal (+0.04 AUC).
- **Level-set components are a dead end in high-d** (recall@3 = 0.00 across all species): removing
  a ball's worth of fg barely dents neighbors' density when ball mass lives at the boundary
  (median orphan ratio 1.10). Use the ratio score + scoping directly; skip the component machinery.
- Legacy k-ball detector: hit-rate@5 0.31 — beaten by the scoped distance-only score (rank ~10).
- **Fusion form VALIDATED in-scope (follow-up A):** rank by r_F, statistic as binary veto — rank 13
  / AUC 0.83 with probes eligible and no oracle; but the veto can NOT replace the scope quantile
  (best unscoped 8,708 ≈ full statistic's 8,860 vs scoped 13) — universe stays load-bearing.
- **Empirical-null calibration NOT worth it for ordering (follow-up B):** in-sample +0.008 AUC over
  the LCB inverts on split-half (0.829 vs 0.842); keep `Δ − 0.5σ`. Its Tukey fence does self-tune
  A's veto ceiling to ~log 2 (matches the best fixed gate unscoped).
- **Walk/bipartite-Gramian geometry does NOT rescue concept gaps (follow-up C):** AUC 0.47→0.49
  (chance) vs matched controls, 0 orphans; radial collapses there (0.84→0.51). Miner monopoly
  stands; mechanism only fires for rare-shared-keyword concepts (unit fixture: walk 1.0, raw 0.52).
- **Miner does NOT lead a composite (P1):** radial signatures exist (15/16 wiped patterns, doc AUC 0.67) but rank 317 vs fused 6.5; corridor conjunctions too thin (4/16 found, rank ~1.8k vs voltage 12) — real fusion required.
- **Level-set residualization works as a CHANGE detector (P2):** recall@3 0.00→1.00 radial / 0.75 corridor at change-scaled selection (top-frac 0.01); standing pockets cancel; concept stays 0 — Track 2 (T vs T+Δ) machinery validated, still no single-snapshot Track 1 use.
- **All-pairs voltage CHANGE field is species-agnostic (P3a):** |flow-share Δ| over 66 cluster pairs ranks radial 1.5 / corridor 3.5 / concept 20.5 (AUC 0.79/0.82/0.64), beating the single-pair specialist without the pair oracle; after-only/static at chance — Track 2 form only.
- **Commute/resistance geometry refuted (P3b):** concept at chance (0.47–0.49) in bipartite and LSA-graph commute (flat bipartite spectrum = degeneracy regime); radial rank 10→291 — no place in the composite.
- **Low-df tail is a null (P4):** planted rank 128.5→128.5 under cap-10k and full-window tails, tail attribution share 0.00; walk flip refuted (0.45–0.48) — the miner is bounded by natural emerging patterns, not the keyword budget.
- **Type-blindness priced (Track 1 finish, Part A):** noisy-OR percentile fusion of geo/pat/volt with self-selected voltage pairs costs +6 ranks radial (6.5→12.5), +55 corridor (12→67; pair-blindness 12→29.5), ~0 concept (doc-space home already near-chance); overall blind found@50 0.56 over 48 mixed trials.
- **Minimum recoverable gap = ~64 docs (~4% of the field) at k=5–10 (Track 1 finish, Part B):** below 64 the orphan-core footprint collapses (8-doc: none anywhere; 16/32-doc: cores at 4–10/16 banked locations, ranks ~1.5–4.4k); k=20 doubles the threshold; corridor voltage same 64-doc edge.
- **Track 2 wiki temporal holdout (first REAL-gap test):** at T=2003-01/Δ=2y (store is the 400k oldest ids, 2001–2004; T slid back on a counts-only gate), only absolute LEVEL-SET components out-fill growth/density-matched nulls (1.28, clustered CI [1.06, 1.48]); fused 1.10 and voltage 1.80 honest negatives, miner inconclusive (top-50 patterns = 4 register-pair neighborhoods) — planted-gap rank does NOT transfer to arrival prediction; matching machinery must be null-calibrated (v1 disjointness rule biased nulls to 1.27) and bootstraps clustered.
- **Reroute ≠ arrival — two orthogonal fields (task_oa_dissoc, OA NLP):** citation FLOW-CHANGE measures what the field newly ROUTES THROUGH (canonical infrastructure: 54% of top hubs are shared-task/tagset/treebank-style reference papers, old+cited+empty) and ARRIVAL-DENSITY measures where new work LANDS (level-sets, the 1.92 predictor); Spearman ≈ 0 residualized, top-50 overlap 0, no lead–lag (W1→W2 1.08 [0.81, 1.33]) — for curricula: flow-change picks the backbone/reading-list spine, arrival-density picks the live frontier, and high-both = revolution fronts (attention-NMT 2015–19).

**Cheapest validation:** plant synthetic holes (delete a ball / a concept's extent / a bridge) in
the wiki or OpenAlex clouds and measure recovery precision of formulations 1–3 — same
plant-and-recover design as the matrix50 anchors, no GPU required.

---
*Provenance: written 2026-06-10 during the matrix50 model×mode sweep; brainstorm captured from the
working session. Regenerate: this is a thinking doc — edit by hand.*
