# Fuzzy / soft topic membership (retire hard assignment) — the right design

**Motivation.** The pipeline is *soft on relevance, hard on assignment*: PageRank gives a continuous
mass per work, but each work is hard-assigned to one topic by **embedding-only** clustering
(agglomerative/bisecting k-means → dendrogram cut → hard labels; citations don't even enter the
assignment). Masses only weight contributions *within* a topic. The integer "works per topic" is an
artifact of that argmax — `summary.md` already reports a fractional **"Effective # of Works"**, so
the honest count was never an integer.

This is the highest-leverage fix for the residual specificity debris: **both** panel books'
remaining debris reduce to the *degenerate-cluster* pathology (a single dominant work *being* a
chapter — Neuromorphic's padding tail, and Color Image's catalog-blurb book; see
[[curriculum-specificity-semantic-reweight]] and [[curriculum-filter-frontmatter-noncontent-works]]).
Soft membership dissolves that at the root.

## The principle

A topic is a mode/region in the joint citation+embedding space. A work's membership is a
**responsibility** `r_{i,k} = P(topic k | work i)` — a proper distribution per work
(`Σ_k r_{i,k} ≤ 1`, the slack going to a "background"/none-of-the-above sink). The weight that flows
downstream is **`w_{i,k} = mass_i · r_{i,k}`** (importance × membership). This completes the weighting
philosophy: the integer count becomes a conserved soft count `n_k = Σ_i mass_i·r_{i,k}` (total mass
split across topics + background, no double-counting), and the honest chapter size is its ESS
`(Σ_i mass_i·r_{i,k})² / Σ_i (mass_i·r_{i,k})²` — the per-topic generalization of the
"Effective # of Works" already in `summary.md`.

## The right `r_{i,k}` for THIS pipeline: α-damped absorbing-walk (harmonic) responsibilities

Because the project is built on diffusion over the citation(+similarity) graph, membership should
come from the *same* walk, not a bolted-on GMM:

1. **Anchors.** Each topic `k` gets a high-confidence **core set** `S_k` — bootstrap from the
   existing dendrogram cut (the top-mass works of each cut group). A small core set (not a single
   node) is robust.
2. **Absorbing walk.** From each work `i`, run an **α-damped absorbing personalized PageRank** on the
   (row-normalized) citation+similarity graph: follow edges with prob α, get **absorbed** on hitting
   any anchor, **teleport to a background sink** with prob (1−α). Then `r_{i,k}` = probability of
   absorption into `S_k` before teleporting; the leaked mass is `r_{i,background}`.
3. **Solve** by power iteration like PageRank (no matrix inverse): with transient states Q and
   absorbers A=∪S_k, iterate the absorption matrix `B ← P_QA + P_QQ·B`; aggregate columns by topic.
   Cheap over the ~5k selected works × N topics.

Why this is *right*, not just convenient:
- **Uses both signals** — on the mixed citation+similarity graph, membership reflects citation
  structure *and* embedding proximity (today's k-means is embedding-only).
- **Proper distribution, no temperature knob.** Softmax-over-PPR needs a temperature; the absorbing
  formulation self-normalizes (mutually-exclusive absorption). The **background sink is free** — it's
  just PageRank's existing `(1−α)` leak — so works far from every anchor get low membership
  everywhere and high background instead of being force-assigned. That "none of the above" outlet is
  exactly what stops junk from anchoring a chapter.
- **Reduces to today under `argmax_k r_{i,k}`** → strict generalization, directly A/B-able vs the
  hard pipeline.
- **Reuses the absorbing primitive already in `pagerank.py`** (per-node `alpha=0`, used today only
  for the topic×topic ordering matrix in `compute_frontier_adjacency`). The only new thing is keeping
  the per-work *distribution over topics* instead of aggregating it away.

## Downstream (everything inherits `w_{i,k} = mass_i·r_{i,k}`)

- **Keywords:** work `i` contributes tokens to topic `k` with weight `w_{i,k}` (to several topics if
  it bridges). `count_topic_tokens` already takes per-doc weights — feed these and let a work appear
  in multiple pools. The **contrastive log-odds handles bridging gracefully**: a token shared by two
  topics a work belongs to isn't *distinctive* to either, so it won't dominate either label.
- **Centroids / sim-to-target / ordering:** topic centroid = `w`-weighted mean; chapter ordering
  stays topic-level (anchors via `compute_frontier_adjacency`), unchanged in mechanism.
- **Reading artifact:** discreteness re-enters *only* here — list a work under `argmax_k r_{i,k}` for
  the clean path, and optionally under any topic where `r_{i,k}` exceeds a threshold ("also relevant
  to ch. 7").

## What it fixes

- **Degenerate clusters can't form.** A would-be 1-work chapter instead has its profile built from
  everyone weighted by responsibility → Lim's catalog-blurb tokens dilute among real image-processing
  works and vanish; Neuromorphic's padding never arises. Both books' debris reduce to this one fix.
  Subsumes B1 (min-cluster-size); makes dynamic-k mostly unnecessary (still useful as display honesty
  for genuinely thin topics).
- **N becomes forgiving.** Over-specify N and a spurious anchor collects little total responsibility
  (low ESS) rather than manufacturing a degenerate chapter — so "pick N exactly" softens; could merge
  high-overlap anchors or drop near-empty ones. (Eval still fixes N to the book's ToC.)
- **Faithful** to works that genuinely span topics.

## Honesty about knobs

Only real parameters: **N** (forgiving now), **α** (already a pipeline parameter — sets how much
un-topical mass leaks to background), and the **exemplar threshold** (presentation only). Crucially
**no temperature knob** — the reason the absorbing formulation beats softmax-PPR.

## Alternatives considered

- **softmax-over-PPR** from anchors — needs a temperature knob; absorbing avoids it. Worse.
- **GMM / soft k-means** on embeddings — embedding-only, ignores citations. Worse fit.
- **Archetypal analysis** (each work = convex combination of N extreme "archetype" works) — the one
  principled alternative; very interpretable soft memberships, but embedding-only and a heavier
  optimization. Loses to the absorbing walk on graph-coherence grounds.

## Prior art — this is largely a TRANSFER, not new research

The gap-detection side already built, tested, and **shipped** the multi-cluster soft-membership idea
([[multicluster-softmax-membership]], `artifacts_scripts/gap_softmax_enrich.py`, Phases 2–3). It's
mathematically the same model: per-point **softmax membership over nearest basin centroids,
per-member-normalized for size-fairness, one-shot**, optional ∅ background, weighted keyword
extraction (`DocWeighting.RELEVANCE` — what the curriculum keyword pipeline already uses). Curriculum
topics map onto "basins"; the N chapters compete in the softmax; ∅ = "belongs to no chapter"; drop
the book-anchored-competitor machinery (no books inside a single gap). **Adapt this rather than
build from scratch.**

Two design choices are now **empirically settled**, not assumptions:
- **One-shot, never iterate.** Every iterated multi-cluster config diverged (`final_delta 0.75–0.99`,
  the diffusion-to-uniform fixed point); they shipped one-shot only.
- **Anti-diffusion regularizer is mandatory.** The naive `K_G/K_total` iteration is power iteration
  on `D⁻¹K` → uniform; no init saves it ([[full-data-soft-mixture-degenerate-bootstrap]]). The fix
  is per-member normalization (softmax framing) or the absorbing/fixed-anchor + background sink
  (this doc's framing). These are the SAME anti-diffusion trick.

So the **absorbing-walk variant here is the citation-aware cousin to A/B against the validated
embedding-softmax**, not the default. Theirs is embedding-kernel-only with a perplexity-temperature
knob; ours uses the citation+similarity graph (matters for curriculum, where the whole
product-of-experts leans on citation structure) and drops the temperature. Plan: **start from the
shipped one-shot softmax membership; treat the absorbing walk as the graph-aware comparison arm.**

## Topic-anchor selection: facility-location / DPP (kills redundant chapters)

Both the softmax model and the absorbing walk need **N anchors**. Today that's embedding
k-means/dendrogram cores. The deflation program left a **validated, no-knob facility-location** impl
(`fl_bookdemand_resid`, ported to `bookgap/llm/gap_algorithms/deflation_strategies.py`) and an
**untested DPP** (`_dpp` in `artifacts_scripts/gap_deflation_bakeoff.py`) — both coverage/diversity
selectors. This attacks a defect nothing else here addresses: **Color Image's ~8 near-duplicate
histogram-equalization chapters** (redundant topics). FL (maximize coverage of the gap's works) or
DPP (high-quality + mutually-dissimilar anchors) directly de-duplicate the anchor set. FL is
battle-tested + intrinsic; **DPP is genuinely open (no recorded result)** and is the textbook
anti-redundancy tool. Worth trying FL/DPP as the anchor selector under the fuzzy model.

## What from the n_eff / gap-sizing program is NOT worth porting

Gap-*sizing* (n_eff ranking, k-ball cohesion / hill-climb / moving-centroid radius reopt,
dual-landmark) is gap-detection-side and already characterized ("raw KBALL ≥ n_eff except on
article-dense topics"; dual-landmark validated only as a diagnostic). Their useful *concept* (ESS /
participation ratio) is already in this design (soft count `Σ mass·r`, ESS chapter size) and could
even replace dynamic-k's relative-floor. The metrics themselves don't transfer.

## Increment 1 (prototype) — mechanism validated (2026-06-02)

`artifacts_scripts/curriculum/gap_fuzzy_topics.py` (non-invasive): reads a built run-dir, computes
one-shot softmax responsibilities over the topic centroids (τ by target perplexity 1.5), reports
hard vs soft (mass-weighted) sizes + ESS, and the most-degenerate hard topic's top soft members.
Probes saved as `fuzzy_topics_probe.md` in the example folders.

- **Neuromorphic** (degenerate ch with hard size **1** = the lone VCSEL paper): soft **ESS 3.2**, top
  soft members are all real VCSEL/spiking-neuron papers (`Spin-Polarized Quantum-Dot VCSEL`,
  `Controlled Propagation of Spiking Dynamics in VCSELs`, …). **Fuzzy dissolves the degeneracy** —
  the soft assignment re-attracts the semantically-nearest works the hard *citation* clustering had
  isolated. Keyword extraction over that pool would yield clean VCSEL keywords, not the boilerplate
  tail. ✓
- **Color Image** (hard size 1 = Jae Lim's book): soft ESS 4.1 but top members are still Lim +
  *more boilerplate* (`Copyright Page - Volume 1`). Its centroid is the **blurb embedding**, which
  sits in a boilerplate neighborhood, so soft re-attraction pulls in boilerplate, not content.
  **Confirms the two-mechanism split:** fuzzy fixes the degenerate-cluster mechanism but NOT the
  bad-embedding one — that needs abstract hygiene ([[curriculum-filter-frontmatter-noncontent-works]]),
  which is upstream and orthogonal. ✗ (correctly).

**Increment-2 (next):** thread `mass × r` into keyword extraction (let a work contribute to every
topic above a responsibility floor; reuse `count_topic_tokens`'s weight support) and re-judge the
panel — does Neuromorphic's specificity move from tie toward a win once the debris chapter carries
clean VCSEL keywords? Then ordering/exemplars, then the absorbing-walk (citation-aware) comparison
arm, then FL/DPP anchors. Prototype currently uses centroid-kernel softmax (simplification of the
per-member-normalized form) and bootstraps centroids from the hard `topic_work_ids`.

## Caveats / sequencing

- Bigger change than dynamic-k — thread `r_{i,k}` through clustering → keywords → summary → exemplar
  selection. But each stage already takes weights, so it's mostly plumbing + the one new absorbing
  solve.
- Does NOT fix the catalog-blurb *embedding* pollution upstream of gap detection — that's abstract
  hygiene in [[curriculum-filter-frontmatter-noncontent-works]] (orthogonal, complementary).
- **Sequencing:** cheap abstract-hygiene / paratext pass first if convenient, then this. Eval via the
  ToC-reconstruction panel — watch specificity (Color Image is the remaining loss) and confirm
  degenerate chapters vanish. Validate the generalization with `argmax(r) ≈ current hard labels`
  before trusting the soft outputs.
