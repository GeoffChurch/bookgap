# Curriculum: smooth diffusion densification + objective quality metrics

Two coupled work items spun out of the joint-graph v2 result
(`docs/wikipedia-openalex-joint-curriculum.md`): replace the **hard** 1-hop OA expansion with
**smooth** graph diffusion (with a principled diffusion-time T), and build **objective proxies**
for curriculum quality so we stop eyeballing. **Living doc** — grows as items are tackled.

Context: joint-graph v2 fixed OA mass-starvation with (#1) hard 1-hop OA citation-neighborhood
expansion + (#3) per-corpus-select. Ablation: synergistic, both needed (247→378→644 OA papers
into the curriculum). #1 is a crude **truncated (k=1, unweighted) diffusion** — the smooth
generalization should do better *and* let us choose the scale instead of fixing it at one hop.

## A. Smooth diffusion densification of the OA backbone (replaces hard #1) — ✅ DONE (validated)

**Status: validated and implemented** (`artifacts_scripts/wiki/joint_graph_curriculum.py`
`--oa-densify diffuse`; results in `docs/wikipedia-openalex-joint-curriculum.md` §v3). Smooth
GDC (heat kernel) at the held-out-link-chosen `t=8` with `--diffusion-top-m 32` **beats** the
hard 1-hop expansion: OA into the curriculum 644→874 (19.8%→27.0%), 770 wiki-anchored (v2 641),
pedagogy preserved (parsing + MT + language-modeling wins survive). All four T-criteria computed
and reported; three of four agree at `t ∈ [4,16]` (held-out-link and modularity exactly at t=8),
all with interior optima (over-smoothing guard fires). Key finding: `top_m` is the binding knob —
matching it to wiki's mean degree (~131) OVER-densifies and collapses coverage to 6.6%; the sweet
spot is `top_m ≈ 32`. The original design rationale is retained below for reference.

Replace the 1-hop expansion with **Graph Diffusion Convolution** (Gasteiger/Klicpera et al.,
*"Diffusion Improves Graph Learning"*): `S = Σ_k θ_k T^k` over the OA citation graph, with the
**heat kernel** (`θ_k = e^{-t} t^k/k!`, time `t`) or **PPR** (`θ_k = α(1-α)^k`, restart `α`), then
re-sparsify (top-k / ε-threshold) + renormalize. Smooth, weighted, multi-hop connectivity; no
arbitrary in/out cutoff. NB: PageRank **is** PPR diffusion, so this is graph-rewiring *before* the
curriculum's own PageRank. **Bonus:** choosing T to match OA's diffused connectivity to wiki's
could subsume *both* #1 and #3 — dissolve the 120× mass gap structurally. Cost: T is a scalar →
cheap 1-D sweep.

### Choosing the diffusion time T — all four promising; the criterion MUST penalize over-smoothing
(As `t→∞` diffusion collapses to the stationary/dominant eigenvector and loses topical resolution —
the GNN over-smoothing failure. A naive "more connectivity = better" criterion fails this.)
1. **Held-out citation-link reconstruction** *(primary; circularity-free).* Withhold a fraction of
   OA citation edges, diffuse over the rest, score AUC/AP at ranking them vs negatives, sweep T,
   take the max. Under-diffuse → low recall; over-diffuse → low precision → interior peak.
   Self-supervised, and **not** a curriculum metric — so it doesn't Goodhart the evaluation.
2. **OA↔wiki connectivity/mass matching.** Solve for the T where diffused-OA mass-holding (median
   PR mass / degree distribution) ≈ wiki's. Target-match; directly removes the heterogeneity that
   caused starvation (could replace the per-corpus-select hack).
3. **Modularity peak** of the OA/joint clustering vs T. Under → fragmented (low Q); over → one blob
   (low Q); interior max = best-resolved community structure. Curriculum-adjacent → use as a check,
   not the sole T-objective.
4. **Spectral-gap / diffusion-map** (Coifman–Lafon). Pick the diffusion scale with the cleanest
   eigengap of the diffusion operator. Principled, label-free.

Plan: select T by **(1)**, sanity-check vs **(2)**, report whether **(3)/(4)** agree; evaluate the
resulting curriculum with the §B basket (kept separate from T-selection to avoid circularity).

## B. Objective proxies for curriculum quality (stop eyeballing)

Caveat: curriculum quality is a human pedagogical judgment — use a **basket** to *detect failures*,
validate against *one extrinsic anchor*, and never optimize a single proxy (max-modularity
fragments; max-coverage takes over). Prefer intrinsic/no-knob metrics (the project's standing
metric-validity preference).

### Intrinsic (no external data — do FIRST)
- *Ordering:* **Sim→Target monotonic ascent** (already in the OA summary — should climb toward the
  goal); **prerequisite-completeness on the citation/link DAG** (fraction of each topic's
  graph-prerequisites that appear in *earlier*-ordered topics — no knobs); **citation/temporal
  directionality** (Kendall-τ of order vs median pub-year and vs in-degree — a *sanity check*: a
  *negative* τ = frontier-first = clear failure; we don't *maximize* recency).
- *Clustering:* **topic coherence** (NPMI / c_v of top keywords — reuse `compare_clusterings.py`),
  **cross-topic distinctness** (avg pairwise Jaccard / NPMI), **partition modularity** on the
  prereq graph, **cluster-size balance** (size entropy / Gini — catches the giant-cluster +
  singleton failures we hit).
- *Path:* **step smoothness** — consecutive-topic embedding/graph distance (a big jump = a missing
  prerequisite).

### Extrinsic (closest to ground truth — do LAST, more work)
- **Reference syllabus / textbook TOC** (NLP: Jurafsky & Martin) — concept precision/recall + order
  correlation.
- **Concept-prerequisite datasets** — **LectureBank** (NLP/ML concept-prereq labels) +
  prerequisite-chain-learning datasets; does the order respect labeled prereq pairs? (Confirm
  current availability.)
- **LLM-as-judge** — reuse the local ollama coverage-eval adapter; rate order / prereq pairs at
  scale (noisy; tiebreaker + eyeball-at-scale).

## Plan / status
- [x] **B-intrinsic** *(done — `docs/curriculum-quality-metrics.md`)* — built the reusable intrinsic
  quality basket `artifacts_scripts/curriculum/curriculum_quality.py` (pure `score_curriculum`,
  reusing `compare_clusterings` + `pagerank.compute_frontier_adjacency`) and validated it on
  wiki-native, joint v1/v2, and the OA-citation curriculum. **Discriminates well:** singletons /
  size-Gini / size-entropy, modularity Q, sim→target ascent (caught wiki-native's frontier-first
  ordering), keyword crowding. **Weak proxies (flagged):** NPMI-over-titles (too sparse, negative
  for good curricula), embedding keyword-coherence (catches garbage singletons but NOT the v1
  toolkit grab-bag — that's a pedagogical judgment no intrinsic metric here sees), Jaccard
  distinctness (uniformly high). **Surprise:** v2's win over v1 is *coverage* (644 vs 247 OA into
  the curriculum), NOT cluster-size balance (v1 is marginally more balanced). The toolkit-topic miss
  is the prime motivation for the extrinsic phase.
- [x] **A-diffusion** *(done — `docs/wikipedia-openalex-joint-curriculum.md` §v3)* — added
  `--oa-densify {expand,diffuse}` (GDC heat/PPR) + the four T-criteria (primary: held-out-link).
  **Yes, smooth diffusion beats the hard 1-hop** (OA 644→874, 19.8%→27.0%) at the held-out-link
  `t=8`, `--diffusion-top-m 32`; the criteria agree well (3/4 at `t ∈ [4,16]`, held-out-link +
  modularity exactly t=8), all with interior optima. Binding knob is `top_m` (wiki-degree ~131
  over-densifies → 6.6% collapse; sweet spot ~32).
- [ ] **Synthesis** — score the diffusion-densified curriculum on the §B basket
  (`artifacts_scripts/curriculum/curriculum_quality.py`) vs v2 (hard expansion). *(NOT yet run —
  separate task; the A-diffusion run delivered coverage + frontier spot-check only.)*
- [ ] **B-extrinsic (last)** — syllabus / LectureBank / LLM-judge validation.
