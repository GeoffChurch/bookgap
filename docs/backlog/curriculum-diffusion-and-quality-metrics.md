# Curriculum: smooth diffusion densification + objective quality metrics

Two coupled work items spun out of the joint-graph v2 result
(`docs/wikipedia-openalex-joint-curriculum.md`): replace the **hard** 1-hop OA expansion with
**smooth** graph diffusion (with a principled diffusion-time T), and build **objective proxies**
for curriculum quality so we stop eyeballing. **Living doc** — grows as items are tackled.

Context: joint-graph v2 fixed OA mass-starvation with (#1) hard 1-hop OA citation-neighborhood
expansion + (#3) per-corpus-select. Ablation: synergistic, both needed (247→378→644 OA papers
into the curriculum). #1 is a crude **truncated (k=1, unweighted) diffusion** — the smooth
generalization should do better *and* let us choose the scale instead of fixing it at one hop.

## A. Smooth diffusion densification of the OA backbone (replaces hard #1)

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
- [ ] **B-intrinsic** *(starting)* — build the reusable intrinsic quality basket; validate it
  discriminates the existing curricula (wiki-native, joint v1/v2, an OA-citation curriculum). This
  is the "quality dashboard" that replaces eyeballing.
- [ ] **A-diffusion** *(starting)* — add `--oa-densify {expand,diffuse}` (GDC heat/PPR) + the four
  T-criteria (primary: held-out-link); does smooth diffusion beat the hard 1-hop expansion, and do
  the criteria agree on a scale?
- [ ] **Synthesis** — score the diffusion-densified curriculum on the §B basket vs v2 (hard
  expansion).
- [ ] **B-extrinsic (last)** — syllabus / LectureBank / LLM-judge validation.
