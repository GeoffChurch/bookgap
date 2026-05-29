# Curriculum ranking: keep PageRank, fix the input (+ separate the ordering role)

Curriculum relevance ranking uses (bidirectional) personalized PageRank. **Keep
it** — sparse power iteration over 40M nodes in ~100s is unmatched at this scale,
and "teleport to the gap, flow along citations to prerequisites" is sound. The
defects seen in the first 5 curricula (`subjective-curriculum-eval.md`) are
**input-side** (popularity bias + semantics-blind edges), not PR's math, so the
fixes modify the *input*, not the algorithm.

## Observed defects (motivation)

- **Off-topic intrusion**: an unrelated, heavily-cited "PCM distortion analysis"
  (signal-processing) cluster of 693 works intruded into the Intelligent-Tutoring
  curriculum (T11902). Forward personalized PR leaks α-mass to globally-popular
  hubs in the seeds' citation neighborhood.
- **Diffuse generic foundations**: broad foundation clusters with mushy keywords
  (`theory, first, system`); citation edges treat a tangential-but-popular citation
  the same as an on-topic one.

## Levers (keep PR; modify the input), in suggested try-order

1. **Source seeds → bidirectional `product` corridor.** The first runs were
   forward-only (`source_work_ids` empty). Seeding sources with the field's
   established foundations makes `two_direction_pagerank(combine="product")` keep
   only works on the **foundations↔gap corridor**; an off-corridor cluster (PCM)
   gets ≈0 reverse mass → product ≈0 → filtered. Natural automatic source set:
   the gap's **covering books** (established works for the topic, in the context
   JSON's `covering_books`). *(Being tested first — see Status.)* Open question:
   books may have sparse OpenAlex citation connectivity → weak reverse signal;
   fallback source sets = top-cited works within the gap's field / OpenAlex
   ancestor-topic canon.
2. **Semantic edge weighting.** `SimilarityGraphConfig` (`similarity.enabled/
   knn_k/mix_weight`) blends citations with an embedding-kNN graph so mass flows
   along semantically-coherent citations. Already plumbed but **off by default**.
   ⚠ Cost: `build_similarity_adjacency_from_cloud` builds kNN over the whole
   cloud (~40M works) — assess runtime/memory before treating as "cheap".
3. **Background / specificity normalization.** Rank by personalized-PR ÷ global-PR
   (topic-sensitive PR's trick) to demote works popular *everywhere*; shrinks the
   bloated generic foundation clusters. Cheap post-transform of the relevance
   vector; small code change in the ranking step.
4. **Weighted seeds.** Weight `target_work_ids` by P(in true gap) instead of
   uniform, so the walk starts from the gap core. Same weights as the backlogged
   `kde_ratio` soft-membership idea in `subjective-curriculum-eval.md` (Recipe C →
   Future refinement).

## Separate concern: relevance ≠ ordering

PR gives a **relevance scalar** — right for *selection* (which 5,000 works) and
clustering mass-weights. But a curriculum also needs an **ordering** (prerequisite
partial order), and PR magnitude ≠ prerequisite depth/direction. The pipeline
currently orders via frontier-adjacency absorbing-PR (a fine proxy), but
"X precedes Y" is a DAG/flow property. Consider a **directional** method for the
*ordering step only*: **CircuitRank** (already a `RankMethod` option; effective
resistance / current flow is inherently source→target directional) or an explicit
citation-DAG topological / longest-path order. Keep PR for relevance.

The **pedagogical-subspace** idea (`curriculum-generalization-pedagogical-subspace.md`)
compounds with #2 and clustering — it changes the embeddings both consume.

## Canonical defaults (derived)

The score is one family `p^a · log(p/g)`; given `α_fg`, the other two knobs are pinned:

- **`a = ½`** — rank by elevation *per unit of graph-noise*. PR is deterministic, so
  the "variance" is over the citation graph as a noisy sample of latent dependencies:
  Var(log p) ∝ 1/(p·M) ⇒ z ∝ √p·log(p/g) ⇒ a=½ (the Fightin'-Words variance-norm).
  a=0 = raw log-lift (tail-noisy, wants a Laplace ε); a=1 = KL / Bayesian-surprise.
- **`α_bg = α_fg`** — read `log(p/g)` as a likelihood-ratio test of the gap-restart vs
  uniform-restart model; a clean LRT needs the *same kernel*, differing only in teleport.
- **`α_fg`** is the lone free knob = curriculum depth (data-derivable via a
  community-scale knee that also fixes K; else PR-standard 0.85). α_fg and α_bg are
  independent — `p` depends only on α_fg, `g` only on α_bg, and no ratio/combination
  collapses them (changing α reshapes, doesn't rescale).

## Status / working notes

- [x] **Experiment 1 — source-from-books on T11902** (2026-05-29). All 29 covering
  books resolved as sources; bidirectional `product` (208 targets × 29 sources).
  **Result: mixed — mechanism validated, but `product` is too aggressive.**
  - ✅ **Off-topic intrusion gone.** Signal/speech clusters 2 → 0; every topic now
    ITS/knowledge-tracing-relevant (`knowledge_tracing`, `student_modeling`,
    `cognitive_diagnosis`, `Q-matrix`, `DKVMN`, intelligent_tutoring…). The PCM /
    signal-processing cluster (sz 693 in baseline) vanished entirely. Confirms the
    intrusion was a popularity/input artifact fixable by input shaping.
  - ❌ **Clustering degenerated.** 8 of 15 clusters are **singletons** — several are
    the seed nodes themselves (sources at `simS=1.0`: *Intelligent Tutoring
    Systems*, *Mind Bugs*; targets at `simT=1.0`) — plus one **937-work mega-cluster**.
    The `product` of two PR distributions is too peaked: seed nodes dominate the
    top-k and the rest collapses into one residual blob. Not usable as a curriculum
    structure as-is.
- [x] **Experiment 2 — background normalization on T11902, no sources** (2026-05-29).
  Canonical default `a=0.5`, `α_bg=α_fg=0.85` (`seed_from_gap --background-normalize`).
  **Result: clean win — implemented and validated.**
  - ✅ **Off-topic intrusion gone** — signal/speech clusters 2 → 0; the PCM sz-693
    cluster vanished.
  - ✅ **Clusters stayed balanced** — 1 singleton (= baseline), max 780 vs 817, no
    collapse (vs the corridor's 8 singletons) — confirms the no-multiplication theory.
  - ✅ **Better foundations surfaced** — *Statistical Theories of Mental Test Scores*
    (IRT/psychometrics), *Foundations of Intelligent Tutoring*, *Representing Student
    Performance*, PDP, Bayes-net methods replaced the generic *AI:AMA* + PCM. Demoting
    universally-popular mass let the genuinely-prerequisite roots rise. simT ascent
    clean (0.84 → 0.96). Output `gap_curriculum_T11902_bgnorm`.
- [x] **Regression — background normalization on T11273** (2026-05-29, a known-good
  topic). **Result: nuanced — a lateral trade, not a strict win.** Stayed on-topic
  (0 off-topic) and balanced (no singletons, max 543). Surfaced *more* network-
  embedding-specific works (Katz's *Status Index*, SDNE, *Watch Your Step*, *Learning
  From Networks*). **But demoted semi-universal foundations** — notably *Laplacian
  Eigenmaps* and *PageRank* — which are gap-relevant yet globally popular, so their
  `p/g` wasn't high enough to survive. Confirms the predicted universal-foundation
  casualty: bgnorm trades universal foundations for gap-specificity.
  **Conclusion: keep `background_normalize` opt-in (NOT default-on).** Clean win only
  where there's a real off-topic intrusion (T11902); on already-good topics it's
  lateral. `α_bg` is the aggressiveness dial — a gentler `α_bg` (e.g. 0.5) might retain
  foundations like Laplacian Eigenmaps while still removing egregious hubs (open).
- Deferred (source path only): `combine="linear"` additive blend + excluding seed
  nodes from clustering — relevant only if source seeds are reintroduced (currently
  out of scope; source-picking left to downstream users). `combine` is hardcoded
  `"product"` at `build_curriculum.py:791` if revisited.

## Pointers

- Ranking: `artifacts_scripts/curriculum/pagerank.py` (`two_direction_pagerank`,
  `compute_pagerank`), `build_curriculum.py` (`:746-795` seed construction,
  `:698-713` similarity graph, RankMethod handling).
- Driver: `artifacts_scripts/curriculum/seed_from_gap.py` (`--source-from-books`).
- Baseline curricula: `subjective-curriculum-eval.md` (Results 2026-05-29).
