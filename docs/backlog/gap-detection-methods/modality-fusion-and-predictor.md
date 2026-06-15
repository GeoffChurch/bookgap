# Modality fusion + an a-priori modality predictor

**Status: not started.** Two sub-questions left by the code triangulation
(`docs/code-gap-triangulation.md`).

## (a) Fusion — are graph and embedding strict substitutes, or complementary?
Lean's embedding detector was weak-but-nonzero (M_emb 0.62 alongside graph M1 0.93); Rust/Bazel
embedding ≈ 0.59. Add a **fused detector** to `core/plant_recover.py` (e.g. rank-sum or z-scored
M1 + M_emb) and re-run on the 5 cached datasets (Lean, Python, Go-cha/rta, Rust, Bazel). **Does fused
AUC ever exceed `max(M1, M_emb)`?** If never → modalities are strict substitutes (the dichotomy is
clean, a clean negative worth stating). If sometimes → residual cross-modality signal a combined
detector should exploit. **Cheapest possible:** the plant-recover already computes both score vectors
per trial; add one fused score + its AUC, re-run on cached data.

## (b) A-priori modality predictor
We pick the detector post-hoc. Is there a *cheap intrinsic* diagnostic that predicts a gap's modality
*before* plant-recover? **Coupling does NOT** (articles are the most-coupled domain, +0.215, yet
embedding-modality). Candidate predictors: graph-vs-embedding **neighbor agreement** (does a node's
graph-neighborhood overlap its embedding-neighborhood?), the spectral gap of the graph vs the
embedding kNN graph, or — most likely — a property of the gap *definition* itself (whether it
references graph structure or embedding density). **Difficulty: open / research-y.**

## Reuses / difficulty
`core/plant_recover.py` + the 5 cached datasets. **Difficulty: easiest of the follow-ons** — do the
fusion half first (a one-detector addition + re-runs); the predictor half is the open part.
