# Good-edge chain construction — push more ToCs into the clean-chain regime

The curriculum orders chapters by a **greedy good-edge** topical-adjacency walk
(`close_loop_metrics._good_edge_order`). Validated 2026-06-16: this recovers the human chapter sequence as an
**undirected** chain, and a generality anchor (eigen-centrality) then orients it **~89% correctly CONDITIONAL on
the chain being clean (|τ|≥0.25)** — but only ~62% unconditionally, because noisy chains have nothing to orient.
So **chain quality is now the binding constraint** on full directed-ToC replication — not the intro anchor
(solved: ch1 is the most-general chapter 76–79%, p≤0.008) and not the tree (average-linkage is a *placement*
win that actually **degrades** the ordering chain — see the placement≠ordering caveat in
`docs/curriculum-dendrogram-quality.md`).

**Idea — replace/augment the greedy walk with a better path construction** to raise |τ| and push more books
past 0.25:
- **Optimal-path (TSP-style) ordering** — maximize total adjacency (Σ consecutive cosine) over the chapter
  graph instead of greedy nearest-neighbor (myopic). Held–Karp for small N (chapters ≤ ~25), or 2-opt /
  Christofides for larger. Greedy good-edge is the baseline to beat.
- **Better adjacency metric for the ordering step** — ordering wants different geometry than placement, so try a
  denoised/whitened space or a graph-Laplacian / diffusion adjacency for the chain (not the raw centroid cosine).
- **Anchor-aware path** — jointly optimize the path AND the general→specific orientation, with the eigen-centrality
  generality as a soft endpoint constraint (intro at the front).

**Eval:** the n=25 ToC good-edge percentile (`ordering_genre_firmup.py`) + the 4 close-loop directed-τ
(`intro_orientation.py`). Success = more books at |τ|≥0.25 AND orientation-correct, vs the greedy baseline.

Provenance: orientation findings in `docs/curriculum-close-loop.md` (Exp B / B′),
`artifacts_scripts/curriculum/intro_orientation.py`, `ordering_genre_firmup.py`.
