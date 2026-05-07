# Backlog

Unscheduled ideas, one per file. Each file is self-contained — describes the idea, why it might work, and any concrete variants to test. References to source files or notes are supplementary context.

Convention: when an idea is executed (and either validated or refuted via experiment), delete the file. The idea's permanent home is then the experiment writeup or a note under `docs/`. If the idea is *refuted*, move it to `docs/dead_ends/<theme>/` with a failure-and-diagnosis writeup.

## Gap detection methods

How candidate gaps are produced and refined.

- [article-as-seed-better-inclusion-policy](gap-detection-methods/article-as-seed-better-inclusion-policy.md) — lower min_post threshold, mean_post objective
- [article-as-seed-lockstep-expansion](gap-detection-methods/article-as-seed-lockstep-expansion.md) — all seeds expand simultaneously; same-cardinality competitors as background
- [article-as-seed-higher-max-iters](gap-detection-methods/article-as-seed-higher-max-iters.md) — adaptive cap for book-poor topics where natural cluster size > 500
- [cohesion-cut-on-slt](gap-detection-methods/cohesion-cut-on-slt.md) — replace max-bookless's "stop at first book" with a cohesion-aware cut criterion on the single-linkage tree
- [multi-start-union-pipeline](gap-detection-methods/multi-start-union-pipeline.md) — combine candidates from fixed/moving-centroid/hill-climb/article-seed/cohesion-cut, dedup at the end
- [kball-vs-hdbscan-statistical-scoring](gap-detection-methods/kball-vs-hdbscan-statistical-scoring.md) — finish the cross-scope sweep with `bookgap/depletion_test.py`
- [hdbscan-relevance-weighted-replication](gap-detection-methods/hdbscan-relevance-weighted-replication.md) — fold per-work weights into HDBSCAN via point replication

## Cohesion metric design

Per-gap quality measurement and filtering.

- [boundary-centroid-cosine](cohesion-metric-design/boundary-centroid-cosine.md) — cosine between gap centroid and boundary-books centroid; high = book exists, low = templated-data leak
- [topic-purity-axis](cohesion-metric-design/topic-purity-axis.md) — OpenAlex topic_id purity; useful at field/subfield scope (degenerate at single-topic scope)
- [dedup-cache-member-sets](cohesion-metric-design/dedup-cache-member-sets.md) — small optimization for Jaccard dedup loop

## Keyword extraction

(Mostly migrated from `artifacts_scripts/curriculum/TODO.md`. Some are cross-pollination from `mycooc/docs/backlog/`.)

- [spectral-neighborhood-signatures](keyword-extraction/spectral-neighborhood-signatures.md) — k×k cooc submatrix eigenvalues as token-quality signature
- [soft-hubness-correction-csls](keyword-extraction/soft-hubness-correction-csls.md) — iterative residual penalty for cross-topic-promiscuous tokens
- [confidence-stratified-denoising](keyword-extraction/confidence-stratified-denoising.md) — aggressive correction only on low-HHI tokens
- [lattice-additive-tokenisation](keyword-extraction/lattice-additive-tokenisation.md) — semtok-style overlapping segmentation with conductance weighting
- [doc-level-boilerplate-filter](keyword-extraction/doc-level-boilerplate-filter.md) — regex blacklist at ingest for scraped UI fragments in abstracts
- [display-side-keyword-dedup](keyword-extraction/display-side-keyword-dedup.md) — drop shorter-phrase keywords whose token sets are subsets of longer-phrase keywords (cosmetic only)

## Infrastructure

Performance, indexing, refactoring.

- [gpu-kball-brute-force-knn](infrastructure/gpu-kball-brute-force-knn.md) — GPU brute-force kNN + manual radius filter to replace HNSW range search
- [cuml-hdbscan-pynndescent](infrastructure/cuml-hdbscan-pynndescent.md) — RAPIDS cuml.HDBSCAN or PyNNDescent for HDBSCAN speedup
- [cloud-denoiser-path-port](infrastructure/cloud-denoiser-path-port.md) — eliminate the `load_scope` bypass; route everything through Cloud(filter_pair, denoiser_config)
