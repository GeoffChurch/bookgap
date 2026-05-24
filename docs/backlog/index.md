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
- [port-refine-pipeline-for-resid](gap-detection-methods/port-refine-pipeline-for-resid.md) — production resid is raw-deflation only; port LLR-refine→dedup→effective_size behind the resid option if the deferred ollama LLM-coverage check shows raw resid underperforms its refined bake-off numbers
- [radius-fraction-bakeoff](gap-detection-methods/radius-fraction-bakeoff.md) — sweep the never-tuned k-ball `radius_fraction` (fixed 0.75/k=1 all along) with resid+recall@K curve across the 4 topics, plus a **data-driven per-seed radius** arm (σ-tied / margin / density-adaptive) — directly tests whether n_eff's book-rich shortfall is a fixed-radius handicap vs a formula issue
- [kball-residual-mass](gap-detection-methods/kball-residual-mass.md) — four discount-by-book-claim variants for k-ball gap sizing (virtual-centroid / partition mutual-NN / global k-NN discount / KDE-on-KDE), implemented sequentially as alternative `gap.size_metric` options

## Cohesion metric design

Per-gap quality measurement and filtering.

- [multicluster-softmax-membership](cohesion-metric-design/multicluster-softmax-membership.md) — replace per-seed soft-EM with single global multi-cluster softmax over kNN basin centroids (incl. book-anchored basins as fixed competitors). Subsumes `tau_supp`, `min_final_size`, Phase 1/3 enrichment into a single E-step output. Two variants: ∅-free vs. with marginal-density background ∅.
- [boundary-centroid-cosine](cohesion-metric-design/boundary-centroid-cosine.md) — cosine between gap centroid and boundary-books centroid; high = book exists, low = templated-data leak
- [topic-purity-axis](cohesion-metric-design/topic-purity-axis.md) — OpenAlex topic_id purity; useful at field/subfield scope (degenerate at single-topic scope)
- [dedup-cache-member-sets](cohesion-metric-design/dedup-cache-member-sets.md) — small optimization for Jaccard dedup loop
- [dual-landmark-eff-size-reporting](cohesion-metric-design/dual-landmark-eff-size-reporting.md) — report `eff_size_books`/`eff_size_chapters` per gap; **validated in the bake-off harness** (drives the LIP/LPP mirror-image result), production integration pending
- [data-quality-artifact-prefilter](cohesion-metric-design/data-quality-artifact-prefilter.md) — drop structural data debris (N3C, taxonomy, …); **bake-off reinforced**: genuine debris, binomial p discriminates it, raw sig/size ranking does not
- [book-rediscovery-eval-candidates](cohesion-metric-design/book-rediscovery-eval-candidates.md) — **living tracker**: methods worth running through the held-out eval. ⚠ Metric caveat: the eval was a full-set-coverage misnomer until fixed to genuine recall@20 (see `docs/gap_deflation_results.md`); prior conclusions resting on it must be re-checked
- [heldout-book-recovery-production-eval](cohesion-metric-design/heldout-book-recovery-production-eval.md) — promote held-out recovery to a production-grade standalone eval emitting a paper-ready recall@K table (genuine recall@K, not the full-set misnomer)

## Keyword extraction

(Mostly migrated from `artifacts_scripts/curriculum/TODO.md`. Some are cross-pollination from `mycooc/docs/backlog/`.)

- [spectral-neighborhood-signatures](keyword-extraction/spectral-neighborhood-signatures.md) — k×k cooc submatrix eigenvalues as token-quality signature
- [soft-hubness-correction-csls](keyword-extraction/soft-hubness-correction-csls.md) — iterative residual penalty for cross-topic-promiscuous tokens
- [confidence-stratified-denoising](keyword-extraction/confidence-stratified-denoising.md) — aggressive correction only on low-HHI tokens
- [lattice-additive-tokenisation](keyword-extraction/lattice-additive-tokenisation.md) — semtok-style overlapping segmentation with conductance weighting
- [doc-level-boilerplate-filter](keyword-extraction/doc-level-boilerplate-filter.md) — regex blacklist at ingest for scraped UI fragments in abstracts
- [display-side-keyword-dedup](keyword-extraction/display-side-keyword-dedup.md) — drop shorter-phrase keywords whose token sets are subsets of longer-phrase keywords (cosmetic only)
- [wire-keywords-into-batch-generator](keyword-extraction/wire-keywords-into-batch-generator.md) — attach `keywords` to every `GapWithContext` in `_compute_gaps_for_topic`; opportunity to audit `artifacts_scripts/curriculum/` for primitives that should move into `bookgap/`

## Infrastructure

Performance, indexing, refactoring.

- [gpu-kball-brute-force-knn](infrastructure/gpu-kball-brute-force-knn.md) — GPU brute-force kNN + manual radius filter to replace HNSW range search
- [cuml-hdbscan-pynndescent](infrastructure/cuml-hdbscan-pynndescent.md) — RAPIDS cuml.HDBSCAN or PyNNDescent for HDBSCAN speedup
- [cloud-denoiser-path-port](infrastructure/cloud-denoiser-path-port.md) — eliminate the `load_scope` bypass; route everything through Cloud(filter_pair, denoiser_config)
- [share-fullindex-knn-dual-landmark](infrastructure/share-fullindex-knn-dual-landmark.md) — factor full-corpus member kNN out of compute_gap_stats so dual-landmark calls don't duplicate it (~2×→~1.2×)
- [ff-stats-adaptive-calibration](infrastructure/ff-stats-adaptive-calibration.md) — partially validated: m-side calibration delivers 3-5× speedup; K-side rarely beats baseline at default tolerance (75/76 runs picked K=1500)
- [wire-ollama-generator-as-selectable-backend](infrastructure/wire-ollama-generator-as-selectable-backend.md) — `OllamaProposalGenerator` already exists but isn't config-selectable; wire it as an opt-in backend (default OpenAI) to save $$ + unblock local LLM-coverage
- [stream-ollama-proposal-metrics](infrastructure/stream-ollama-proposal-metrics.md) — move per-proposal embedding + coverage/gap-stats INTO the Ollama generation loop instead of deferring to a bulk pass at the end; gives `tail -f`-able coverage progress for hour-long runs
- [share-build-topic-index-helper](infrastructure/share-build-topic-index-helper.md) — `_build_topic_index` now verbatim-duplicated in production `deflation_strategies.py` + research `kball_gap_probe.py`; extract one shared `bookgap/` source (isolated refactor, byte-identity-gated)
- [columnar-works-audit](infrastructure/columnar-works-audit.md) — audit all ~50 `cloud.works()` callers; many only need 1-3 columns. Highest-impact: `subcloud.get_subcloud_retained_idxs` iterates 1.6M Python dicts even for filters that inspect 2-3 fields. Sketch: `Filter.required_columns()` parallel to existing denoiser `required_corpus_columns` → fast columnar path. **2026-05: hook landed (`Filter.compute_columnar_mask`); TemporalFilter wired**; further filters tracked below.
- [columnar-mask-non-empty-parent-filter](infrastructure/columnar-mask-non-empty-parent-filter.md) — extend `Filter.compute_columnar_mask` to handle non-None `subfield`/`topic`/`work_types`; today `TemporalFilter` falls back to dict iteration whenever any parent field is set
- [columnar-subclouds-for-topic](infrastructure/columnar-subclouds-for-topic.md) — wire columnar mask into `get_subclouds_for_topic` (154 SubClouds/run on AI subfield, currently dict-iteration). Blocked on `columnar-mask-non-empty-parent-filter`. Largest concrete win in production gap-detection runs.
