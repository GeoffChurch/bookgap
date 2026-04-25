# Curriculum TODO

Backlog of experiments and improvements for the curriculum pipeline. Add or remove entries as we make progress.

Format: each item has a one-line summary; deeper context lives in linked notes / source files. Items are roughly ordered by impact-per-effort within each section.

---

## Keyword extraction

- [ ] **Trigrams as a follow-up to bigrams.** With bigrams now in production, run a second NPMI pass on the bigrammed corpus to catch `einstein_podolsky_rosen`, `quantum_key_distribution`. Cheap on top of existing bigram infra; expect only a handful of trigrams to survive at sensible thresholds.
- [ ] **Vocabulary count floor at the keyword level.** Drop words with global count below a small threshold (e.g. 5) before running FW. With `weight_by_freq=True` this is mostly redundant but cheap belt-and-suspenders for the singleton-noise tail.

## Data hygiene

- [ ] **Doc-level boilerplate filter for abstracts.** High-count UI fragments observed in the bigram pass (`Articles citing`, `Reflect usage`, `Get alerts`, `Reuse`, `Published in Issue`, `Advertisement return`, `EasyBib`, `Bookends`) appear at counts ~120 spread over ~120 papers — strongly suggests those papers' `abstract` field is entire scraped page content, not the actual abstract. Right move is *document-level rejection* (skip the paper for keyword extraction; consider also for embedding) rather than token-level strip. Investigate a sample first to confirm; build a small regex set; apply in the snapshot ingest path so the cleanup benefits every downstream consumer (LLM proposal generation reads abstracts too).

## Curriculum clustering — experiments still on the table

(See `CLUSTERING_NOTES.md` for the singleton-blob diagnosis and weighted-bisecting-kmeans solution that's already shipped.)

- [ ] **`min_cluster_weight` at scale.** We shipped the parameter but haven't run a full curriculum with it. Natural successor to the `min_cluster_size=10` experiment — singletons that "earn it" by relevance mass survive, low-relevance globs get merged. Pick a threshold from the topic mass distribution (e.g. 0.5× mean topic mass).
- [ ] **Leiden on the top-5000 citation subgraph.** Citations are the semantic relation curriculum *means*, not embedding distance. Should be the highest-quality topic structure. Needs `igraph` + `leidenalg` (~30–50 lines).
- [ ] **Average-linkage agglomerative.** Different shape bias than bisecting K-means — control experiment. Tractable on top-5000 (n² fits in memory).
- [ ] **HDBSCAN on top-5000 with relevance-weighted point replication.** Density-based, naturally variable cluster sizes, no forced cluster count. Replicate each work proportional to relevance to fold weighting into HDBSCAN's unweighted core.

## PageRank variants

- [ ] **Multi-source bidirectional run.** Exercise the now-unified two-direction PR with a non-empty `source_work_ids` (textbook foundations the user already knows). Quality test for the bidirectional path beyond the forward-only fallback.

## HDBSCAN gap detection (paused thread)

- [ ] **Pick a winner among denoisers.** The probe (`artifacts_scripts/hdbscan_gap_probe.py`) tested PCA, multi-class LDA, LPP, LIP, CCLIP, hierarchical, criteria_union, rNOJD. rNOJD and criteria_union looked promising but we never converged on a final choice. Resume by re-running with the binomial p / FDR / effect-size scoring on a held-out evaluation set.

---

Done:
- ~~NPMI bigram mining → production~~ — shipped 2026-04-25. Frequency-weighted NPMI ranking, no count floor needed. `mine_bigrams` and `apply_bigrams` in `keyword_extraction.py`; `KeywordConfig.bigrams_*` knobs; default ON.

Removed (kept for context, do not pursue):
- ~~Linearly-combined bidirectional PageRank (`combine="linear"`, θ=0.5)~~ — scrapped 2026-04-25 in favour of the existing `product` combine.
- ~~Hybrid α-mixed background for keyword extraction~~ — superseded by the marginal-bg + freq-weighting formulation, which has no knob (shipped 2026-04-25).
- ~~Display-side budget (top-K-uni + top-K-bi)~~ — no longer needed; marginal-bg + freq drives cross-topic crowding to ~zero on its own.
- ~~MMR-style top-K diversification~~ — superseded by marginal-bg + freq for the same reason.
- ~~External background corpus~~ — superseded by marginal background, which subsumes it as a special case.
