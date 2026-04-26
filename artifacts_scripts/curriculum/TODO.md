# Curriculum TODO

Backlog of experiments and improvements for the curriculum pipeline. Add or remove entries as we make progress.

Format: each item has a one-line summary; deeper context lives in linked notes / source files. Items are roughly ordered by impact-per-effort within each section.

---

## Keyword extraction

- [ ] **Vocabulary count floor at the keyword level.** Drop words with global count below a small threshold (e.g. 5) before running FW. With `weight_by_freq=True` this is mostly redundant but cheap belt-and-suspenders for the singleton-noise tail.

## Cross-pollination from mycooc

Ideas borrowed from `~/src/mycooc/`. The branching-entropy item subsumes the original "doc-level boilerplate filter" plan — we can detect contamination as a signal property of the tokens themselves rather than maintaining a regex blacklist.

- [ ] **Data-driven BE threshold (closed — absolute floor wins for stability).** The single-pass probe on `__24` (`be_phrase_probe.py`) compared four BE-threshold recipes:
  | recipe | wins | boilerplate | candidates |
  |---|---|---|---|
  | C: NPMI·c·BE (BE > 0.5 bits absolute, current default) | 7 | 1 | 3 750 |
  | D: NPMI·c·BE (BE > median of candidate BEs) | 7 | 1 | 2 536 |
  | E: NPMI·c·BE (BE > 0.4·H_W) | 5 | 1 | 58 |
  | F: NPMI·c·normalized_BE (norm_BE > 0.5) | 6 | 1 | 2 518 |
  Findings from porting each into the iterative production loop:
  - **D (quantile)**: degraded — Topic 2 got back `interference_spectroscopy_part`, `post_reddit_share`, `plain_text_citation_alert_save_article`, the giant linkedin/citeulike/mendeley chain. The BE distribution shifts each iteration (later passes see fewer candidates → median BE drops → quantile=0.5 admits boilerplate the absolute floor catches).
  - **F (count-normalised)**: equivalent to C on every real topic; only Topic 9 / Topic 13 (singletons) differ in *which* random noise tokens win the FW ranking. The intrinsic iteration-stability is theoretically nice (per-candidate normalisation doesn't depend on population shifts) but doesn't translate to a measurable production improvement on this corpus.
  - **E (α·H_W)**: dominated.
  Current production: `min_boundary_be=0.5` absolute (stable across iterations), `be_quantile=0.0` (off, available as opt-in stricter top-up combined via `max`), `normalize_be=False` (off, available as opt-in alternative score function). All knobs exposed in `mine_phrases_lattice` and `KeywordConfig.bigrams_*`. The iteration-stability principle (per-candidate normalization > population statistics in iterative-merge loops) is itself a generalizable insight; logged for mycooc as `mycooc-backlog/transport-and-optimization/per-candidate-vs-population-normalization-iterative.md`.
- [ ] **Lattice / longest-match iterative phrase parser (semtok-style).** *(Done — shipped 2026-04-25 with the lattice rewire of `summarize_topics`.)* Maintains the phrase vocabulary as a trie of underlying-token tuples, and on each iteration re-parses the raw corpus from scratch with longest-match against the entire current vocab. Naturally implements semtok's "exclusive entropy" effect — counts attributable to longer extensions never appear in shorter-phrase context distributions, because longer phrases consume their constituent positions on every parse. Reference: `semtok/src/semtok/tokenizer/morpheme_extractor.py:614–896`, `semtok/src/semtok/tokenizer/backend.py:150–215` for the original.
- [ ] **Spectral neighborhood signatures for token quality.** From `mycooc/docs/backlog/signatures-and-features/spectral-neighborhood-signatures.md`. Take each token's k×k cooc submatrix (its k nearest neighbours in cooc space), compute sorted eigenvalues as a signature. Boilerplate tokens form *degenerate* neighborhoods (all pointing to one hub); genuine topic tokens form *richer* neighborhoods. Could rank candidates pre-keyword-extraction or de-rank tokens whose spectral signature flags as boilerplate-shaped.
- [ ] **Soft hubness correction (CSLS-style).** From `mycooc/soft_coocmap.py:_gep_csls`. Iterative residual penalty for tokens that appear as candidates in too many topics' top-K simultaneously. Generalises bookgap's marginal-bg + freq-weighting: marginal demotes globally-frequent words once; CSLS-style demotes them iteratively, with soft (not hard) cutoffs that preserve borderline genuine domain words. Probably overkill given marginal+freq already drives cross-topic crowding to ~zero, but worth knowing about.
- [ ] **Confidence-stratified denoising.** From `mycooc/docs/backlog/hubness-and-gep/gep-confidence-stratified.md`. Apply aggressive correction only to *low-confidence* (low-HHI) tokens; leave *high-confidence* (peaked, well-supported) tokens untouched. For bookgap: stronger boilerplate/noise filtering on diffuse candidates (scattered across many docs weakly), conservative on focused candidates (concentrated in a few high-relevance docs). Naturally pairs with branching-entropy as the confidence signal.
- [ ] **Lattice / additive tokenisation (semtok-style).** From `mycooc/vocab/semtok_vocab.py`. Both `un` and `unhappy` coexist as tokens; conductance-based forward-backward weighting picks among overlapping segmentations without a count floor. More complex than greedy left-to-right merging and addresses the order-dependence we accepted in the iterative phrase miner — rarely needed for canonical scientific phrases but useful for less canonical text. Lower priority.

## Data hygiene

- [ ] **Doc-level boilerplate filter for abstracts.** *(Now subsumed by the branching-entropy item above for the keyword path; still relevant for embedding/LLM-proposal-gen if those want a hard reject signal rather than a token-level demotion.)* High-count UI fragments observed in the bigram pass (`Articles citing`, `Reflect usage`, `Get alerts`, etc.) come from papers whose entire `abstract` field is scraped page content, not the actual abstract. A regex set applied at snapshot-ingest time would benefit every downstream consumer, not just keywords.

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
- ~~Iterative phrase mining (trigrams via 2 passes)~~ — shipped 2026-04-25. `bigram_iterations` parameter, default 2. On `__24` surfaces `einstein_podolsky_rosen`, `american_institute_physics`, `detection_photon_pairs` plus new iter-2 bigrams (`crystal_lattices`, `party_alice`, `party_bob`, `algebraic_alphabet` etc.).

Removed (kept for context, do not pursue):
- ~~Linearly-combined bidirectional PageRank (`combine="linear"`, θ=0.5)~~ — scrapped 2026-04-25 in favour of the existing `product` combine.
- ~~Hybrid α-mixed background for keyword extraction~~ — superseded by the marginal-bg + freq-weighting formulation, which has no knob (shipped 2026-04-25).
- ~~Display-side budget (top-K-uni + top-K-bi)~~ — no longer needed; marginal-bg + freq drives cross-topic crowding to ~zero on its own.
- ~~MMR-style top-K diversification~~ — superseded by marginal-bg + freq for the same reason.
- ~~External background corpus~~ — superseded by marginal background, which subsumes it as a special case.
