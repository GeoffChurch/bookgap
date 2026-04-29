# Curriculum TODO

Backlog of experiments and improvements for the curriculum pipeline. Add or remove entries as we make progress.

Format: each item has a one-line summary; deeper context lives in linked notes / source files. Items are roughly ordered by impact-per-effort within each section.

---

## Current verdict on clustering methods (2026-04-28)

**Reframe**: seeds (target/source works) are *fair game* — they don't need to remain as singletons; getting clustered with related works is fine, getting dropped to -1 is fine. What matters is: cluster topical coherence, keyword distinctness, mass distribution, and whether earned non-seed landmarks (Shannon-MTC-class works) emerge.

**Cross-corpus rankings** from `compare_clusterings.py` (QC: quantum-crypto, 0 sources; Robotics: vision→robotics, 4 sources):

| Method | QC Q | Rob Q | QC tpc/kw | Rob tpc/kw | Singletons | Notes |
|---|---|---|---|---|---|---|
| leiden | **0.759** | **0.638** | 1.41 | 1.80 | **0** (refuses) | Citation-graph champion when cleanly producing ≥ K real communities. |
| **multi_view_spectral** (CSLS + KRD-2011, λ auto-tuned) | **0.691** | 0.484 | 1.42 | 1.68 | 0 (refuses) | **Best cross-corpus exact-K alternative to leiden.** No mega-cluster pathology (gini 0.27/0.32, max% 15%/17%); all 15 clusters substantive on both corpora. Beats wbk on robotics by Q +0.175. |
| **weighted_bisecting_kmeans** | 0.384 | 0.309 | **1.54** | **1.32** | 0–4 (knob) | **Best earned-singleton-supporting method.** Best keyword distinctness on both corpora; surfaces non-seed landmarks. |
| WU (citation_avg_linkage, weighted_unified) | 0.320 | 0.287 | 1.18 | 1.27 | 5 | Close to wbk on QC; behind on robotics. |
| binary (citation_avg_linkage) | 0.258 | 0.169 | 1.09 | 1.15 | 5–7 | Dominated by WU on QC, by wbk on robotics. |
| WPD (citation_avg_linkage, weighted_per_direction) | 0.248 | 0.176 | 1.18 | 1.12 | 5–8 | Most singletons but lowest Q among citation_avg variants. |
| average (avg-linkage on embeddings) | 0.378 | 0.131 | 1.24 | 1.09 | 5–8 | Cross-corpus unstable — worst Q on robotics. |
| spectral | 0.548 | 0.547 | 1.18 | 1.21 | 0 | Mega-cluster pathology on QC (max%=57%). |
| graph_bisect | 0.215 | 0.569 | 1.88 | 1.67 | 0 | Cross-corpus unstable — slices through real communities on QC. |

**Default**: `linkage_method = "weighted_bisecting_kmeans"` (`config.py:499`). Cross-corpus rationale confirmed by full sweep on 2026-04-28; retains the default because it's the only top-tier method that produces earned non-seed singletons (Shannon-MTC-class landmarks).

**When to deviate from wbk**:
- **leiden** if you want pure citation-driven communities, don't care about earned landmarks, and the corpus produces ≥ K real communities. Highest Q when it works.
- **multi_view_spectral** if you want exact-K with no mega-cluster pathology and balanced sizes, especially when leiden under-produces communities (it only found 9 real ones on robotics, padding with empties; multi_view_spectral produced 15 substantive ones). Best Q-vs-stability trade-off among non-singleton-allowing methods.
- **WU** if you specifically want citation-graph-driven clustering *with* the chance of earned non-seed singletons (it found 3 such on QC: EPR, Algebraic Alphabet, Bell-theorem QKD).

**Avoid**: average (cross-corpus instability), graph_bisect (cross-corpus instability + slicing real communities), WPD (no consistent advantage over WU), spectral (mega-cluster pathology).

**Multi-view spectral verdict (2026-04-28, complete)**: shipped with CSLS-corrected views and op-norm-based λ auto-tune (`λ = 1.5 · op_norm_per_K`). Did not dethrone leiden as the citation-graph champion (Q delta −0.07 on QC, −0.15 on robotics) and did not surface earned singletons (it's exact-K by construction). But: substantially beats wbk on robotics (Q +0.175) with cleaner cluster-size balance, and is the only method that produces 15 substantive clusters on both corpora without singletons or mega-cluster pathology. Net: a compelling third option in the table above, not a default replacement.

---

## Keyword extraction

- [x] **Vocabulary count floor at the keyword level.** *(Skipped 2026-04-28.)* Superseded by `weight_by_freq=True`: the FW score smoothly demotes low-count words proportional to count, so a hard binary floor is a degraded version of what the smooth weight already does. Reconsider only if a count-1/2 phrase ever sneaks into top-K with anomalously high NPMI — not observed to date.

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
- [x] **Lattice / longest-match iterative phrase parser (semtok-style).** *(Done — shipped 2026-04-25 with the lattice rewire of `summarize_topics`.)* Maintains the phrase vocabulary as a trie of underlying-token tuples, and on each iteration re-parses the raw corpus from scratch with longest-match against the entire current vocab. Naturally implements semtok's "exclusive entropy" effect — counts attributable to longer extensions never appear in shorter-phrase context distributions, because longer phrases consume their constituent positions on every parse. Reference: `semtok/src/semtok/tokenizer/morpheme_extractor.py:614–896`, `semtok/src/semtok/tokenizer/backend.py:150–215` for the original.
- [ ] **Spectral neighborhood signatures for token quality.** From `mycooc/docs/backlog/signatures-and-features/spectral-neighborhood-signatures.md`. Take each token's k×k cooc submatrix (its k nearest neighbours in cooc space), compute sorted eigenvalues as a signature. Boilerplate tokens form *degenerate* neighborhoods (all pointing to one hub); genuine topic tokens form *richer* neighborhoods. Could rank candidates pre-keyword-extraction or de-rank tokens whose spectral signature flags as boilerplate-shaped.
- [ ] **Soft hubness correction (CSLS-style).** From `mycooc/soft_coocmap.py:_gep_csls`. Iterative residual penalty for tokens that appear as candidates in too many topics' top-K simultaneously. Generalises bookgap's marginal-bg + freq-weighting: marginal demotes globally-frequent words once; CSLS-style demotes them iteratively, with soft (not hard) cutoffs that preserve borderline genuine domain words. Probably overkill given marginal+freq already drives cross-topic crowding to ~zero, but worth knowing about.
- [ ] **Confidence-stratified denoising.** From `mycooc/docs/backlog/hubness-and-gep/gep-confidence-stratified.md`. Apply aggressive correction only to *low-confidence* (low-HHI) tokens; leave *high-confidence* (peaked, well-supported) tokens untouched. For bookgap: stronger boilerplate/noise filtering on diffuse candidates (scattered across many docs weakly), conservative on focused candidates (concentrated in a few high-relevance docs). Naturally pairs with branching-entropy as the confidence signal.
- [ ] **Lattice / additive tokenisation (semtok-style).** From `mycooc/vocab/semtok_vocab.py`. Both `un` and `unhappy` coexist as tokens; conductance-based forward-backward weighting picks among overlapping segmentations without a count floor. More complex than greedy left-to-right merging and addresses the order-dependence we accepted in the iterative phrase miner — rarely needed for canonical scientific phrases but useful for less canonical text. Lower priority.

## Data hygiene

- [ ] **Doc-level boilerplate filter for abstracts.** *(Now subsumed by the branching-entropy item above for the keyword path; still relevant for embedding/LLM-proposal-gen if those want a hard reject signal rather than a token-level demotion.)* High-count UI fragments observed in the bigram pass (`Articles citing`, `Reflect usage`, `Get alerts`, etc.) come from papers whose entire `abstract` field is scraped page content, not the actual abstract. A regex set applied at snapshot-ingest time would benefit every downstream consumer, not just keywords.

## Curriculum clustering — experiments still on the table

(See `CLUSTERING_NOTES.md` for the singleton-blob diagnosis and weighted-bisecting-kmeans solution that's already shipped.)

- [ ] **`min_cluster_weight` at scale.** We shipped the parameter but haven't run a full curriculum with it. Natural successor to the `min_cluster_size=10` experiment — singletons that "earn it" by relevance mass survive, low-relevance globs get merged. Pick a threshold from the topic mass distribution (e.g. 0.5× mean topic mass).
- [x] **Leiden on the top-5000 citation subgraph.** *(Shipped 2026-04-26; clear winner on the quantum-crypto corpus.)* `linkage_method='leiden'` runs Leiden community detection on the citation subgraph, then truncates to top-N communities by size. At `leiden_resolution=0.5` on `__24`'s targets, produces a 90%-coverage / 0-singleton clustering with cluster sizes `[734, 685, 543, 461, 424, 342, 218, 200, 161, 156, 148, 127, 119, 114, 89]` and very distinct subtopics. **But:** on the vision→robotics corpus, Leiden only produced 9 real communities and the truncation padded with 6 empty topics; spectral clustering on the same inputs produced 15 distinct non-empty topics with finer subtopic structure (motion planning, visual servoing, UAV navigation, depth-from-defocus, etc.). **Choice between Leiden and spectral is corpus-dependent**: Leiden when natural community count is close to desired_n_topics; spectral when exact-N control matters or natural count is far below desired.
- [x] **Spectral clustering with exact K control.** *(Shipped 2026-04-26 as `linkage_method='spectral'`.)* sklearn's `SpectralClustering` on the symmetrised citation subgraph; takes K = `desired_n_topics` directly. On the vision→robotics corpus produced finer subtopic structure than Leiden: 15 distinct non-empty topics surfacing motion planning, visual servoing, UAV nav, depth-from-defocus, mathematical morphology, OCR — concepts Leiden lumped into broader generic topics. Tradeoff: more skewed cluster sizes (max=1840 vs 1163) and some 3–6-work fragments. Quality depends on corpus connectivity; on tightly-knit subgraphs both methods do well.

- [x] **Recursive Fiedler graph bisection (`graph_bisect`).** *(Shipped 2026-04-26 as an option; NOT recommended as default after objective evaluation.)* Pick-largest-first recursive bisection using each subgraph's Fiedler vector. The pick-largest heuristic produces uniform cluster sizes but **at the cost of slicing through real communities**. On objective metrics (`compare_clusterings.py`), graph_bisect scores worst-or-tied-worst on every metric on both corpora: lowest modularity Q, highest conductance, lowest citation flow capture, lowest keyword distinctness. The earlier subjective claim that graph_bisect "surfaces real subfields" was confirmation bias — its keywords appear in 1.7-1.9 topics on average, the highest cross-topic crowding of any method tried. Useful as an option for callers who want size-uniformity, but not the right default.

- [x] **Objective clustering quality comparison.** *(Shipped 2026-04-26 as `compare_clusterings.py`.)* Computes modularity Q, average conductance, citation flow capture, Gini of cluster sizes, max-cluster fraction, fragment count, singleton count, and keyword distinctness across all curriculum runs found under `artifacts/curriculum/`. Groups by (target_work_ids, source_work_ids) so within-corpus comparisons are fair. The Gini + max-fraction + fragment metrics directly catch the mega-cluster pathology that the original metrics (Q, conductance, flow) couldn't see — spectral on QC has gini=0.77, max=57% (one cluster swallowed 2847/5000 works); Leiden has gini=0.37, max=16%. **Method ranking is corpus-dependent and depends on whether earned singletons matter:** community-detection methods (Leiden) win the citation-graph metrics by structural design but cannot produce earned singletons (modularity Q penalises them by definition); mass-threshold-cut methods (citation_avg_linkage, weighted_bisecting, average) recover landmark singletons but at lower coverage.

- [x] **citation_avg_linkage = citation analog of weighted_bisecting.** *(Shipped 2026-04-26 (binary mode); weighted modes shipped 2026-04-27.)* Pairwise Jaccard / Ruzicka distance between (in+out) citation neighbourhoods → scipy avg-linkage → mass-threshold cut. Three modes selectable via `citation_jaccard_mode`:
  - `"binary"` (default): unweighted in+out neighbourhood, ordinary Jaccard.
  - `"weighted_unified"` (WU): row-normalised random-walk transition matrix `P[i,j] = A[i,j]/out_deg(i)`; neighbourhood is `max(P, P^T)`. Generalised Jaccard (Ruzicka). Down-weights citation-promiscuous reviewers automatically.
  - `"weighted_per_direction"` (WPD): separately ℓ1-normalise out-distribution (row-normalised P) and in-distribution (column-normalised A) per node, concatenate, then Ruzicka. Decouples out-degree noise from in-degree noise.
  Implementation uses the L1-distance identity `Ruzicka = (R_u + R_v − L1) / (R_u + R_v + L1)` (cdist with cityblock), so all three modes share a fast path. Tests: `tests/curriculum/test_citation_jaccard_modes.py`.
  
  **Cross-corpus results** (see top-of-file verdict table for the full ranking):
  - **QC**: WU > binary > WPD on Q (0.320 / 0.258 / 0.248). WU promoted Bell-theorem QKD to its own earned landmark singleton (binary kept it in a 4-cluster). WPD's directional separation didn't pay off here.
  - **Robotics**: WU > WPD > binary on Q (0.287 / 0.176 / 0.169), but **wbk beats all three on Q** (0.309) AND on keyword distinctness (1.32 vs 1.27/1.12/1.15). WU drops 1 source (Haralick) to -1; binary anchors a 52-work Haralick-citer cluster; WPD preserves all 7 seeds as singletons + 1 earned. Per the reframe, dropping seeds isn't bad in itself.
  
  **Net**: WU is the strongest citation_avg_linkage variant cross-corpus, but wbk dominates it on robotics. Default remains wbk.

## Multi-view clustering

- [x] **Multi-view spectral with CSLS-corrected views.** *(Design locked 2026-04-28; shipped 2026-04-28; auto-tune validated cross-corpus 2026-04-28.)* Generalises the (embedding-similarity, citation-similarity) trade-off past convex combination. Top-level convex blend was rejected because it collapses two evidence sources into one number before clustering — downstream can't tell whether (i,j) are close because abstracts agree or citations agree. Multi-view spectral keeps the views structurally separate through the eigendecomposition; coupling happens in embedding space, not in raw similarities.

  **Implementation** (`artifacts_scripts/curriculum/multi_view_spectral.py`, `linkage_method='multi_view_spectral'`):
  1. **Two views**: embedding cosine similarity (on denoised embeddings) and WU citation Ruzicka.
  2. **CSLS hubness correction per view** before forming the Laplacian.
  3. **Co-regularized multi-view spectral** (Kumar-Rai-Daumé 2011): alternating eigendecomposition of `L_v − λ U_w U_wᵀ` until subspace convergence; K-means on row-normalised concat `[U_1 | U_2]`.
  4. **λ auto-tune** via `λ = scale * (op_norm(L_emb) + op_norm(L_cite)) / (2 * K)`, default `scale=1.5`. Free (reuses the eigendecomposition the solver already computes for U_v init). `co_reg_lambda=None` (the new default) triggers it; numeric override still works.

  **Cross-corpus probe results (2026-04-28, `artifacts_scripts/curriculum/lambda_probe.py`):**

  | Corpus | n | K | Empirical optimum λ | Q at optimum | Q at λ=0 | Auto-tuned λ | Δ Q from optimum |
  |---|---|---|---|---|---|---|---|
  | QC quantum-crypto | 5000 | 15 | 0.100 | 0.693 | 0.206 | 0.097 | −0.019 |
  | Robotics vision→robotics | 5000 | 15 | 0.100 | 0.485 | 0.049 | 0.095 | **−0.002** |

  Both corpora hit the empirical optimum at λ=0.1 *exactly* and the sweet spot is broad (Q ≥ 95% of peak across λ ∈ [0.03, 0.3]). The op-norm-based auto-tune lands in the sweet spot on both corpora — a fixed default of 0.1 would also work, but the auto-tune is corpus-adaptive at zero extra cost and would track changes in Laplacian magnitude (e.g. different K, different denoiser, different csls_k).

  **Heuristic shootout** (cross-corpus, |Q-delta| from empirical optimum, smaller is better):

  | Heuristic | QC Δ Q | Robotics Δ Q | Status |
  |---|---|---|---|
  | **op_norm_per_K** | −0.019 | −0.002 | **Winner; shipped as auto-tune default** |
  | trace_per_n | −0.023 | −0.016 | Solid runner-up |
  | gap_K | −0.487 | −0.436 | Degenerate (collapses to ~0) |
  | sqrt_gap_product | −0.487 | −0.436 | Degenerate |
  | mean_bottom_K | invalid (negative) | invalid | Degenerate |
  | Kth | invalid | invalid | Degenerate |

  **Why bulk-spectrum heuristics dominate**: after CSLS+clip, both Laplacians are near-rank-deficient — the data decomposes into 15+ near-disconnected components, so the bottom-K eigenvalues are all at machine precision (~1e-16). Any heuristic indexed at the K-th eigenvalue or its gap collapses to numerical noise. Bulk quantities (trace, op_norm) are smooth in the matrix entries and stay informative.

  **Per-work confidence weights (optional, deferred):** scale rows of W_v by per-work confidence — embedding confidence (post-denoise L2 norm, abstract length, mean k-NN distance) and citation confidence (log(in_deg)+log(out_deg), 1/age). Skip unless a corpus emerges where the auto-tuned uniform-weight version underperforms.

  **Skipped/superseded approaches** (kept for context):
  - **Top-level convex-blend distance** (`α · cosine + (1-α) · Ruzicka`): rejected as too information-destructive. Bakes the trade-off into the distance itself, destroying which-view-said-what before clustering.
  - **Per-edge α via min/max/harmonic of per-work confidences**: subsumed by CSLS. "Non-hubness = informativeness" maps to confidence; CSLS is the operational version with hubness as the noise model.
  - **Kernel product** (`K_emb · K_cite`, "AND-like"): worth as a fallback if multi-view spectral underperforms on a future corpus; gives the categorical "meet" interpretation directly.
  - **CCA / PCCA shared latent**: principled (Bach-Jordan PCCA = generative model with per-view noise variance) but harder to wire into existing pipeline. Reserve for later.

- [x] **Cross-corpus comparison vs wbk and WU.** *(Completed 2026-04-28.)* Result: multi_view_spectral places #2 on QC (Q=0.691 vs leiden 0.759, wbk 0.384) and #4 on robotics (Q=0.484 vs leiden 0.638, graph_bisect 0.569, spectral 0.547, wbk 0.309). Beats wbk on robotics by Q +0.175 with cleaner structure (gini 0.32 vs 0.57, max% 17% vs 21%, 0 fragments vs 6); on QC by Q +0.307 with similar advantages (gini 0.27 vs 0.64). Does NOT dethrone leiden on Q on either corpus, but unlike leiden produces all 15 substantive clusters on robotics (leiden's truncation pads with empties; only 7-9 real communities). Keyword distinctness mid-pack: tpc/kw 1.42 (QC) / 1.68 (robotics) — comparable to leiden, slightly worse than wbk on robotics. **Verdict**: best "exact-K, no mega-cluster, cross-corpus stable" method; default remains wbk (which retains the earned-landmark advantage), but multi_view_spectral is the recommended deviation for corpora where leiden under-produces communities. See verdict-table at top of this file.
- [x] **Average-linkage agglomerative.** *(Run 2026-04-25; cross-corpus completed 2026-04-28.)* Tighter QC clusters than weighted_bisecting (cleaner keywords like `bell_inequality`, `gedankenexperiment`, `experimental_test_local_hidden_variable_theories`, `optical_parametric_noise`) but only 37% coverage. **Cross-corpus instability**: QC Q=0.378 (3rd-best), robotics Q=0.131 (worst). Not a viable default. Useful only as a sanity-check baseline.
- [ ] **HDBSCAN on top-5000 with relevance-weighted point replication.** Density-based, naturally variable cluster sizes, no forced cluster count. Replicate each work proportional to relevance to fold weighting into HDBSCAN's unweighted core.

## PageRank variants

- [x] **Multi-source bidirectional run.** *(Done 2026-04-26 as `curriculum_robotic_vision_1777207239` — vision→robotics curriculum with 4 source vision textbooks (Hartley, Forsyth, Horn, Haralick) and 3 target frontier robotics works (autonomous driving HD maps, context-aware navigation, semantic environment models). Leiden produced only 9 communities at resolution=0.5 because the citation neighborhood between sources and targets is much more tightly knit than the quantum-crypto run's. The 9 substantive topics form a coherent vision→robotics progression: general CV intro → projective geometry/Hartley → Haralick book → CV textbook → object recognition → semantic perception → embedded vision → DIBR → autonomous driving.)*

## Bugs surfaced by the bidirectional run

- [x] **Skip / mark empty topics in keyword extraction.** *(Shipped 2026-04-26 as `1226894` — empty topics now return `[]` keyword list instead of FW-on-zero-mass garbage.)* When Leiden produces fewer real communities than `desired_n_topics`, the truncation pads with empty (size-0) topics. The FW marginal+freq keyword extractor on empty-topic counts produces meaningless garbage tokens that look identical across all empty topics (`untextured, envircmmait, dry, affinely, submits, chronically, set_candidate, vxworks` etc. on `__2`'s topics 4–9). Fix: detect empty topics in `summarize_topics` (no docs sampled) and either skip them entirely or emit a sentinel `[empty]` keyword list. Also relevant for any clustering method that produces fewer-than-desired topics.

## Gap detection — state of the evidence

Three gap-detection algorithms in play:
- **K-ball** (`bookgap/llm/gap_algorithms/kball_algorithm.py`) — radius-based NN, in production via `batch_generator.py`.
- **Voronoi** (`bookgap/llm/gap_algorithms/voronoi_algorithm.py`) — Voronoi tessellation, also in production. Selectable via `CellType` enum.
- **HDBSCAN** (`artifacts_scripts/hdbscan_gap_probe.py`) — exploratory, paused.

**What we have quantitatively:**
- `artifacts/denoiser_analysis/denoiser_analysis_drop{1,2,5,10,20,50,100,374}_GIST_SMALL.csv` — 29 denoisers × 8 drop_k levels with intrinsic metrics (SNR, variance retention, scatter components). **WC_CCLIP family wins on SNR consistently** (drop_k=50/100/374 — all WC_CCLIP variants tie at top), then LPP family, then CCLIP/LIP family. Caveat: SNR is general embedding quality, not gap-detection performance directly.
- `artifacts/plots/{GIST_SMALL,GEMMA_300M}/{kball,avgref}_*_by_denoiser_*.png` — K-ball-specific gap-detection comparisons across denoisers and drop_k. Numbers exist behind the plots but not in tabular form (would need to re-run producing script to dump CSV).

**What we don't have (next-session focus):**
- Quantitative HDBSCAN denoiser comparison under a held-out metric. **Scoring decided 2026-04-28**: rNOJD-style binomial-p / FDR / effect-size on sphere occupancy.
- K-ball vs HDBSCAN side-by-side numbers under that same statistical scoring on the same scope.
- (K-ball-vs-Voronoi comparison **deprioritised 2026-04-28** — focus is K-ball-vs-HDBSCAN.)

**WC_CCLIP** (`bookgap/denoisers.py:865`, `WithinClassCCLIPDenoiser`) is worth knowing about: same numerator as CCLIP (`(1-α) S_B + α S_L`) but denominator is `S_W` (within-class scatter) instead of `S_T` (total scatter). Avoids the `S_T = S_B + S_W` objective conflict where the optimiser is implicitly rewarded for the very class separation it's trying to suppress. Equivalent to "Laplacian-regularised Anti-LDA." Variants: WC_CCLIP (α=0.5), WC_CCLIP_1 (α=1), WC_CCLIP_01 (α=0.1), WC_LIP (α=1, no S_B in numerator).

- [~] **K-ball vs HDBSCAN under statistical scoring.** *(Code shipped 2026-04-29; sweep run pending.)* Reusable `bookgap/depletion_test.py` module factors the binomial-p / Benjamini-Hochberg FDR / effect-size pattern out of `hdbscan_gap_probe.py` (with per-row pass-mask helper for CSV use). HDBSCAN probe refactored to use it; CSV output added at `artifacts/hdbscan_gap_probe/{scope}__{denoiser}__drop{drop_k}.csv`. **WC_CCLIP added** (S_W denominator instead of S_T) to denoiser dispatch + CLI choices. K-ball probe now also writes CSV at `artifacts/kball_gap_probe/{scope}__{algo}__rf{rf}__num{n}.csv` with the existing percentile/zscore/Shapiro stats — note that K-ball's "depletion test" is *not* the same per-gap signal as HDBSCAN's: K-ball gaps are by construction book-free spheres (radius stops at the nearest book), so binomial-on-occupancy collapses to "always extreme." K-ball's native gap-significance is the percentile/zscore/Shapiro framework; cross-algorithm comparison happens at "qualifying gap counts" rather than per-gap p-values. **Still TODO**: actually run the sweep (10 denoisers × multiple drop_k × representative scope) on HDBSCAN, plus a K-ball baseline, then aggregate the CSVs into a comparison table.
- [ ] **K-ball denoiser sweep.** Currently K-ball runs only with the cloud's default denoiser (NONE). The bypass path mirrors HDBSCAN's exactly: `load_scope(scope) → scope-fit denoiser → split by type mask → build scope-sized FAISS indices → kballs_with(ambient_index, landmark_index, query_emb)`. Cost is symmetric to HDBSCAN's per-denoiser flow (HDBSCAN's `fit_predict` also rebuilds its mutual-reachability tree per denoiser; same scope-restricted scale of work). The Cloud abstraction's `Cloud(denoiser='X')` path *would* be expensive — that's corpus-scope denoising on 40M works — but the bypass avoids it entirely and operates at scope scope just like the HDBSCAN probe does. Implementation: a new function in `kball_gap_probe.py` that mirrors `hdbscan_gap_probe.py`'s denoiser dispatch + uses `bookgap.index.kballs_with` directly with manually-built FAISS indices.
- [x] **Dump CSVs from the K-ball denoiser plots.** *(Shipped 2026-04-29.)* `artifacts_scripts/analyze_book_article_gap.py::plot_col_by_denoiser_pyplot` now saves a sibling CSV (`{y_col}_by_denoiser_{color_col=}_pyplot.csv`) with the same weighted mean+std the PNG visualises, ranked ascending. Same data the plot shows; tabular form for direct cross-run comparison.

---

Done:
- ~~Citation_avg_linkage weighted modes (WU, WPD)~~ — shipped 2026-04-27. Three modes via `citation_jaccard_mode` config: `binary` (default), `weighted_unified`, `weighted_per_direction`. Tests at `tests/curriculum/test_citation_jaccard_modes.py`.
- ~~Cross-corpus method comparison with full coverage~~ — completed 2026-04-28. wbk and average filled in for robotics; `compare_clusterings.py` re-run gives complete table. Verdict at top of this file.
- ~~Default linkage_method = wbk~~ — confirmed 2026-04-28 with quantitative cross-corpus rationale (best Q + best keyword distinctness on both corpora among singleton-allowing methods). No change to config — it was already there.
- ~~NPMI bigram mining → production~~ — shipped 2026-04-25. Frequency-weighted NPMI ranking, no count floor needed. `mine_bigrams` and `apply_bigrams` in `keyword_extraction.py`; `KeywordConfig.bigrams_*` knobs; default ON.
- ~~Iterative phrase mining (trigrams via 2 passes)~~ — shipped 2026-04-25. `bigram_iterations` parameter, default 2. On `__24` surfaces `einstein_podolsky_rosen`, `american_institute_physics`, `detection_photon_pairs` plus new iter-2 bigrams (`crystal_lattices`, `party_alice`, `party_bob`, `algebraic_alphabet` etc.).

Removed (kept for context, do not pursue):
- ~~Linearly-combined bidirectional PageRank (`combine="linear"`, θ=0.5)~~ — scrapped 2026-04-25 in favour of the existing `product` combine.
- ~~Hybrid α-mixed background for keyword extraction~~ — superseded by the marginal-bg + freq-weighting formulation, which has no knob (shipped 2026-04-25).
- ~~Display-side budget (top-K-uni + top-K-bi)~~ — no longer needed; marginal-bg + freq drives cross-topic crowding to ~zero on its own.
- ~~MMR-style top-K diversification~~ — superseded by marginal-bg + freq for the same reason.
- ~~External background corpus~~ — superseded by marginal background, which subsumes it as a special case.
