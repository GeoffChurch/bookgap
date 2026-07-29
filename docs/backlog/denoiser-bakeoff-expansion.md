# Denoiser bake-off expansion (living doc)

> **Note — tooling removed.** The `artifacts_scripts/curriculum/` modules referenced in this doc were deleted in `0f2d0ad`; findings stand, commands do not. See `docs/curriculum-pipeline-removed.md`.

Tracks proposed and tried denoiser arms beyond the current head-to-head
in Appendix B. Update as arms are queued, run, or refuted. Migrates to
`docs/denoiser-bakeoff-results.md` (or similar) once the bake-off
stabilizes for a publication.

## Current bake-off state

**Locked-$k$ design:** $k=10$ for GIST (retain 374D of 384D), $k=394$
for Gemma (retain 374D of 768D). Apples-to-apples 374D operating point
matching the rest of the paper.

**Current arms** (post 2026-05-26 trace-normalization fixes):
- NONE, LPP, ANTI_LDA, ANTI_CSP, CCP, CCLIP, LIP, JAD_EQUAL,
  AUG_LPP, CITATION_ERROR_LDA, SYM_NORM_LPP.
- JAD_EQUAL_DEPRECATED (cached pre-fix data, kept for pre/post comparison).

**Topic set:** 20-topic random subset of the 77 AI topics
(`TOPIC_SUBSET_RANDOM_20` in
`artifacts_scripts/gemma4_quantitative_sweep.py`).

## Arms — queued for future runs

### From existing `denoisers.py` (no new code)

| Arm | Constructor | Why | Cost | Expected outcome |
|---|---|---|---|---|
| **WC_CCLIP** family (TN) | `WithinClassCCLIPDenoiser(α, normalize_traces=True)` for α∈{0, 0.01, 0.1, 0.5, 1} | Uses $S_W$ as constraint matrix instead of $S_T$. Code-only audit flagged $S_W$ as the "principled" denominator for class-conflation since $S_T$ implicitly rewards the between-class separation the numerator is suppressing (objective conflict). Never properly tested under the trace-normalization fix. | 6 cells × 2 embedders = 12 cells (~4-6 h compute) | Likely competitive with the S_T-denominator CCLIP family; may or may not beat LPP |
| **CCLIP_01** (TN, α=0.01) | `CCLIPDenoiser(alpha=0.01, normalize_traces=True)` | Historical Gemma top performer at 79.90% — but under the bug, so α was effectively 1. Re-test under the fix to verify whether the small-α regime is genuinely interesting or was an artifact. | 2 cells (~40 min-1 h) | Either tracks CCP closely (small-α ≈ pure class-conflation) or fills a gap in the α-grid we don't see today |
| **AUG_LPP at small k** | Same enum, varied k | Historical best was at k=2 (89.26% on GIST, beating LPP's 86.52 at k=10). Locked-$k$ design hides this. | 2 cells at k=2 (GIST) + 2 at k=50 (Gemma) | Might genuinely change headline ranking on GIST |

### From published literature (new code required)

| Arm | Paper | Why | Cost | Expected outcome |
|---|---|---|---|---|
| **DLPP** | Yu, Teng & Liu 2006, *Image and Vision Computing* 24(3):239–248 — [doi link](https://doi.org/10.1016/j.imavis.2005.11.006) | Most adjacent published method to CCLIP. Uses a *ratio* of locality and class-scatter (not convex combination). Tests "did we re-invent DLPP?" + gives the paper a published-prior-art comparison. | ~80 LOC: one new GEP solve, scatter computed from class-Laplacian-of-classes | Probably competitive but differentiable from CCLIP. Will at minimum strengthen §3 narrative. |
| **MFA** | Yan, Xu, Zhang et al. 2007, *TPAMI* 29(1):40–51 — [doi link](https://doi.org/10.1109/TPAMI.2007.250598) | Replaces class-membership scatter with K-NN graphs: intraclass-compactness vs interclass-separability. Nonparametric; doesn't assume Gaussian-per-class. **Likely the best fit for our books-vs-articles data** since classes aren't tight clusters in embedding space. | ~120 LOC: K-NN graph construction + GEP | Most likely "actually beats LPP" candidate |
| **Regularized LPP** | Cai, He & Han 2007, *IEEE TKDE* 19(11) — [doi link](https://doi.org/10.1109/TKDE.2007.190614) | Adds Tikhonov ridge: $X^T L X \mathbf{a} = \lambda (X^T D X + \varepsilon I) \mathbf{a}$. Tests whether numerical conditioning matters. | ~10 LOC tweak to `LPPDenoiser` + an `epsilon` flag | Most likely "doesn't change much" — clean datapoint for the paper |
| **scatter-based CSP (scaCSP)** | Cao et al. 2023, [arXiv:2303.06019](https://arxiv.org/abs/2303.06019) | Unified framework subsuming CSP and FDA via scatter matrices. Could clarify CSP↔LDA relationship on our document (not trial) data. | ~50 LOC | Speculative — may or may not differ meaningfully from ANTI_CSP+ANTI_LDA |
| **Nonparametric DLPP (NDLPP)** | Various follow-ups to Yu et al. 2006 | KNN-based scatter rather than class-mean scatter. Should handle book/article distinction better since neither class is Gaussian. | ~100 LOC | More likely than vanilla DLPP to beat LPP |
| **L1-norm robust LPP** | e.g. Pang et al., [Fisher discriminant L1-norm](https://ieeexplore.ieee.org/document/6571245/) | Robust to outliers; might help the language/data-quality failure topics (Spanish/German articles, Latin species names). | ~150 LOC; iterative not closed-form | Speculative; unclear if outliers are a meaningful limiting factor here |

## CSLS-inspired ideas (novel applications)

CSLS (Conneau et al. 2018, "Word translation without parallel data" [arXiv:1710.04087](https://arxiv.org/abs/1710.04087)) corrects similarity scores for hubness: $\mathrm{CSLS}(x,y) = 2\cos(x,y) - r_k(x) - r_k(y)$, where $r_k(x)$ is $x$'s mean cosine to its $k$ nearest neighbors in the *other* class. The project already uses CSLS-style hubness corrections at the keyword level (see [`keyword-extraction/soft-hubness-correction-csls.md`](keyword-extraction/soft-hubness-correction-csls.md), which originates from `mycooc/soft_coocmap.py:_gep_csls`). The denoiser/gap-detection level hasn't been explored.

| Idea | Mechanism | Rationale | Cost | Status |
|---|---|---|---|---|
| **CSLS-weighted LPP affinity** | Replace cosine-based affinity in LPP's adjacency $W$ with CSLS-based affinity. Articles that share hub neighbors get *lower* weight; mutually-close articles get higher weight. | The standard LPP affinity treats all neighbor edges equally. Hubness asymmetry is information being thrown away. Could give a more discriminative locality signal. | ~30 LOC (modify affinity computation in `LPPDenoiser`) | Queued |
| **Hubness-suppressing denoiser** | Find subspace dimensions where the rank-asymmetry (hubness) of nearest-neighbor relationships is maximally concentrated; project them out. Construction: for each candidate direction, compute Gini coefficient (or similar) of in-degree distribution under cosine ranking; drop high-Gini directions. | Genuinely novel as far as the literature pass surfaced. Hubness is a known artifact of high-dim isotropic embeddings; suppressing it should improve any downstream geometric task. Conceptually parallel to Anti-LDA but in the rank-asymmetry domain rather than class-discriminability. | ~150 LOC: per-direction hubness statistic + ranking; could be expensive (O(N²) per direction unless approximated) | Speculative |
| **CSLS-corrected gap detection** | Replace the "nearest book" Voronoi-membership test with CSLS-based nearest-book. A book is x's nearest only if x is in the book's CSLS-neighborhood too. | Addresses the "hub book" problem where a popular book gets attributed many articles in its vicinity regardless of true topical relevance, shrinking detected gaps. Cross-references `gap-detection-methods/kball-residual-mass.md` Variant 3 (per-partition mutual-NN), which is conceptually similar (variable-k mutual-NN) but for gap *sizing* not *membership*. | ~50 LOC modification to gap-detection nearest-book lookup | Cross-ref [kball-residual-mass](gap-detection-methods/kball-residual-mass.md); the gap-sizing variant probably ships first |

## $k$-value extension plans

| Plan | Cells added | Cost | Trade-off |
|---|---|---|---|
| **Round 1 — locked $k$** (10 / 394, retain 374D both) | 0 | 0 | Apples-to-apples 374D; matches embedders on *absolute* retained-D but at very different retained-fractions (97% GIST, 49% Gemma) |
| **Round 2 — retain-128** (drop_k=256 GIST, drop_k=640 Gemma) | 10 denoisers × 2 embedders = 20 cells | ~8-12 h | Stress test: more aggressive denoising (33% / 17% retained). Gives 2nd point per curve; expect Gemma to degrade more sharply due to its lower retained fraction. Queued in `gemma4_quantitative_sweep.py`. |
| **Round 3 — autonomous best-$d$ per (denoiser, embedder, metric)** ✅ implemented | per-pair adaptive (capped 8 evals) | ~30 h worst case, ~17 h realistic | Promoted to "Option C" after round 2 revealed (a) class-suppression methods still improving at d=128 (might want lower); (b) metrics can diverge under aggressive denoising — coverage and proposal-percentile no longer agree at low d; (c) need to defend against degenerate metric saturation at very low d. Seeds d=1 explicitly as a degeneracy probe, then bisects around per-metric argmax. PCA included as the "generalization of NONE" null baseline. Implementation: `artifacts_scripts/autonomous_d_search.py` (planner), `run_cells_from_json.py` (dispatcher), `run_autonomous_d_search_loop.sh` (loop). |
| **Option B (deferred): 3-point sweep** ({2, 10, 50} GIST etc.) | 3 per denoiser × 2 embedders | ~9-18 h | Space-filling alternative; superseded by Round 3 |
| **Round 4 — GP-suggested $k$ (future)** | per-denoiser, varies | per cell | If round 3's adaptive bisection misses interior maxima or has high noise, fit per-denoiser GP on $\log k \to$ coverage with hierarchical sharing; pick next $k$ per embedder by max Expected Probability of Ranking Change (EPRC). ~80 LOC sklearn GP. |

**Rationale for retain-128 (round 2):** the existing $(10, 394)$ point is so mild on GIST (97% retained) that the denoiser is barely doing anything; we need a more aggressive point to see ranking changes. 128 is a defensibly-round number (power of 2), continues the absolute-retained-D convention, and is aggressive enough on both embedders that we should observe meaningful degradation in at least some denoisers — which is the diagnostic value.

**Failure mode to watch:** if 128 is too aggressive and all denoisers collapse toward NONE-like performance, ranking resolution is lost. Mitigation: round 3 picks an intermediate point (e.g., retain-256) using the round-2 data.

## Cross-pollination tracker

### From mycooc → bookgap

The mycooc backlog has two directly-relevant themes: `hubness-and-gep/` and `signatures-and-features/`. CSLS-style hubness mitigation has already crossed over once at the keyword level (see [`keyword-extraction/soft-hubness-correction-csls.md`](keyword-extraction/soft-hubness-correction-csls.md), origin `mycooc/soft_coocmap.py:_gep_csls`).

| Idea | Origin | Mechanism | Application to bookgap | Cost | Priority |
|---|---|---|---|---|---|
| **HHI bisection** | `mycooc/transport.py:25-50` (`bisect_hhi_temperatures`) | Binary-search per-row softmax temperatures so each row's distribution has target HHI ≈ 1/effective-k. Normalizes for row-magnitude differences. | **(a)** Soft Voronoi membership for gap detection: each article's distribution over books has target effective-k, then "gap" = articles whose mass on the gap centroid exceeds threshold (replaces hard nearest-book). **(b)** HHI-stratified denoising: only apply LPP/CCLIP/etc. to low-HHI (low-confidence) points; preserve high-confidence embeddings (cf. `mycooc/docs/backlog/hubness-and-gep/gep-confidence-stratified.md`). **(c)** Per-seed kball radius: pick radius so resulting article-set has target HHI on a relevant distribution. | (a) ~60 LOC; (b) ~80 LOC; (c) ~40 LOC | (a)/(c) high — addresses kball-residual-mass; (b) medium — interesting variant |
| **CSLS as Laplacian preprocessor** | `mycooc/docs/backlog/signatures-and-features/csls-as-laplacian-preprocessor.md` (which itself originates as bookgap → mycooc flow via `bookgap/artifacts_scripts/curriculum/multi_view_spectral.py::csls_correct`) | Apply CSLS correction to the W matrix BEFORE forming the Laplacian, not at retrieval. Mycooc note: "hub points appear as high-degree nodes... clusters get distorted"; fixing at the matrix level fixes the eigenstructure. | **(a)** CSLS-corrected affinity in LPP: replace cosine-based W with CSLS-corrected W (clipped at 0) before forming `L = D - W`. **(b)** CSLS-corrected scatter in CCLIP family: apply CSLS to the per-class scatter inputs. | (a) ~30 LOC; (b) ~50 LOC | Already lives in `multi_view_spectral.py` for clustering; (a) extends it to LPP denoiser |
| **gep-auto-alpha** | `mycooc/docs/backlog/hubness-and-gep/gep-auto-alpha.md` | Per-pair auto-tuned α from data | Auto-tune CCLIP's α from data (e.g., minimize NN-classification of class labels in the projected subspace), not as a hyperparameter | ~80 LOC | Medium — would defuse "why this α?" reviewer concern |
| **gep-confidence-stratified** | `mycooc/docs/backlog/hubness-and-gep/gep-confidence-stratified.md` | Apply denoising only to low-HHI (bottom-33%) subset | "Conservative denoising" arm: identify high-confidence works (high local-NN HHI) and keep their embeddings unchanged; denoise only ambiguous works. Could combine with `kball-residual-mass`'s virtual-centroid variant. | ~100 LOC | Speculative but interesting |
| **partial-density-shrinking, per-dim reliability weighting, softer spectral subtraction** | `mycooc/docs/backlog/signatures-and-features/` | Various scoring/weighting variants of denoising | Read in detail before committing — likely some are direct bake-off arms | per-item ~50-100 LOC | Defer to next review pass |

### From bookgap → mycooc (reverse osmosis)

Established channel (CSLS-preprocessor already flowed from bookgap to mycooc backlog). Worth nominating for future flow:

| Idea | Bookgap origin | Mycooc application |
|---|---|---|
| **Trace-normalization for combined scatter matrices** | The bug we just fixed in `CCLIPDenoiser`, `JADDenoiser`, `AntiCSPDenoiser`, `AugmentedLPPDenoiser`: when combining $S_B + S_L$ (or any additive mix of scatter operators), pre-rescale each to `tr=d`. Without this the operator with the larger trace dominates. | Mycooc's GEP-based methods (`hubness-and-gep/`) likely combine scatter matrices similarly; same fix likely applies. Specifically `gep-pair-heterogeneity.md` and any "fused" GEP variants. |
| **Dual-landmark eff-size reporting** | `cohesion-metric-design/dual-landmark-eff-size-reporting.md` | Mycooc might benefit from reporting per-cluster effective-size on multiple landmark sets (e.g., dictionary words + frequency-stratified words). |
| **K-ball + virtual-centroid gap sizing** | `gap-detection-methods/kball-residual-mass.md` Variant 4 | Could replace mycooc's cluster-membership scoring with a "virtual prototype vs real prototypes" Voronoi check. |
| **Per-class trace normalization for CSP-style methods** | This session's lit-pass finding (Ramoser 2000 canonical normalization) | If mycooc uses any CSP-like or Anti-LDA-like decomposition on imbalanced class sizes, per-class trace normalization may have been overlooked. |
| **LIP as a denoiser arm** | Round 1 bake-off finding (2026-05-26): LIP is dead last on GIST but first on Gemma — a clean ranking inversion suggesting dimensionality-dependent behavior worth probing in mycooc's keyword-embedding context. | Mycooc's GEP family currently only has *residual-projection* variants ($E^\top E$ vs $X^\top X + \varepsilon$). LIP-style "class-scatter as $B$" projection would be a sharper class-suppression tool — particularly for language-as-class. |
| **z-PMI word embeddings + regularized-LDA arm** | The keyword-extraction Fightin' Words finding (success of $\delta / \sqrt{\text{var}}$ over raw log-odds) — see [[document-representations/zscored-logodds-doc-embeddings]] and [[keyword-extraction/topic-fisher-signatures]]. | **Two ideas for mycooc.** (a) **z-PMI embeddings**: SVD of $\text{PMI}(w,c) / \sqrt{\widehat{\text{Var}}(\text{PMI})}$ instead of plain SVD(PMI). Variance from binomial-of-occurrence ($\propto 1/\text{count}$). Up-weights statistically reliable co-occurrences without GloVe-style ad-hoc weighting; nobody appears to have done this explicitly. Plausibly a methods note in its own right. (b) **Regularized-LDA GEP arm**: $S_B v = \lambda (X^\top X + \varepsilon I) v$ with class = language / domain / any side-channel. Cross-pollinated from this session's "Fisher is the Gaussian cousin of Fightin' Words" observation. |

## Results log

Bake-off results are mirrored to `outputs/gemma4_sweep/run_summary_gap0.csv` and rendered into `paper/artifacts/latex_tables/gap0/articles_10_books_5/combined_best_denoisers.tex`. This section captures *qualitative* notes per arm as it's tried — surprises, gotchas, "did the audit prediction hold."

### 2026-05-26 sweep (in progress at the time of writing)

- **ANTI_CSP** (fixed, $k=10$): GIST cov 78.86, pct 72.78, 2 failures. Lands with ANTI_LDA / LIP / NONE — class-conflating methods cluster together as audit predicted. ~9pp below LPP on coverage.
- **AUG_LPP** (fixed, $k=10$): GIST cov 80.36, pct 76.40, 1 failure. Middling. As predicted by the locked-$k$ analysis (AUG_LPP wants $k=2$). Worth re-running at small-$k$ before drawing conclusions.
- **CITATION_ERROR_LDA, SYM_NORM_LPP, JAD_EQUAL (fresh fixed)**: pending.
- **Gemma cells**: pending.

## Refuted / parked

(empty — populate as ideas don't pan out, with diagnosis. Per [`docs/dead_ends/index.md`](../dead_ends/index.md), promote refuted ideas with a failure-and-diagnosis writeup.)
