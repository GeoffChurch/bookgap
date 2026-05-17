# Denoiser bake-off: results

Findings from the two-phase denoiser bake-off (spec/plan:
`docs/superpowers/specs/2026-05-15-denoiser-bakeoff-llr-design.md`,
`docs/superpowers/plans/2026-05-15-denoiser-bakeoff-llr.md`). Harness:
`artifacts_scripts/gap_denoiser_bakeoff.py` over the LLR-refine
pipeline. Topic studied: **T10181** (NLP), `17/1702/T10181`, with
T11303 used for the reproducibility gate.

All numbers below are **byte-reproducible** (`--index-type hnsw` with
the corrected `_build_topic_index` params, verified by double-run; see
"Reproducibility" below). They are the **cheap geometric / dual-landmark
proxy**, *not* LLM target-gap-coverage — promotion to the paper still
needs a proxy-vs-coverage check (spec §8).

## Headline: LIP and LPP respond to row-normalization in mirror image

Phase 1, T10181, frozen graph; `sig` = #gaps with binomial
`books_below_rate_pval < 1e-3`; `leaks` = top-10 title-homogeneity
artifact flags; four-corner = (bc, bC, Bc, BC), first char book / second
chapter, lower=low eff_size, upper=high.

| config | sig | leaks | bc | bC | Bc | BC |
|---|---|---|---|---|---|---|
| **LIP + binary (self_loops=T, row_normalize=F)** | **58** | **0** | 127 | 0 | **7** | **121** |
| LIP + stochastic (row_normalize=T) | 51 | 1 | 124 | 2 | 63 | 63 |
| LPP + binary | 45 | 1 | 125 | 2 | 73 | 55 |
| **LPP + stochastic** | **55** | **0** | 129 | 0 | **9** | **121** |

Row-normalization (the "avgref" 1/N edge weighting) **hurts LIP**
(sig 58→51, an artifact leak appears, the clean `Bc≈7 / BC≈121`
structure scrambles to 63/63) and **helps LPP** (sig 45→55, leak
removed, snaps into the same clean `Bc≈9 / BC≈121` profile LIP had
under binary).

**Mechanism (denominator-explained).** Both solve a GEP `XᵀLX v = λ B
v`, smallest eigenvectors. LIP uses `B = S_T` (total scatter,
**degree-blind**); LPP uses `B = XᵀDX` (**degree-weighted**). The
locality numerator `XᵀLX` on a binary graph is dominated by the dense,
reliable citation backbone; row-normalization flattens that. LIP has no
denominator term to compensate, so it loses the backbone signal. LPP's
degree-weighted denominator already encodes degree, so it *wants* the
row-normalized numerator. **"Row-normalization hurts" is therefore
objective-specific, not universal** — a sharper, more defensible claim
than treating it as a global graph-construction fact.

**Best configuration:** LIP on the binary (self-loop, un-normalized)
graph — or equivalently LPP on the row-normalized graph. They converge
on the same high-quality regime (sig ≈ 55–58, **zero** artifact leaks,
clean four-corner).

## Cross-arm comparison (binary T,F, all 7 arms)

| arm | sig | leaks | jac_vs_none | character |
|---|---|---|---|---|
| none (384d) | 13 | 0 | 0 | large coherent subfields but ~½ not book-sparse (p≈0.3); denoising clearly helps (floor) |
| pca | 56 | 1 | 0.23 | high sig but artifact/language-cluster content — **significance without coherence** |
| **lip** | **58** | **0** | 0.24 | broad textbook-shaped subfields (coreference, GEC, NER, ontology-QA); proxy winner |
| lpp | 45 | 1 | 0.16 | sharp niches with *extreme* depletion (p down to 5e-16); LIP's complement, co-best under stochastic |
| cclip_auto | 30 | 1 | 0.154 | artifact-prone (N3C, language clusters) |
| cclip_norm_0.5 | 30 | 1 | 0.154 | **identical to cclip_auto** (see below) |
| cclip_norm_0 | 28 | 3 | 0.17 | pure anti-LDA, weakest |

Qualitative leaderboard read (top-N by `effective_size`, sampled
in-gap titles):
- **`none`/`lip`** surface large (n=160–320) recognizable NLP subfields
  — genuine textbook-gap candidates. `lip`'s are more often
  statistically book-sparse than `none`'s.
- **`lpp`** surfaces smaller (n≈80) but far more extreme-significance
  niches (LLM knowledge-injection, open-vocab segmentation, sentence
  compression, collocation extraction). Different geometry, not a
  lip-clone. Neither dominates: LIP = textbook-scale, LPP =
  monograph-scale niches.
- **`pca`/`cclip_*`** repeatedly surface the **N3C Concept Set**
  dataset-record cluster and monolingual (Korean/French) language
  clusters in the top-10. Crucially the **binomial p correctly flags
  these as non-significant** (N3C p≈0.3). The reproducibility fix did
  not change this — it was never a recall artifact; it is genuine tight
  data-debris. ⇒ **rank/filter by the binomial p (or the
  artifact-prefilter), never by raw `sig_count`/`effective_size`.**

## `cclip_auto` ≡ `cclip_norm_0.5` (scale-pathology, empirical proof)

`cclip_auto` (parameter-free trace-*equalization* on raw scatters,
α* = tr(S_B)/(tr(S_B)+tr(S_L))) produced **α\* = 3.1e-8** and a
scorecard row **byte-identical** to `cclip_norm_0.5` (pre-normalized
α=0.5): sig 30/30, leaks 1/1, four-corner 116,1,68,50 both. This is
direct evidence that raw `tr(S_L) ≫ tr(S_B)` by ~7 orders of magnitude
(the scale pathology motivating trace-normalization), and that
parameter-free auto-α degenerates — not a viable no-knob default.

## Reproducibility (resolved)

First runs showed large run-to-run variance (LIP four-corner `Bc/BC`
69/57 → 24/104). Root cause was **not** nondeterminism but
**catastrophic HNSW under-recall**: `kball_gap_probe._build_topic_index`
hardcoded `efSearch=128` while K-ball seed range-search must return
gap neighborhoods of hundreds of articles (efSearch ≪ result-set ⇒
recall collapse, *and* run-unstable). Fix: `efc=512, efs=2000` (match
`setup()`'s proven floor). At efs=2000, recall is high enough that
multi-threaded construction nondeterminism is **immaterial** — two
multi-threaded runs are byte-identical (T11303 lip top-10 Jaccard
10/10 = 1.000; T10181 `none` is byte-identical binary vs stochastic, a
graph-invariant integrity check that passes). Exact-FlatL2
(`--index-type flat`, `BOOKGAP_EXACT_INDEX`) is retained only as a
slow ground-truth escape. omp/thread-determinism machinery is not used.

## Recommendations for the paper

1. Use **LIP on the binary (self-loop, un-normalized) citation graph**.
   Report the LIP/LPP mirror-image as the *mechanistic explanation* of
   the row-normalization effect (denominator degree-awareness), not as
   a bare "binary is better" observation.
2. Treat **data-debris via the binomial depletion p or a structural
   artifact-prefilter**, not raw significance/size counts — PCA shows
   high `sig_count` is reachable by artifacts.
3. `cclip_auto`/parameter-free trace-equalization is a dead end;
   pre-normalized CCLIP underperforms plain LIP on this proxy.

## Caveats

- Single topic (T10181, NLP). The Phase-0 graph winner (self_loops=T,
  row_normalize=F) was a 3-topic majority vote; this results doc's
  Phase-1 depth is T10181 only.
- Geometric/dual-landmark proxy, not LLM coverage (spec §8 non-goal).
- `top10_artifact_leaks` is an approximate title-homogeneity screen.
- Dual-landmark stats add ~2× Phase-3 cost (backlog:
  `share-fullindex-knn-dual-landmark`).
