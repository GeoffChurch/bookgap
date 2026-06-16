# Dendrogram quality for curriculum-as-a-path — *the linkage IS the lever (average-linkage, real-metric-validated); the space/PCA fix over-promised in-space*

**Results doc** — validated → SHIPPED 2026-06-15: average-linkage is now the curriculum default
(`artifacts_scripts/curriculum/config.py`: `linkage_method='average'`, guarded to require the top-k subset).
Motivating question: the curriculum-as-a-path-through-the-dendrogram
idea is enticing (intro/conclusion as near-top clusters; UOT to match a flat cut to a book's ToC, dropping
the conclusion many books omit). Is it *salvageable with a better dendrogram*? How do we experiment with and
evaluate dendrogram quality directly, rather than only end-to-end curriculum quality?

Related: [[curriculum-diffusion-and-quality-metrics]] (the curriculum-quality metric basket — this entry is
the *clustering/tree* half of that), [[curriculum-real-toc-baseline]] (book-ToC reconstruction eval that
supplies the ground truth here). Dead-end to remember: `docs/dead_ends/curriculum/keyword-coclustering.md`
("keep clusters embedding-defined; any multi-view blend tanks cohesion") — so the "better space" lever below
should be **denoise/re-metric the embedding**, NOT fuse in citations as a view.

## The diagnosis (the reframe)

Both load-bearing curriculum evals operate on a **single flat horizontal cut** of the tree (UOT matches one
frontier to chapters; the good-edge walk runs on chapter centroids). Both are **invariant to tree shape** —
vertical placement, nesting, leaf-order all discarded before scoring. So a "better dendrogram" is literally
invisible to them, which is almost certainly why the prior verdict was "cut method is second-order under UOT."
To *optimize* the dendrogram you need a metric sensitive to tree shape. We built two and ran them.

## What was tested (2026-06-15)

Scripts under `artifacts_scripts/curriculum/dendro_eval/` (each has a provenance footer):
- `intrinsic_dendro_sweep.py` — Track 1, label-free intrinsic metrics on one ingested field.
- `toc_recovery.py` — Track 2, ToC-anchored recovery on the 4 close-loop books + the intrinsic↔extrinsic join.

Construction grid (all embedding-only, raw GIST-small 384d, default denoiser = NONE): repo default
**weighted_bisecting_kmeans** (`scalable_clustering.bisecting_kmeans_linkage`, O(N), scales to corpus),
**scipy Ward**, **scipy average-linkage (cosine)** (both O(N²), fine on per-book top-k subsets).

### Track 1 — intrinsic (NLP topic T10181; N=128,833 → 18k tree → 2,500-pt metric set; 3 seeds)
| construction | cophenetic | Dasgupta/random | silhouette@50 |
|---|---|---|---|
| scipy average (cosine) | **0.653** | 0.984 | 0.049 |
| weighted_bisecting_kmeans *(default)* | 0.275 | 0.990 | 0.010 |
| scipy Ward | 0.199 | 0.994 | 0.024 |

- **Average-linkage preserves the embedding's distance structure 2.4× better than the default** (cophenetic
  0.65 vs 0.275), robustly across seeds. The production default is **mid-pack**.
- **Dasgupta cost ≈ random for all three** (0.98–0.99× a random binary tree) — non-discriminative here, likely
  because a single field's cosines sit in a narrow band, collapsing the similarity weighting. Cophenetic is the
  **only** intrinsic metric that separates the constructions.

### Track 2 — ToC recovery (4 books; labeled point set = top-N field works by max-cosine-to-any-chapter, N = recon `n_gap_articles`)
**Scope correction (verified):** only **cyberbullying** ran at subfield 1702; **mcs / neuromorphic /
color_image ran at field 17** (their `cl_*_recon.json` have `scope_subfield: null`). Each book loaded at the
scope its own gap JSON declares.

Flat-cut **ARI@k=n_chapters** (the metric matching the curriculum's frontier cut), avg-linkage vs default:
mcs 0.616 vs 0.553, neuromorphic 0.603 vs 0.398, color_image 0.983 vs 0.927 — **average wins 3/4** (loses only
on cyberbullying, the noisiest: 0.110 vs 0.206). Cophenetic per-book: **average wins 4/4**. Dendrogram-purity
(DP, bottom-up leaf purity): **average wins 0/4** — it chains, which hurts leaf purity but aligns better once
cut to k. Since curricula cut flat → chapters, **ARI@k is the right operating point**, and average wins it.

**The join (n=12):** raw corr(cophenetic, DP) = +0.80, corr(coph, ARI) = +0.76 — but **dominated by
between-book geometry**. Within-book (demeaned): corr(coph, ARI) = **+0.20** (weak), corr(coph, DP) = **−0.55**
(anti). So intrinsic cophenetic is a good **corpus-screening** tool (which books even have recoverable
chapters) but **not a within-corpus optimization target**.

**Intro/conclusion-near-top:** directional only — flagged chapters more general in average-linkage 3/4 books,
n.s. at n=4, and it **breaks on the 24-chapter book** (generality swamped by a few large body chapters). Not a
mechanism to bet the path on.

## Verdict so far

> **⚠ In-space read — partly SUPERSEDED by the Stage 2 RESULT below.** On the REAL recon metric the actionable
> conclusion reverses: average-**linkage** is the bankable lever (and *large*, not "modest"), while "the real
> lever is a better **space**" (point 3) is **refuted** — the PCA space-fix did not survive. Only "the corpus
> sets the between-book ceiling" (point 1) stands.

1. **The dendrogram is not the wall — the embedding geometry is.** Within a book, swapping the algorithm moves
   ARI ~0.1–0.4; between books, the corpus moves it ~0.8. Clean geometry (color_image ARI 0.98) → any tree
   recovers chapters; noisy (cyberbullying 0.25) → no tree rescues it.
2. **Average-linkage is a modest, free win for the curriculum's flat-cut operating point** — adopt it for the
   per-book (top-k subset) dendrogram, where O(N²) is cheap. *Needs validation on the real close-loop metric
   first* (see caveats).
3. **Curriculum-as-path is "salvageable" only where the geometry already separates chapters.** The real lever
   is a **better space**, not a better linkage.

## Caveats (don't overclaim)
- **Circularity:** work labels are argmax-cosine to same-space chapter embeddings → absolute DP/ARI inflated.
  Cross-construction comparison and the join correlation are fair (identical points; coph on same X).
- **ARI-vs-recon-baseline lift is partly a confound:** the old `cl_*_recon.json` ARIs (0.007–0.299) label a
  *different* set (gap_article_ids) and cut the *production* tree at `n_our_clusters`, not k=n_chapters. Much of
  the apparent lift is the cleaner labeled set + matched-k, not pure algorithm. Validate properly before shipping.
- **Underpowered:** n=3 constructions, n=4 books; sign tests are descriptive.
- Single embedder, embedding-only (citation-fusion deferred — and likely a loser per the coclustering dead-end).

## Stage 1 result (2026-06-15) — geometry is NOT a hard wall; better space is a (book-dependent) lever

Script `artifacts_scripts/curriculum/dendro_eval/geometry_ceiling.py` (+ `geometry_ceiling_results.json`,
per-book labeled-set npz under `geometry_ceiling_npz/`). Reused `toc_recovery.py`'s loader + labeled-set
construction verbatim. Per book, a 3-rung space ladder {raw GIST-384, PCA-50 (unsup), LDA-oracle on gold
chapters}, fit on a stratified 70/30 train split, evaluated on the **held-out test** works (kNN-F1 k=10; unsup
average-linkage ARI@k).

| book (recon ARI) | raw→PCA50→LDA unsupARI | kNN-F1 raw→PCA→LDA | read |
|---|---|---|---|
| cyberbullying (0.079) | 0.121 → **0.368** → 0.341 | 0.50 → 0.73 → 0.64 | **LEVER** |
| mcs (0.299) | 0.714 → 0.677 → 0.711 | already separable |
| neuromorphic (0.166) | 0.415 → 0.432 → 0.262 | already sep (LDA over-compresses) |
| color_image (0.007) | 0.887 → 0.680 → 0.879 | already separable |

- **No book overlaps under supervision → geometry is not a hard wall.** On the hard book (cyberbullying) the
  ceiling breaks: raw 0.121 nearly triples to ~0.34–0.37. The chapters were linearly separable; raw 384-d
  buried the signal under ~334 noise dims.
- **The win is reachable WITHOUT labels** — PCA-50 ≈ LDA-oracle on cyberbullying. The chapter structure is in
  the dominant variance; the tail is noise. So plain PCA-denoising is the candidate lever.
- **Book-dependent:** pays off only where raw is weakest (cyberbullying); the already-separable books gain
  nothing (LDA can even hurt via ≤C−1 over-compression).

**Two caveats (unbanked):** (1) the lever rests on **n=1 hard book**; (2) the in-space probe is an *optimistic
intermediate* metric, not the real operating point — color_image scores 0.887 here but its real close-loop
recon ARI is **0.007**. Same lesson the project keeps relearning (intermediate/synthetic over-promises vs real
holdout). **The lever is not banked until it shows on the real close-loop recon metric.**

## Stage 2 (next) — validate the space lever on the REAL close-loop metric

Re-run the actual close-loop recon (proper labels = gap_article_ids; production/ matched cut) for the 4 books
under **PCA-denoised space + average-linkage** (both levers together) vs the default (raw + bisecting-km),
scoring the *real* recon ARI / coverage:
- The repo denoiser framework is mostly wired already (`Cloud.denoiser_projection()`; PCA / anti-LDA / RNOJD
  configs; cf. [[cloud-denoiser-path-port]]). The PCA-space arm should be config, not new code.
- **Add 2–3 more low-recon books** (find them via the recon ARIs / cophenetic corpus-screen) so it's not n=1.
- Decision: if cyberbullying/color_image real recon ARI moves → shippable curriculum improvement (adopt
  PCA-denoise + average-linkage for the per-book dendrogram). If the in-space win evaporates at the real
  operating point → decisive "intermediate metric over-promised again", and the path is corpus-bound (redirect:
  placement already closes; ordering's open problem is the 1-bit orientation — neither fixed by space/tree).
- Watch: average-linkage's gain is the flat-cut ARI operating point (not bottom-up DP); LDA over-compression
  warns against supervised projections in production — prefer unsupervised PCA / the repo denoisers.

## Stage 2 RESULT (2026-06-15) — TREE survives to the real recon; SPACE does not. Adopt average-linkage; drop PCA.

Ran the actual close-loop recon (`book_curriculum` → `book_toc_reconstruction`, real `adjusted_rand_index`)
for each close-loop book under **4 arms = {denoiser: NONE, PCA-50} × {linkage: weighted_bisecting_kmeans,
average}**, top-k held FIXED at 5000 for ALL arms (so only {space, tree} vary). PCA-50 = repo `WorkDenoisers.PCA`,
`drop_k=334`, fit unsupervised on the scope corpus, applied to the clustered works AND to the gold chapters
(`book_toc_reconstruction` now projects chapters through the same `cloud.denoiser_projection()` so the
chapter-assignment space matches Track-A clustering — the minimal wiring added; CLIs gained `--denoiser/--drop-k/
--linkage-method`). Scripts + results JSON under `artifacts_scripts/curriculum/dendro_eval/`
(`space_tree_recon_ab.py`, `space_tree_verdict.py`).

**Sanity gate PASSED (exact).** The internal `NONE_bisecting` arm reproduced the published recon ARI to the
digit on every book — cyberbullying 0.07948=0.07948, mcs 0.29936≈0.29936, neuromorphic 0.16625≈0.16625 — because
the published A1 runs already used `rank_before_clustering, top_k=5000, weighted_bisecting_kmeans, denoiser=NONE`
(i.e. exactly this arm; see `artifacts/curriculum/cl_A1_*/metadata.json`). So top-k did NOT move the baseline and
the PCA/average deltas are trustworthy.

Real-recon ARI (4 arms; published == NONE_bisecting):

| book (scope) | NONE_bis (=pub) | NONE_avg | PCA_bis | PCA_avg | PCAeff@bis | avgeff@NONE | joint |
|---|---|---|---|---|---|---|---|
| cyberbullying (sf 1702) | 0.0795 | **0.2890** | 0.1349 | 0.1327 | +0.055 | **+0.210** | +0.053 |
| mcs (fld 17) | 0.2994 | 0.4209 | 0.3147 | **0.6173** | +0.015 | +0.122 | **+0.318** |
| neuromorphic (fld 17) | 0.1662 | **0.3738** | 0.1744 | 0.1482 | +0.008 | **+0.208** | −0.018 |
| color_image (fld 17) | 0.0074 | 0.0148 | 0.0217 | **0.0421** | +0.014 | +0.007 | +0.035 |

**Verdict (n=4, complete):**
1. **TREE (average-linkage) SURVIVES to the real metric — it's the bankable lever.** `avg_at_none` = +0.210, +0.122,
   +0.208 → mean **+0.137, positive 4/4** (adding color_image +0.007 — floor-bound on its 24-chapter ToC, but still positive), biggest single effect, and largest exactly on the hard book the in-space
   probe flagged (cyberbullying 0.079→0.289, purity 0.38→0.66). This is NOT the color_image-style evaporation: the
   tree win is real on the real metric.
2. **SPACE (PCA-50) does NOT survive as a clean lever.** At fixed tree, `pca_at_bisecting` = +0.055, +0.015, +0.008, +0.014
   (mean **+0.023**, positive 4/4 but small). And PCA **ERODES the average-linkage win**: `pca_at_average` =
   −0.156, +0.196, −0.226, +0.027 → negative on 2/4. The best corner is usually **NONE+average**, NOT PCA+average. The lone
   exception is mcs, where PCA+average compounds (0.42→0.62) — so PCA is **book-dependent and unreliable**, the
   opposite of a shippable default. The in-space probe's "PCA nearly triples cyberbullying (0.121→0.368)" did NOT
   carry to the real recon (PCA@bisecting only +0.055; PCA@average −0.156). **The color_image worry is confirmed
   for the SPACE lever specifically.**

**Part B — hard-book set expanded.** `hard_book_screen.py` ran the cheap in-space raw-separability screen (raw
kNN-F1 / unsup-ARI) over 22 CS-textbook ToCs at field 17; the lowest-separability (cyberbullying analogues) were
boyd_convex (0.56), sutton_barto_rl (0.64), koller_pgm (0.65), szeliski_cv (0.68). Their own-gap k-ball is EMPTY at
field-17 scope (nearest field-17 book too close → gap_size 0), so the real recon was infeasible; `hard_book_4arm_proxy.py`
ran the 4-arm grid on the in-space top-N-cosine PROXY instead (clearly flagged as OPTIMISTIC). It **reproduces the
direction**: average-linkage `avg_at_none` positive **4/4** (mean +0.22); PCA `pca_at_bisecting` NEGATIVE (mean
−0.07, only 1/4 positive), `pca_at_average` negative 3/4. So the TREE lever generalizes to more hard books; the
SPACE lever does not.

**Decision (Stage-2 closes the lever question):** **adopt average-linkage** for the per-book (top-k subset)
dendrogram — it is the cheap, consistent, real-metric win, concentrated where it matters. **Do NOT adopt PCA-denoise**
as a default — it's neutral-to-negative on the real recon, erodes the average-linkage gain on 2/4 books, and only
helps one (mcs). This is the project's recurring lesson again: the in-space/intermediate probe over-promised the
SPACE lever (cyberbullying 0.121→0.368 in-space vs PCA@bisecting +0.055 real; color_image was 0.887 in-space vs
0.007 real), while the TREE lever — which the cophenetic intrinsic metric flagged as a *corpus-screening* (not
optimization) signal — held up. Salvage for PCA: it's a per-book wildcard (big on mcs), not a default; revisit only
if a book-specific selector emerges.

**Placement ≠ ordering caveat (2026-06-16).** This adoption is a **placement** win (recon ARI — which works
land in which chapter). It does NOT transfer to **ordering**: an end-to-end A/B on the 4 books
(`intro_orientation_avglink.py`) found average-linkage *lowers* good-edge chain quality vs bisecting (mean
undirected |τ| 0.479→0.395, worse 3/4) and leaves the 1-bit orientation at 2/4 — average-linkage's chaining
spins off straggly tail clusters whose noisier centroids make a messier topical-adjacency chain (same shape as
the ARI-beats-DP split). So: average-linkage for **content/placement** (the default; primary close-loop metric);
**bisecting gives cleaner ordering chains** (n=4, suggestive). The orientation bottleneck is *not* fixed by the
tree. → `docs/curriculum-close-loop.md` (Exp B′).

## Provenance
Regenerate (don't hand-edit the JSON):
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.intrinsic_dendro_sweep`
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.toc_recovery`
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.space_tree_recon_ab`  (Stage 2: real 4-arm recon)
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.hard_book_screen`  (Part B: hard-book screen)
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.hard_book_4arm_proxy`  (Part B: hard-book 4-arm proxy)
- `.conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.space_tree_verdict`  (consolidated verdict)
Results: `artifacts_scripts/curriculum/dendro_eval/{intrinsic_dendro_sweep,toc_recovery,space_tree_recon_ab,hard_book_screen,hard_book_4arm_proxy,space_tree_verdict}_results.json` (+ `space_tree_verdict.json`).
