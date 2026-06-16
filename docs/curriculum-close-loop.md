# Closing the loop: does gap-targeting + ordering reconstruct a real book's ToC?

The end-to-end test of the curriculum pipeline. Stages 1→3 say: detect a gap, select
articles there, order them into chapters. This asks whether running that loop at a real
book's *own gap* reproduces the book a human actually wrote — selection (does the
curriculum land on and partition the book's material?) and ordering (does it recover the
chapter sequence?) — with **no LLM judge**.

## Setup

Three arms, each a curriculum built by the same pipeline (`book_curriculum.py`,
`desired_n_topics` = the book's chapter count), differing only in the seed region:

- **A0** — seed a **count-matched random region** (`random_region_gap.py`: the N nearest
  articles to a uniform-random center, N = |own-gap|). A density/size floor.
- **A1** — seed the book's **detected own-gap** (`book_own_gap.py`: articles within
  0.75·dist-to-nearest-other-book of the book).
- **A2** — A1's chapters **re-ordered by good edges** (greedy topical-adjacency chain over
  cluster centroids). Same works as A1, so placement metrics are identical; only order moves.

Ground truth = the book's real ToC, embedded and populated with the gap articles
(`book_toc_reconstruction.py` Track B). Co-primaries (`close_loop_metrics.py`):

1. **Placement** — size-weighted best-match **ruzicka work-overlap coverage** of the book's
   chapters by ours (order-invariant). Discriminates A1 vs A0.
2. **Ordering** — **forward citation-flow prereq-violation share** (`toc_metrics`,
   `G[i,j]`=i cites j ⇒ b is a's prereq ⇒ a-before-b is a violation), A1's clusters under
   frontier vs good-edge order. Plus **Kendall-τ to the human chapter order** (shared matching).

Panel: 4 niche books with real own-gaps — Cyberbullying (subfield 1702), Mobile-Crowdsensing
(MCS), Color-Image Processing, Neuromorphic Photonics (field 17).

## Panel result

```
book             bk_ch  cl |  covA1  covA0   Δcov | purity   ARI |  pvA1  pvA2   pvR |   τA1    τA2
color_image          9  24 |  0.069  0.000  0.069 | 0.165 0.007 | 0.466 0.700 0.500 | 0.000 -0.231
cyberbullying        7   8 |  0.160  0.000  0.160 | 0.385 0.079 | 0.560 0.692 0.498 | 0.159  0.476
mcs                  9  10 |  0.229  0.000  0.229 | 0.633 0.299 | 0.319 0.533 0.499 | 0.000  0.617
neuromorphic         9  13 |  0.175  0.000  0.175 | 0.458 0.166 | 0.295 0.634 0.498 | 0.296 -0.593
pooled: placement A1-A0 +0.158 (4/4 +); prereq A1-A2 -0.230 (0/4); τ A2-A1 -0.047 (2/4)
```

**1. Placement — validated, unanimous (4/4).** A1−A0 coverage +0.158 mean, and A0 is a
literal **0.000 in every book** — random regions recover *none* of the book's works,
gap-targeting recovers 7–23%. The first half of the loop closes: the detected gap lands the
curriculum on the book's material, and the floor is a true zero (full dynamic range).

**2. Reconstruction fidelity — partial, and bounded by the gap's locality.** Purity 0.17–0.63,
ARI 0.01–0.30. Every book's Track B populated only ~9 chapters regardless of ToC width
(Color-Image: 9 of 24) — **the own-gap is one neighborhood, so it reconstructs a ~9-chapter
core, not the whole book.** MCS reconstructs well (purity 0.63); the broad 24-chapter
Color-Image essentially fails (0.17). Work-identity ruzicka is also a strict bar (same OpenAlex
works). So the loop lands in the right region but does not sharply reproduce the exact partition,
and degrades as the book outgrows its gap.

**3. Ordering — the sequence is recovered; the direction is not.** This is the subtle part.

- **Prereq-violation: good-edge is unanimously worse (0/4).** A1−A2 share −0.230 — good-edge
  ordering reliably *raises* citation-prereq violations (frontier sits at/below random ~0.50;
  good-edge well above).
- **τ-to-human looked mixed (2/4)** — but the orientation diagnostic
  (`close_loop_order_diag.py`) explains it. Good-edge's `consec-cos` exceeds frontier's in all
  4 books (it *is* the smoother topical chain), and **|good-edge τ| beats frontier in all 4**:

  ```
  book           frontier τ   good-edge τ   reversed τ   |undirected|
  cyberbullying     0.159        +0.476       -0.476         0.476
  mcs               0.000        +0.617       -0.617         0.617
  color_image       0.000        -0.231       +0.231         0.231
  neuromorphic      0.296        -0.593       +0.593         0.593
  ```

  So **good-edge recovers the human chapter sequence as an *undirected* chain** (|τ| 0.23–0.62,
  every one > frontier). The only failure is a **1-bit orientation**: which end is Chapter 1.
  And that bit is **not recoverable from citation structure** — both candidate anchors
  (agree-with-frontier-order, and minimize-prereq-violations) flip *all four* books, yet the
  human orientation agrees with the citation direction in only 2/4. The reason is finding (3)'s
  echo: **for Cyberbullying and MCS the human order is the *higher*-prereq-violation direction —
  the author orders chapters *against* the citation DAG.** Human textbook ordering is not a
  topological sort of the citation graph.

## Verdict

- **Selection/placement half of the loop: closes.** Gap-targeting puts the curriculum on the
  book's material (A0 ≡ 0); reconstruction of a ~9-chapter core is real but partial.
- **Ordering half: the sequence closes, the direction is open.** Good-edge recovers the human
  chapter chain up to reversal (beats the citation-frontier order 4/4); orienting it needs a
  *non-citation* anchor (publication-year foundational→recent, or general→specific), because the
  human's choice of starting end is not prereq-consistent. The general→specific anchor is now
  tested (next section): **the raw "intro is more general than the conclusion" bit is real and
  significant (≈77–79% over 29 ToCs, p<0.01), but it does not reliably orient the good-edge
  chain** — partial, not a clean close.

## Ordering — the general→specific anchor (orientation, no year/size)

Tested whether a **generality/centrality anchor**, from chapter/cluster *embeddings only* (NO
publication year, NO cluster size), supplies the missing 1-bit orientation: orient the
good-edge chain general→specific (Chapter 1 = most general). Two embedding-only generality
measures: **eigen-centrality** (|principal eigenvector| of the chapter-cosine affinity S) and
**transport-entropy** (row-normalize S→P, Hᵢ = −Σ Pᵢⱼ log Pᵢⱼ — a UOT proxy). Script:
`artifacts_scripts/curriculum/intro_orientation.py`; results `artifacts/book_gaps/intro_orientation.json`.

The pre-registered risk — *conclusions are also general (summary) and last*, so generality
could be U-shaped over chapter order and fail to orient — is measured, not assumed.

**Experiment A (n=29 harvested ToCs, k≥5 — the well-powered arm).**
- *Shape on the true human order.* The conclusion confound is **real but minority**, not
  dominant: for eigen-centrality only **3/29 ToCs are U-shaped** (10 monotone-decreasing,
  14 flat, 2 increasing); the last chapter sits in the top generality quartile only **10%** of
  the time, vs Chapter 1 at **62%**. Mean Spearman(chapter#, generality) = −0.19 (weakly
  decreasing). Transport-entropy is flatter/noisier (22/29 flat; last-in-top-quartile 38% — it
  picks up the conclusion more).
- *The pure 1-bit test (chain-free): is Chapter 1 more general than the LAST chapter?* **Yes,
  significantly** — eigen **22/29 = 76%** (binom p=0.008), entropy **23/29 = 79%** (p=0.002).
  So the *signal* exists: intros really are more general than conclusions, the U-shape does not
  erase it.
- *But orienting the good-edge CHAIN is only partial.* Pooled, the gradient/endpoint rules hit
  **18/29 = 62%** (eigen, n.s. p=0.27) and 52% (entropy). The dilution is a **chain-quality
  confound**: when the undirected chain is itself noise (|τ|<0.10, 6 ToCs) orientation is a coin
  flip *by construction* (1/6, dir-τ≈0). Stratifying on chain quality is decisive:

  ```
  eigen, gradient rule        n   correct   mean directed-τ
  chain good   |τ|≥0.25        9    8/9=89%      +0.272
  chain mid    0.10–0.25      14    9/14=64%     +0.058
  chain noise  |τ|<0.10        6    1/6=17%      −0.021
  ```
  So **in the regime where the chain actually recovered the sequence, the eigen anchor orients
  it correctly ~89%** (directed-τ goes solidly positive). Eigen-centrality clearly beats
  transport-entropy (66% vs 67% on good chains, but 80% vs 45% conditioned on a clean monotone
  shape; entropy's interior is too noisy for the gradient). Gradient and endpoint rules tie for
  eigen because the good-edge chain already *seeds at the broadest node*, which is ~the
  eigen-most-central end — the anchor rarely overrides the chain's built-in start.

**Experiment B (4 close-loop books, our generated clusters; tree = `weighted_bisecting_kmeans`,
the on-disk cl_A1_* runs — NOT the new average-linkage default).**

  ```
  book           undir |τ|   eigen-orient τ   entropy-orient τ   random
  cyberbullying     0.476       +0.476            −0.476          ~0
  mcs               0.617       +0.617            −0.617          ~0
  neuromorphic      0.593       −0.593            −0.593          ~0
  color_image       0.231       −0.231            +0.231          ~0
  ```
  Eigen recovers the human direction on 2/4 (cyberbullying, mcs) and fails on 2/4 (neuromorphic,
  color_image) — it hits the undirected ceiling when right and the floor when wrong (the chain is
  near-perfectly directed up to a sign, so orientation is all-or-nothing here). On neuromorphic
  the failure is genuine, not a bug: both measures rank the chain's broad-node start (cluster 11)
  marginally above the other end (eigen 0.287 vs 0.260), but the *human* Chapter 1 maps to the
  opposite end — **this author ordered against the generality gradient**, the same way they
  ordered against the citation DAG.

**Experiment B′ — does the average-linkage default lift chain quality / orientation? NO (null).**
Experiment B reused the OLD `weighted_bisecting_kmeans` A1 clusters. Since average-linkage is now
the dendrogram default (+0.137 ARI over bisecting on the recon metric), we regenerated all four A1
curricula with `linkage_method='average'` (same gap, same `ranking_top_k=5000`, same chapter
count) and re-ran the *identical* generality+orientation pipeline — Track B (`cl_B_*`, the human
ToC populated with gap articles) is linkage-independent and shared, so only A1 changes. Script:
`artifacts_scripts/curriculum/intro_orientation_avglink.py`; results
`artifacts/book_gaps/intro_orientation_avglink.json` (the bisecting arm reproduces Experiment B's
numbers exactly, confirming an apples-to-apples pipeline).

  ```
  book           bisecting           average             Δ|τ|    orient (eigen+grad)
                 |τ|   dir-τ  orient  |τ|   dir-τ  orient          bis → avg
  cyberbullying  0.476 +0.476  OK     0.316 +0.316  OK     −0.160  OK → OK
  mcs            0.617 +0.617  OK     0.370 −0.370  ✗      −0.247  OK → ✗  (FLIP)
  neuromorphic   0.593 −0.593  ✗      0.394 −0.394  ✗      −0.199  ✗ → ✗
  color_image    0.231 −0.231  ✗      0.501 +0.501  OK     +0.270  ✗ → OK  (FLIP)
  pooled: mean |τ| 0.479 → 0.395 (Δ −0.084); |τ| DOWN in 3/4 books; orient-correct 2/4 → 2/4
  ```

- **Chain quality does NOT improve — it drops.** Mean undirected |τ| falls 0.479 → 0.395
  (Δ −0.084), and average-linkage *lowers* |τ| in **3 of 4 books** (only the broad 24-chapter
  Color-Image rises, 0.231 → 0.501). The "#books past |τ|≥0.25" headline nominally goes 3/4 → 4/4,
  but that is an artifact: 3 of the 4 books moved *down* in |τ| and merely stayed above the line;
  Color-Image is the lone genuine crossing. (`n_pairs`, the count of human chapters that overlap
  any A1 cluster, also drops 1–2 per book under average-linkage — its singleton/tail clusters
  cover the human chapters slightly less, e.g. cyberbullying 7 → 5 pairs.)
- **Orientation does NOT improve — it stays 2/4, by a swap.** The eigen+gradient anchor is correct
  on 2/4 in *both* arms; the composition just shuffles (mcs flips correct→wrong as its chain
  degrades 0.617 → 0.370; Color-Image flips wrong→correct as its chain sharpens 0.231 → 0.501).
  Directed-τ positive count is likewise 2/4 → 2/4. The two genuinely reverse-ordered authors
  (neuromorphic, and now also the degraded mcs) remain mis-oriented for the same reason as before
  — they sequence against the generality gradient.

So the cleaner tree does **not** push orientation into the conditional-89% regime here: it does not
raise chain quality (it lowers it on average), so the orientation bottleneck is **unmoved**. The
ARI gain that makes average-linkage the recon default does not translate into a sharper *good-edge
chain to the human chapter sequence* on this n=4 panel.

**Verdict (orientation).** A generality anchor recovers the 1-bit orientation **partially, not
cleanly**. The underlying signal is real and significant (intro > conclusion in generality,
≈77–79%, embeddings only, no year/size), and **eigen-centrality** (not transport-entropy) is the
right measure; the **gradient ≡ endpoint** rule is moot because the good-edge seed already
encodes it. *Conditional on a good undirected chain it reaches ~89%/the undirected ceiling* — but
unconditionally it is ~62% (n.s.), because (a) when the chain is noise there is nothing to orient,
and (b) a real minority of authors order general→specific in reverse (against generality, as they
do against citations). **We cannot yet emit a reliably fully-directed ToC**; we can orient
*correctly most of the time when the sequence itself was well-recovered*. Orientation is narrowed,
not closed. The conclusion-confound (U-shape) was the stated risk and is **not** the main
blocker — chain quality and genuine reverse-ordered authors are. **And switching to the
average-linkage dendrogram default does not relieve either** (Experiment B′): it lowers undirected
|τ| in 3/4 books and leaves orientation at 2/4, so the cleaner recon-tree does not buy us a
fully-directed ToC on this panel.

## Caveats

n=4 niche books; work-identity coverage is strict; the gap is a local neighborhood (caps
reconstruction at the core); the reference book is not yet held out of the cloud (minor leakage,
flagged in `book_curriculum.py`). The eval is the weak link, not the pipeline.

## Reproduce

```
bash artifacts_scripts/curriculum/run_close_loop.sh run_book <slug> <gap.json> <toc.json> <n_chap>
bash artifacts_scripts/curriculum/run_close_loop_field17.sh          # the 3 field-17 books, parallel
python -m artifacts_scripts.curriculum.close_loop_summary            # panel table + sign test
python -m artifacts_scripts.curriculum.close_loop_order_diag --gap-json ... --toc-json ... \
    --run-a1 ... --run-b ...                                          # orientation diagnostic
CUDA_VISIBLE_DEVICES="" python -m artifacts_scripts.curriculum.intro_orientation  # general→specific anchor (Exp A+B)
# Average-linkage A/B (Exp B′): regenerate the 4 A1 curricula with the default tree, then re-run
# the orientation pipeline. Track B (cl_B_*) is linkage-independent and reused; only A1 changes.
for s in cyberbullying:W2898513995_cyberbullying_ai:8 mcs:mcs_gap:10 \
         neuromorphic:neuromorphic_gap:13 color_image:color_image_gap:24; do
  IFS=: read slug gap n <<<"$s"
  CUDA_VISIBLE_DEVICES="" python -m artifacts_scripts.curriculum.book_curriculum \
      --gap-json artifacts/book_gaps/${gap}.json --desired-n-topics $n \
      --ranking-top-k 5000 --output-dir cl_A1_avg_${slug} --no-plot   # default linkage = average
done
CUDA_VISIBLE_DEVICES="" python -m artifacts_scripts.curriculum.intro_orientation_avglink  # A/B table + verdict
```
Artifacts: `artifacts/book_gaps/cl_<slug>_metrics.json`, `cl_<slug>_recon.json`, `cl_summary.json`,
`intro_orientation_avglink.json`.
Connects to [curriculum-toc-dendrogram-matching](curriculum-toc-dendrogram-matching.md) (good-edge
ordering, genre signal) and the book-surprise gap definition (memory `gap-book-surprise-definition`).
