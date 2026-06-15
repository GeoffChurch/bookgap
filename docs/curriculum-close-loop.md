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
  human's choice of starting end is not prereq-consistent. Untested — the clear next step.

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
```
Artifacts: `artifacts/book_gaps/cl_<slug>_metrics.json`, `cl_<slug>_recon.json`, `cl_summary.json`.
Connects to [curriculum-toc-dendrogram-matching](curriculum-toc-dendrogram-matching.md) (good-edge
ordering, genre signal) and the book-surprise gap definition (memory `gap-book-surprise-definition`).
