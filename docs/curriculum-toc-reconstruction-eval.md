# Curriculum ToC-Reconstruction Eval — how to run it

## What this is (and why)

We generate a curriculum for a real book's **own gap** and ask an LLM judge to compare it,
head-to-head, against the curriculum implied by the book's **real table of contents**. The
largest *genuine-book* own-gaps are the most underserved real regions that a human nonetheless
wrote a book to fill, so "does our curriculum reconstruct that book's ToC?" is a direct proxy
for **"does our system serve the highest-value underserved gaps as well as a domain expert
did?"** — i.e. it's the eval for the thing we actually care about.

Two artifacts per book:
- **Track A** — our generated curriculum, seeded from the book's own gap.
- **Track B** — the book's real ToC, materialized into our curriculum format (each ToC chapter
  populated with its nearest gap articles + phrase-mined keywords).

Both are rendered blind (chapters labeled by keywords, not the book's titles) and judged
pairwise (A/B/tie per dimension, both orders, by gemma4:31b and an opus subagent).

Worked examples live in [`curriculum_eval_examples/`](curriculum_eval_examples/) — three
markdown files per book (ours / original book / ToC-derived). Full findings + design rationale
are in [`backlog/curriculum-real-toc-baseline.md`](backlog/curriculum-real-toc-baseline.md).

## The pipeline (per book)

All commands run from the repo root with `export PYTHONPATH=/home/gchurchill/src/bookgap` and the
direct interpreter `/home/gchurchill/src/bookgap/.conda/bin/python` (NOT `conda run`).

1. **Pick a target.** Targets must be (a) clean OA *book* records, (b) with a real own-gap, and
   (c) a recoverable ToC. Canonical/comprehensive textbooks (AIMA, Jurafsky, Koehn) have **empty**
   own-gaps (book-dense regions) and are structurally excluded — only niche, focused books work.
   - Rank every book's own-gap once per scope: `book_own_gap.py --rank-top` (writes
     `*_ranked.json` incl. `all_book_gaps` for any-depth curation later).
   - Filter to genuine candidates: `curate_book_candidates.py` (drops debris/theses/non-English;
     the raw top of the ranking is debris-dominated). ToC-check survivors via **Google Books**
     (publisher pages 403 automated fetches).

2. **The book's gap (exact, fast):**
   ```
   python -m artifacts_scripts.curriculum.book_own_gap --brute-force \
       --book-id https://openalex.org/W4235013513 --field 17 \
       --out artifacts/book_gaps/mcs_gap.json
   ```
   Brute-force (nearest non-self book + article ball via dot-products) — no HNSW, ~minutes at
   any scope. (`--rank-top` builds an HNSW and is only for the all-books ranking above.)

3. **Track A — our curriculum** (match our chapter count to the book's ToC):
   ```
   python -m artifacts_scripts.curriculum.book_curriculum --gap-json artifacts/book_gaps/mcs_gap.json \
       --boundary --desired-n-topics 10 --output-dir mcs_trackA
   ```

4. **Track B — materialize the book's ToC** (needs a ToC json: `{title, book_id, chapters:[{n,title,description}]}`):
   ```
   python -m artifacts_scripts.curriculum.book_toc_reconstruction \
       --gap-json artifacts/book_gaps/mcs_gap.json --toc-json artifacts/book_gaps/W4235013513_toc.json \
       --curriculum-dir artifacts/curriculum/mcs_trackA --mean-center \
       --materialize-dir artifacts/curriculum/mcs_trackB --out artifacts/book_gaps/mcs_recon.json
   ```
   Always use `--mean-center` (raw assignment leaves empty chapters); keywords are phrase-mined
   to match Track A.

5. **Judge (gemma):**
   ```
   python -m artifacts_scripts.curriculum.eval_pairwise_judge \
       --run-dir-a artifacts/curriculum/mcs_trackA --run-dir-b artifacts/curriculum/mcs_trackB \
       --label-a ours --label-b book_toc --model gemma4:31b --repeats 3 --n-titles 0 \
       --out artifacts/curriculum_eval_pairwise/mcs_pairwise.json
   ```
   `--n-titles 0` = keywords-only (verdict is robust to dropping the noisy sample articles).
   For the opus side, render both outlines (`eval_book_judge.render_outline(run_dir, n_titles=0)`)
   to neutral files and dispatch an opus subagent with the same pairwise prompt; de-randomize by
   the known alpha=ours / beta=ToC mapping.

## Run a whole panel

`run_toc_panel.sh` does steps 2–5 for a list of books, parallel-where-safe: it runs the first
book to completion (warming + caching the per-scope citation graph), then fans out the rest
memory-capped (~15 GB/book at CS-field → `MAX_PARALLEL=8`; set 2 for whole-domain), and
serializes the gemma evals on the single GPU. It auto-matches `--desired-n-topics` to each ToC.
Opus is dispatched separately. Edit the `PANEL` array, then:
```
MAX_PARALLEL=8 SEL=--boundary bash artifacts_scripts/curriculum/run_toc_panel.sh
```

## Export the readable markdown (the case + its verdict)

```
python -m artifacts_scripts.curriculum.export_curriculum_markdown \
    --track-a artifacts/curriculum/mcs_trackA_sem --track-b artifacts/curriculum/mcs_trackB \
    --toc-json artifacts/book_gaps/W4235013513_toc.json \
    --eval-json artifacts/curriculum_eval_pairwise/mcs_sem_dk_pairwise.json \
    --out-dir docs/curriculum_eval_examples/mobile_crowdsensing
```
Writes `ours_curriculum.md` (Track A), `original_book.md` (title + OpenAlex abstract + ToC *with
chapter descriptions*), `toc_derived_curriculum.md` (Track B), and — with `--eval-json` — the
pairwise verdict as `eval_results.md`. Tracks A/B are read from their run-dirs via the same
`load_book_outline` the judge uses, so the markdown is exactly what was evaluated; keyword phrases
are backticked so phrase-mined terms (`neural_networks`) read as units. Every file ends with a
provenance footer recording the exact command that produced it. `--only-ours` regenerates just the
Track A file (e.g. after improving the curriculum) without touching the book/Track B files. Worked
examples (current best config: semantic re-rank + dynamic-k) live under
[`curriculum_eval_examples/`](curriculum_eval_examples/).

## What we've found

The book's real ToC beats our generated curriculum, and it generalizes (3-book panel, matched
chapters: gemma 12/12, opus 9/12). The dominant, **universal** defect is **specificity** — our
curriculum wanders into off-topic *foundations* (every book loses specificity, both judges).
A secondary, book-specific issue is **debris chapters** (book-metadata boilerplate clusters,
e.g. Color Image's "series, departments, covers"). Ours wins only sporadically: *progressive*
when its prerequisites are coherent, *comprehensiveness* when its breadth exceeds the book's
narrow gap-coverage. See the backlog for the full table and the specificity-improvement ideas.

## Quantitative chapter correspondence (matrix eval) — complements the LLM judge

The pairwise LLM judge gives a blunt per-axis winner; `gap_chapter_correspondence.py` adds a
**quantitative, knob-light, content-based** view. It builds the chapter×chapter probe **matrix**
between two curricula (each chapter = a weighted set of works), under probes spanning two axes:
**kernel** (work-identity → embedding-geometry: `ruzicka` weighted-Jaccard, `centroid_cos`,
`cover_A` mean-max-cosine) and **direction** (symmetric vs `containment_A` directed). The matrix is
the primary object (row/col maxima = greedy alignment; off-diagonal = splits/merges; the **A-vs-A**
matrix = internal redundancy, the FL/DPP diagnostic). On top: **Hungarian 1-1 alignment**,
**work-overlap coverage** (best-match ruzicka, size-weighted), and internal redundancy. (Per-book
detail in each `<book>/chapter_correspondence.md`. A soft/partial UOT reduction is deferred — the
matrix + Hungarian + greedy coverage are the reliable core; the kernel axis is a resolution knob:
`centroid_cos` saturates in single-topic corpora, `ruzicka`/`containment` discriminate.)

**Panel finding (the reframe):** across all three books our curriculum is **topically a near-complete
cover of the book** (Hungarian alignment `centroid_cos` ≈ 0.97) but shares only **9–22% of the same
papers** (best-match ruzicka: MCS 0.22, Neuromorphic 0.19, Color Image 0.09 — lowest where the book
is a comprehensive 357-reference textbook). So the LLM "book wins all axes" is consistently about
**framing and specific-work choice, not topical coverage** — a more precise and favorable read than
the verdict alone. (All three also have internal redundancy ≈ 0.99: chapters are mutually similar
because each gap is a tight single topic.)
