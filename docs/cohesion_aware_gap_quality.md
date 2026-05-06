# Cohesion-aware gap quality: filters and confluence

Design for three additions on top of the cohesion-aware article-as-seed
pipeline (`artifacts_scripts/gap_article_as_seed.py`). Builds on
`docs/cohesion_design_axes.md` (which establishes the per-gap quality
metric) and `docs/gap_method_comparison.md` (which surfaces the failure
modes this work targets: textbook-exists false-gaps and
templated-data leaks in the top-20).

The three phases are independent enough to land and reassess one at a
time, but designed to compose: Phase 2 makes Phase 3's exact-match
dedup reliable; Phase 1 stays useful as a heuristic cross-check on
Phase 3's structural answer.

---

## Phase 1: Vote-concentration / book-existence filter

Catch gaps that already have a textbook, and templated-data leaks
where votes pile on a topically-random book.

**New script:** `artifacts_scripts/gap_book_existence_filter.py`.
Post-processes any gap CSV (article-as-seed *or* K-ball) and writes an
enriched CSV with the columns below.

**Columns added:**

| column | description |
|--------|-------------|
| `top_book_id`, `top_book_title` | most-voted boundary book (1-NN-book vote across gap articles) |
| `top_book_votes` | vote count |
| `vote_concentration` | `top_book_votes / |G|` |
| `top_keywords` | top-5 keywords for the gap |
| `title_keyword_overlap` | Jaccard between (tokenized lowercased top-book title) and `top_keywords` |
| `book_exists_flag` | bool — `vote_concentration ≥ 0.25 AND title_keyword_overlap ≥ 0.2` |
| `templated_leak_flag` | bool — `vote_concentration ≥ 0.5 AND title_keyword_overlap == 0` |

Thresholds are CLI flags. The defaults come from the existing top-20
on T10181 and will be tuned after the first run.

**Shared helper:** the keyword-extraction code currently inline in
`gap_method_comparison_table.py` (log-odds-ratio with iterative bigram
phrase mining) moves to `artifacts_scripts/_keyword_helpers.py` so
both scripts call the same code.

**Title-overlap heuristic:** Jaccard on lowercased token sets, split on
non-word chars, drop stopwords. Cheapest-thing-that-could-work; no LLM
call. The flag is a soft signal we'll review by hand on the comparison
table, not a hard filter applied silently.

**Diagnostic output (printed):**
- Total gaps in / # `book_exists` / # `templated_leak` / # clean
- Top-20 by `logit_sum` before-and-after applying the flags as a filter

**Validation criterion (T10181):** of the top-20 cohesion-aware gaps,
how many of the visibly-textbook-covered ones (Dependency Parsing rank
1, Chinese Spelling Check rank 4, Korean MT rank 5, Question Answering
rank 10, Wikipedia rank 11, Grapheme-Phoneme rank 12, Chinese Word
Segmentation rank 16) are tagged. Goal: most of these. Templated leaks
(N3C rank 13) should be tagged via the second flag.

---

## Phase 2: Symmetric add-batch + remove-batch fixed-point loop

Replace the strict single-add inner loop in `expand_seed` with a
symmetric add+remove iteration that converges to a fixed point. This
subsumes both the proposed batch-addition variant and the
canonicalization backlog item: a basin's fixed point is invariant to
which seed inside the basin starts the expansion, which makes
exact-match dedup reliable.

**Inner loop (replaces the current strict-single body in
`gap_article_as_seed.py:expand_seed`):**

```
loop:
    candidates = find_frontier(G, top=W)        # W = 50, was 1
    to_add = {c for c in candidates
              if compute_state(G ∪ {c})['min_post'] >= τ}
    G_new = G ∪ to_add
    state = compute_state(G_new)
    to_remove = {i for i in G_new if state.post[i] < τ}
    G_new = G_new \ to_remove
    if G_new == G: stop                         # fixed point
    if iter > max_iters: stop                   # didn't stabilize
    G = G_new
```

**Flag:** `--batch-addition` (default off; preserves current
single-add strict-moving-centroid behavior). The flag enables the full
symmetric loop above — both the batch-add *and* the batch-remove are
inside the same flag, since they're parts of the same fixed-point
iteration. New: `--batch-frontier-width` (default 50; chosen to be
much wider than 1 to give the fixed point room to grow per iteration,
but not so wide that we evaluate hundreds of unlikely candidates each
step. Tunable based on observed iteration counts).

**Singleton-init handling:** the existing article-seed loop already
handles the |G| < 2 startup (force-add nearest neighbor on iter 1
unconditionally because `compute_state` requires |G| ≥ 2). Same
handling carries over to the symmetric loop without modification.

**Why symmetric:**
- Removal is essentially free — `compute_state` already returns the
  per-member posterior array we need.
- Stronger seed-invariance than asymmetric batch-add (which lets early
  members linger even after G drifts).
- Self-healing from over-inclusion: if adding the batch drops some
  members below τ, those get removed and we keep going rather than
  stopping.
- Subsumes the canonicalization backlog item
  (`docs/backlog/gap-detection-methods/article-as-seed-canonicalization.md`,
  which gets deleted with this work).

**Failure modes to watch:**
- Diffuse regions can oscillate or shrink to nothing. Treat
  "didn't-stabilize within `max_iters`" as "not a basin" — discard.
- A simpler "ordered, stop-when-collective-would-drop" variant exists
  if symmetric over-includes empirically. Same flag, different inner-
  loop body. Don't implement preemptively.

**Diagnostics added (printed at end of run):**
- **Confluence rate** (with `--disable-fast-filters`): `len(distinct
  frozensets) / len(seeds processed)`. The load-bearing number.
- **Jaccard-collapse ratio**: `len(distinct frozensets pre-Jaccard) /
  len(post-Jaccard survivors)`. Lower means less path-dependence.
- **Per-seed iteration distribution**: mean / p50 / p99. Should drop
  sharply with batch-add.
- **Fixed-point convergence rate**: fraction of seeds reaching the
  fixed point vs. hitting the iteration cap.

**Validation criterion (T11303):** with `--disable-fast-filters
--batch-addition`, distinct frozenset count drops meaningfully from
the 789 baseline toward 254 (the post-Jaccard survivor count). That's
the go/no-go for Phase 3.

---

## Phase 3: Books treated symmetrically as seeds and members

Drop the asymmetry that makes books "seed-only." Books and articles
participate identically in the symmetric expansion: a book seed inits
G = {book}, expands the same way, and books may end up as members of G
alongside articles. A cluster is **book-anchored** iff its fixed point
contains any book — the cleanest possible structural notion of "this
topic has a textbook."

**Flag:** `--with-book-seeds`. When set, runs book seeds before
article seeds (book-anchored regions claimed early; article seeds
hitting them early-abort cheaply).

**Mechanics:**
- For each book `b`: init G = {b}. First iteration force-adds `b`'s
  nearest non-G work (any type) so |G| ≥ 2 and `compute_state` works;
  thereafter the symmetric loop runs as in Phase 2.
- `find_frontier` drops its article-only filter — frontier is
  centroid-K-NN minus G, full stop.
- **Boundary-book lookup** (used by `compute_state` for the local-bg
  construction): for each *article* member `i ∈ G`, take `i`'s 1-NN
  book *outside G*. Books inside G are interior, not boundary, so
  they don't appear in the boundary set.
- `gap_hashes: dict[frozenset, dict]` (upgraded from `set[frozenset]`):
  records per-cluster metadata including member type counts.

**CSV schema changes:** new columns `n_articles`, `n_books`,
`total_size`, `book_anchored = (n_books > 0)`, and
`member_indices_json` (full members, tagged `[{"idx": int, "type":
"article"|"book"}, ...]`). Keep the existing `final_size` and
`article_indices_json` columns populated for backward-compat —
`final_size` mirrors `total_size`; `article_indices_json` contains
just the article subset of members. Existing readers
(`gap_books_inspector.py`, etc.) keep working unchanged; readers that
need book membership use the new `member_indices_json`.

**Edge cases:**
- Outlier book with no real article-territory → symmetric expansion
  shrinks to a tiny cluster, filtered by `min_final_size`. No
  false-anchoring.
- Book on the boundary of an article-centered basin → fixed point is
  the same regardless of which boundary point seeded; the book ends
  up in G and tags the basin. Correct: a book sitting on the edge of
  a tight cohesive region is covering the topic.

**Filter rule:** `book_anchored = (n_books > 0)`. Start here. Three
escape hatches if it over-flags: threshold the count (`n_books ≥ 2`),
threshold the fraction (`n_books / total_size ≥ ε`), or trust the
cohesion threshold and leave alone. Decide empirically.

**Diagnostics:**
- # book seeds processed / # producing distinct fixed points / #
  collapsing into article-discovered clusters
- Top-20 by `logit_sum` split into book-anchored vs. article-only
- **Phase-1 ↔ Phase-3 agreement matrix** (4-cell): `book_exists_flag`
  × `book_anchored`. High agreement = signals corroborate.
  `book_anchored ∧ ¬book_exists_flag` = Phase 3 catches what the
  heuristic missed (good); `¬book_anchored ∧ book_exists_flag` =
  heuristic false-positive *or* confluence misfire (investigate).

**Validation criterion (T10181):** the K-ball top-20 false-gaps
identified above (Dependency Parsing, Reordering, Morphology, MT
Evaluation, WSD, Chinese Word Segmentation, ...) appear book-anchored.
Templated-data leaks (N3C) do *not* — those have no real anchoring
book, only a spuriously-voted one.

---

## Comparison-table changes

`artifacts_scripts/gap_method_comparison_table.py` consumes the
enriched CSVs and renders the new signal:

- **Reads enriched CSVs** (post-Phase-1 + post-Phase-3 columns).
- **New "Coverage" column per gap** with values
  `{real-gap, book-exists, book-anchored, book-exists+anchored,
  templated-leak}`. Precedence on disagreement:
  `templated-leak > book-anchored > book-exists > real-gap`.
- **Two top-20 sections per pipeline:** unfiltered (current behavior,
  with Coverage column) and filtered (excludes `book-exists`,
  `book-anchored`, `templated-leak`). The filtered view is the
  leaderboard we'd actually ship.
- **Cross-check summary** at the bottom: 4-cell agreement matrix
  between Phase 1 and Phase 3 flags.

---

## Reassessment checkpoints

After each phase: regenerate the comparison table on T10181 (and
T11303 for Phase 2's confluence diagnostic), inspect the diagnostics
listed in that phase, decide whether to proceed.

The Phase 2 → Phase 3 dependency is hard: Phase 3's exact-match dedup
across book and article seeds requires the symmetric fixed-point
property from Phase 2 to fire reliably. If Phase 2 doesn't deliver
strong confluence improvement (T11303 distinct-frozenset count fails
to drop meaningfully toward 254), reassess Phase 3 before starting.

---

## Files touched

- **New:** `artifacts_scripts/gap_book_existence_filter.py`,
  `artifacts_scripts/_keyword_helpers.py`
- **Modified:** `artifacts_scripts/gap_article_as_seed.py` (flags +
  symmetric loop + book seeds + boundary lookup),
  `artifacts_scripts/gap_method_comparison_table.py` (Coverage
  column, filtered top-20, agreement matrix; calls into
  `_keyword_helpers`)
- **Deleted (after Phase 2 lands):**
  `docs/backlog/gap-detection-methods/article-as-seed-canonicalization.md`,
  `docs/backlog/gap-detection-methods/article-as-seed-batch-addition.md`
  (subsumed by Phase 2)
- **Deleted (after Phase 1 lands):**
  `docs/backlog/cohesion-metric-design/vote-concentration-as-book-exists-filter.md`
- **Deleted (after Phase 3 lands):**
  `docs/backlog/gap-detection-methods/article-as-seed-with-book-seeds.md`
