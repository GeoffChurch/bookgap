# Dual-landmark `effective_size` reporting per gap

> ⚠ **Update (2026-05-18, corrected-metric):** as a *gap-selection
> ranking key*, dual-landmark `effective_size_chapters` does **not**
> beat book-landmark n_eff or raw-KBALL ranking on genuine held-out
> recall@20 (deflation fixed at `fl_bookdemand_resid`: T11303
> .069 vs raw .138 / n_eff .103; tied/middling elsewhere). The
> dual-landmark signal is still **informative as a diagnostic
> cross-tab** (the four-corner / LIP-LPP mirror-image result stands),
> but it is *not* validated as a better ranking criterion than raw
> KBALL on the book-rediscovery axis. **recall@K-curve update
> (2026-05-19):** the regime split is now sharper — on modern
> article-dense (T10320) the better top-ranked key is **book-landmark
> n_eff** (raw finds nothing in top-10); `effective_size_chapters` is
> only ever *top-key-competitive* there, never the clear winner on
> any regime/K. See `docs/gap_deflation_results.md`
> (§ "recall@K curve"). Promotion into production must rest on the
> diagnostic value or other axes, not on rediscovery ranking.

## Idea

Each LLR-refined gap should carry `effective_size_books` and
`effective_size_chapters` as separate fields (alongside the current
single `effective_size`). One K-ball seed run, two posteriors at
phase 3 — one against `book`-type landmarks, one against
`book-chapter`-type landmarks.

This makes the per-gap object self-describing about *which kind* of
existing exposition is sparse: book-form, chapter-form, both, or
neither.

## Why

Books and chapters are qualitatively different "coverage" signals:

- **Book** = comprehensive monograph treatment of a subject.
- **Book-chapter** = focused exposition of a subtopic within a
  broader volume.

Conflating them as a single landmark set hides real information.
With dual reporting, downstream filters can address all four corners:

| `eff_books` | `eff_chapters` | Interpretation |
|---|---|---|
| high | high | adequately covered by both — skip |
| high | low | book-form sparse, chapter-form sparse — strongest textbook-gap signal |
| low | high | chapter-covered already; standalone textbook maybe redundant |
| low | low | covered by neither — equivalent to today's "book gap" |

The third row is invisible to a single-landmark pipeline. It is
exactly the case "is this gap actually covered by chapters already?"
that motivated this entry.

## Evidence

Tested on T11303, T10181, T10462 by running the existing
standalone pipeline twice — once with book landmarks, once with
chapter-only landmarks (plus article+preprint ambient).
Top-10 rankings under M9 (`effective_size`) differ substantially:

- **T11303** (Bayesian/probabilistic): 7/10 of book-top picks have
  zero member overlap with chapter-top picks. Book ranking emphasises
  Gaussian graphical models, IRT, BN structure; chapter ranking
  emphasises belief revision, alternative Bayes, ranking
  elicitability.
- **T10181** (NLP): 5-7/10 partial matches at Jaccard 0.2-0.5. Both
  surface real subareas but with different cuts ("Chinese-specific
  analysis" book-top vs "resources/tools" chapter-top).
- **T10462** (RL/robotics): 4/10 partial matches; 6/10 disjoint. T10462
  has only 42 books for 22k works, so book landmarks are very sparse;
  chapter landmarks (300) give noticeably tighter and bigger gaps.

The CSVs are at `artifacts/gap_kball_llr_refine/` (book-only) and
`artifacts/gap_kball_llr_refine/chapter_only/` (chapter-only).
Comparison tool: `artifacts_scripts/gap_chapter_vs_book_overlap.py`.

## Cost

The expensive phase-3 operation is `full_index.search(member_emb,
k=k_query+1)` — per-member kNN over the full ambient cloud. This is
**landmark-independent**. The landmark-specific work is:

- 1-NN of each member against the book index (cheap; landmark
  index is small).
- Top-K articles per *unique* nearest landmark (one HNSW query per
  distinct landmark touched by the gap's members).

Adding a second landmark set (chapters: 3-7× more landmarks than
books) duplicates only the cheap parts. **Estimated phase-3
overhead: 10-30% per gap.** Per-topic total LLR runtime goes from
~250-600s to ~280-720s.

## Sketch

In `compute_gap_stats`, parameterise the landmark side. Returns
either a single posterior or a dict of `{landmark_label:
posterior_stats}`. Caller iterates over landmark configs.

`LLRRefinedGap` gains:

```python
effective_size_books: float
effective_size_chapters: float
median_logit_books: float
median_logit_chapters: float
```

`RefinementConfig` gains:

```python
primary_book_types: tuple[str, ...] = ("book",)
secondary_book_types: tuple[str, ...] | None = ("book-chapter",)
# None disables the secondary posterior, restoring single-landmark
# behaviour and runtime
```

Default `rank_by` becomes `"effective_size_books"` (preserving
today's "book gap" semantics) but can be set to
`"effective_size_books_plus_chapters"` (sum) or any of the new
fields for alternative use cases.

## Open questions

- **K-ball seed: books or books ∪ chapters?** Currently this entry
  assumes seeds-with-books-only (one K-ball run). If seeds use
  both, the gap *shape* changes too, not just the per-gap measure.
  Seed-with-both is more symmetric but loses the "book-sparse"
  framing that today's pipeline guarantees. Tentative answer: keep
  the seed step on the primary book-type set, so the gap set is
  pinned to "where the primary landmarks are sparse" and the
  secondary posterior is purely a filter signal.
- **Should `median_logit` be reported per landmark too?** Probably
  yes — adds negligible compute and lets downstream rank-metric
  experiments use either.
- **Three or more landmark sets?** E.g. books, chapters, reviews.
  Same shape, more bookkeeping. Defer until a third concrete
  category gets requested.

## Related

- The current single-landmark behaviour is hardcoded as
  `is_book = types == "book"` in
  `artifacts_scripts/gap_static_seed_diagnostic.py` (now
  configurable via `setup(book_types=...)` after the May 14
  parameterisation).
- The integration design at
  `docs/superpowers/specs/2026-05-13-cell-type-llr-refined-integration-design.md`
  references this entry under "Out of scope" — promote it to
  in-scope once the initial single-landmark integration has
  landed.
- Comparison tool: `artifacts_scripts/gap_chapter_vs_book_overlap.py`
  matches top-N gaps under two landmark configurations by member
  Jaccard.

## Status / evidence (2026-05-16)

**Prototyped and validated in the bake-off harness**, production
`LLR_REFINED` integration still pending. `gap_denoiser_bakeoff.py`
computes per-gap `effective_size_books` / `effective_size_chapters`
(Task 6B) and the four-corner (bc/bC/Bc/BC) cross-tab. The four-corner
is *informative*, not inert — it is what surfaced the LIP/LPP
mirror-image result (`docs/denoiser_bakeoff_results.md`): the clean
"book-sparse, chapter-low" vs "both-high" structure (`Bc≈7 / BC≈121`)
cleanly separates the best configs from the scrambled ones. Confirms
the dual-landmark signal carries real discriminative information and
is worth promoting into the production pipeline. Cost caveat tracked
in `infrastructure/share-fullindex-knn-dual-landmark`.
