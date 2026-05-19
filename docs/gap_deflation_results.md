# Gap-deflation: results

**Question.** "Find a gap, deflate, get the next" — is there a single
deflation strategy that rediscovers where humans actually wrote books,
across regimes (book-rich classical, modern article-dense, book-sparse),
*without* a regime branch or tuned constants?

**Answer.** Yes: **`fl_bookdemand_resid`** (`artifacts_scripts/
gap_deflation_bakeoff.py`) — facility-location over book-anchored demand
with an intrinsic background coreset built only over book-orphan
articles, sized `M_bg = clip(|O| // max(k_book·n_books, k_book), 0,
4000)`. Zero tuned constants, no regime `if`. It is the best or tied
method on genuine held-out-book recall@20 across all four
regime-spanning topics, and decisively best on the hardest case
(book-sparse T10462).

## The result (corrected metric — see the lesson below)

Genuine `heldout_recall_at20` (top-20 surviving gaps, ranked by
`effective_size`, vs 20% held-out real books; eval in
`gap_deflation_bakeoff.py: heldout_recall_spread`, sliced `ranked[:20]`):

| topic (regime)            | greedy_hard | submodular_fl | fl_bookdemand | **fl_bookdemand_resid** | fl_union_bd_sub |
|---------------------------|------------:|--------------:|--------------:|------------------------:|----------------:|
| T11303 (book-rich)        | .000 | .000 | .103 | **.103** | .000 |
| T10462 (book-sparse)      | .000 | .000 | .125 | **.250** | .000 |
| T10320 (modern dense)     | .000 | .006 | .012 | **.042** | .000 |
| T10820 (book-rich)        | .000 | .000 | .130 | .109 | .000 |

(Validity gate: with the metric run in its old full-set mode, every arm
byte-reproduced the prior numbers — the correction left the underlying
computation untouched. Absolute recall@20 is low by construction —
top-20 gaps vs ~8–165 held-out books is an honest, hard proxy; the
**relative** ordering is the robust signal and triangulates with
qualitative-primary review, never an oracle.)

**Why resid wins.** Only *few-sharp-book-anchored* deflations
(`fl_bookdemand`, `fl_bookdemand_resid`) put held-out-book regions in
their top-ranked gaps. High-volume coverage methods (`greedy_hard`,
`submodular_fl`, `fl_union_bd_sub`) emit ~800–1100 gaps but score ≈0.0
recall@20 — none of their top-20 align with book regions. Among the
book-anchored arms, `resid`'s intrinsic orphan-density-per-book
operating point matches or beats the original `fl_bookdemand` and never
collapses (the original collapses to 30 gaps on book-sparse topics);
`resid ≥ fl_bookdemand`.

## The headline lesson (the most instructive artifact)

**A multi-stage investigation's headline metric was a misnomer, and it
silently inverted the conclusions.** `run_arm` passed the *full* deduped
survivor set (not the top-20) to `heldout_recall_spread`, so
`heldout_book_recall_at_20` actually measured *set coverage* — it
rewarded **emitting many gaps**, not surfacing good ones, and was
completely rank-insensitive. Under that broken metric the coverage
methods looked strong (`submodular_fl` .50 on book-sparse,
`fl_union_bd_sub` an apparent .625 "SOTA"), which produced an elaborate
and entirely spurious narrative: a "fundamental book-vs-manifold
tension", a regime gate, a best-of-both search, and a conclusion that
"the regime-blind `effective_size` ranking is the binding constraint."

Correcting the metric to genuine recall@20 (`ranked[:20]`) dissolved all
of it: the coverage methods score ≈0.0, there is **no tension**, no gate
is needed, and the simplest principled book-anchored method wins. The
detours and their unifying diagnosis are preserved in
`docs/dead_ends/cohesion-metric-design/
deflation-demand-engineering-metric-artifact.md`.

Process takeaway (also in memory): before trusting any comparative
sweep, verify the metric's *implementation* measures what its name
claims — read what set/slice/space it consumes. A plausible column name
is not proof.

## Gap-selection key: raw KBALL vs n_eff vs dual-landmark (2026-05-18)

The session's *founding* question — adapt the paper to rank gaps by
n_eff / book-chapter landmarks *instead of* raw KBALL ranking — is
now answerable on the corrected metric (it was unmeasurable before:
the full-set misnomer made ranking invisible). Deflation fixed at
`fl_bookdemand_resid`, sweeping the `--rank` key, genuine recall@20.
Consistency gate passed: `effsize` byte-matched the prior re-baseline
resid@effsize (.103/.25/.042/.109); `heldout_recall_fullset` is
rank-invariant per topic (.621/.375/.618/.565) as it must be.

| topic            | raw_kball | effsize (n_eff book) | effsize_chapters (dual-landmark) |
|------------------|----------:|---------------------:|---------------------------------:|
| T11303 book-rich | **.138**  | .103 | .069 |
| T10462 sparse    | .250      | .250 | .250 |
| T10320 modern    | .030      | **.042** | .036 |
| T10820 book-rich | **.152**  | .109 | .152 |

**Verdict: the founding premise is NOT supported by held-out
rediscovery.** Raw KBALL ranking is ≥ n_eff (better 2/4, tied 1/4,
worse only on modern-dense T10320); dual-landmark chapters is not a
systematic improvement (worst on T11303, tied-best T10820, middling
else). Correct statement: *no evidence on the book-rediscovery axis
that n_eff or dual-landmark beats raw KBALL; raw is at least as
good.* Small deltas, 4 topics, low absolute recall — triangulate,
not oracle. **If the paper adopts n_eff/dual-landmark it must be
justified on other grounds (qualitative, LLM-coverage); held-out
rediscovery now argues *against* the change.** This bears directly
on the `dual-landmark-eff-size-reporting` backlog entry.

## Status / follow-ups

- **Validated and resolved.** Winner lives in code
  (`_fl_bookdemand_resid`, with docstrings) + this doc.
- **Not yet promoted into the production pipeline.** `resid` is a
  bake-off arm in `artifacts_scripts/`; wiring it into the main
  `bookgap/` deflation path (replacing the current greedy
  hard-subtraction at seed time / the soft joint-norm dedup) is a
  separate, untaken follow-up. CLAUDE.md is intentionally unchanged
  until that promotion happens.
- The deferred pre-paper LLM-target-gap-coverage confirmation should
  use a local ollama adapter (not OpenAI Batch); see the project
  memory note. A confirmatory qualitative + LLM-coverage read on
  `resid` (esp. book-sparse) is the natural next checkpoint before the
  paper relies on this.
- The held-out-recall eval and any conclusion that ever rested on it
  must be read as **genuine recall@20**, not the old full-set value;
  the `book-rediscovery-eval-candidates` tracker carries this caveat.
