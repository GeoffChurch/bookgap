# Share full-index kNN across dual-landmark `compute_gap_stats` calls

## Idea

`compute_gap_stats` (`artifacts_scripts/gap_kball_llr_refine.py` ~line
290) runs the dominant per-member full-corpus kNN
(`full_index.search(agg_member_emb, k=k_query+1)`) internally. The
dual-landmark bake-off (`gap_denoiser_bakeoff.run_refine_pipeline`,
spec §11) calls `compute_gap_stats` twice per gap — once with
book-type landmarks, once with `book-chapter`-type — to get
`effective_size_books` / `effective_size_chapters`. Only the
landmark-side work (1-NN-book + per-book article query) differs
between the two calls; the expensive full-corpus member kNN is
**landmark-independent and currently recomputed both times**.

Refactor: factor the `full_index.search(agg_member_emb, ...)` out of
`compute_gap_stats` into an optional `precomputed_member_nn=(D_NN,
I_NN)` parameter (default `None` → compute as today, preserving every
existing caller). The dual-landmark caller computes the member kNN
once and passes it into both calls.

## Why

Measured during the denoiser bake-off (code review 2026-05-15): the
dual-landmark design costs ~**2× Phase-3 stats per gap**, not the
~10–30% the original dual-landmark backlog estimated. The 2× is
entirely the duplicated full-corpus kNN. Factoring it out makes
dual-landmark ~1.1–1.3× instead of ~2×, restoring the original cost
expectation and making per-gap dual reporting cheap enough to enable
by default.

**Why:** the bake-off accepted the 2× because it is a one-time
3-topic overnight job and modifying `gap_kball_llr_refine.py` was out
of scope for that design (it deliberately reused the function
unchanged). For any larger run (subfield/domain scale, or promoting
dual-landmark into the production `LLR_REFINED` pipeline) the 2×
becomes prohibitive.

**How to apply:** do this before any subfield+-scale dual-landmark
run, or as part of promoting dual-landmark reporting into the
official pipeline (see
`docs/backlog/cohesion-metric-design/dual-landmark-eff-size-reporting.md`).
The change is backward-compatible (new optional kwarg) so all
existing single-landmark callers are unaffected.
