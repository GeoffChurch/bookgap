# Change-form temporal gaps (ΔM1 across epochs) — REFUTED: widening gaps don't get filled

**Tested 2026-06-15** (code-gap triangulation follow-on; full result in
`docs/code-gap-triangulation.md` §Follow-on, script `change_form_temporal.py`). The one shot at a
*positive* temporal signal, since every static snapshot→arrival test is a universal negative: do gaps
that are actively **widening** (`ΔM1 = M1(T) − M1(prev)`) forecast where new operations land, where
static M1 doesn't? Compared static_M1 vs change_dM1 on the same candidates + growth-matched null over
the Python / Rust / Go epoch births.

**Expected:** widening gaps attract new operations (a catch-up / mean-reversion signal).

**Actual:** the centrality-stratified AUC (the de-confounded metric — raw fill-ratio is bootstrap
noise at these sparse-attach sizes, random baseline swinging 0.09→14.7) shows change_dM1 is robustly
**anti-predictive in Python (≈0.26 across 3 epoch-pairs × 2 seeds)**, near-chance in Go (0.55),
too-small in Rust (141 candidates).

**Diagnosis:** the change-form lead does **not** rescue temporal prediction — it *sharpens* the
universal negative. Concepts whose under-exercise is *accelerating* are the *least* likely to attract
new operations (the gap compounds — poor-get-poorer). Preferential attachment dominates even in
differenced form.

**Salvage:** the snapshot→arrival prediction direction now looks exhausted (static **and** change-form
both negative across every measurable domain — articles, Lean, Python, Rust, Go). Any future temporal
lead would need a fundamentally different signal, not a transform of the gap score.
