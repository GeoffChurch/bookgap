# Change-form temporal gaps — do *widening* gaps forecast arrivals where static gaps don't?

**Status: not started.** The one surviving lead for a *positive* temporal signal. Every **static**
snapshot→arrival test is a universal negative (articles, Lean, Python, Rust, Go — a detected gap does
not forecast where the field's next unit lands; preferential attachment wins;
`docs/code-gap-triangulation.md`). But maybe the *rate of change* of a gap does.

## The test
Reuse the epoch `births.npz` already built for Python / Rust / Go. For each candidate node compute
Δ (= log(struct+1) − log(oper+1)) at epoch T **and** at an earlier epoch T−k; rank by the **change**
`ΔΔ = Δ(T) − Δ(T−k)` (a gap that is actively *widening*). Feed this ranker into the existing
temporal-holdout arrival-prediction (growth-matched null on (existing-usage, in-degree) deciles,
calibration-first).

**Question:** does `ΔΔ` predict where new operations land *above the growth-matched null*, where
static Δ gave fill-ratio ≪ 1 / stratified AUC ≈ 0.5? If yes anywhere, it is the **first positive
temporal result** in the whole project. (Also try the differenced book-surprise / level-set-change on
the article domain — `docs/end-to-end-overview.md` already flagged "only level-set *change* survives"
on the article temporal track, so a change-form signal there is the prior with the best odds.)

## Reuses / difficulty
Existing `births.npz` + `core/temporal_holdout.py` (add a differenced-M1 ranker alongside the static
M1). **Difficulty: easy–medium** — all data exists; new ranker + re-run.
