# Gap-deflation formulation

How to deflate one gap to get the next, satisfyingly/optimally. The
current pipeline has two inconsistent deflation models (greedy
hard-subtraction at seed time vs soft joint-norm + hand-tuned dedup
later); see `docs/superpowers/specs/2026-05-17-gap-deflation-bakeoff-design.md`.

Bake-off arms (all in `artifacts_scripts/gap_deflation_bakeoff.py`,
swappable via `deflate(cands, strategy, ...)`): `greedy_hard`
(baseline, byte-anchored to `kball_gap_probe._greedy_kball_gaps_inline`),
`greedy_geom` (residual-restricted pick), `submodular_fl` (sparse
k-ball facility-location), `dpp` (greedy MAP-DPP, candidate-pool
capped), `softmax_mc` (memory-bounded kNN-basin multicluster-softmax;
first build of `multicluster-softmax-membership`). Eval:
qualitative-primary leaderboards + inter-gap metrics + held-out-real-books
recall/spread (signal A) + synthetic-implant gate (signal C). Frozen
context: T10181, LIP, binary (T,F) graph (per
`docs/denoiser_bakeoff_results.md`).

## Status / working notes

- All 5 arms implemented + memory-bounded (no dense A×K anywhere),
  byte-stable harness reused; greedy_hard byte-anchored to the oracle.
- Observation worth following up at analysis time: DPP's Jaccard
  diversity degenerates when the k-ball radius is large relative to
  cluster separation (all candidate balls become near-identical, so
  the determinant has no diversity signal). This is expected to matter
  on the real L2-normalized sphere when books are sparse/far — flag
  when reading the real-run leaderboards.
- (living section — updated as the bake-off runs; migrates to
  `docs/gap_deflation_results.md` on completion)
