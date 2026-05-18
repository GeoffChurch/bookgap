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

### Cross-topic results (2026-05-18) — T10181 + T11303 + T10462

Held-out book recall@20 (greedy_hard / greedy_geom / submodular_fl / dpp):
- T10181 (NLP):        0.128 / 0.128 / **0.156** / 0.044
- T11303 (Bayesian):   **0.483** / **0.483** / 0.379 / 0.103
- T10462 (RL, sparse): 0.375 / 0.375 / **0.500** / 0.375

Conclusions (triangulated w/ qualitative leaderboards, all 0 errors):
- **greedy_geom ≡ greedy_hard** byte-identical on all 3 topics → the
  residual-restriction is inert at top-20. Drop greedy_geom.
- **submodular_fl is topic-dependent, NOT a universal winner**: wins
  T10181+T10462, *loses to plain greedy_hard on T11303*. The
  single-topic "submodular_fl wins" is overturned.
- **dpp robustly refuted for book-gaps**: worst/weakest recall + lowest
  spread on all 3; top gaps often non-book-depleted (high eff, p≈0.5,
  language clusters). Optimizes manifold coverage, not book-relevance.
- **softmax_mc** refuted on T10181 (over-fragments, artifact-prone);
  not re-run cross-topic.
- Original "greedy hard-subtraction → redundant gaps" premise NOT
  borne out: all arms' top-20 near-non-redundant (downstream
  refine+dedup handles it). Real differentiator = book-relevance of
  selection, where greedy_hard is already robust.

**Recommendation:** greedy_hard is the robust default (adequately
solves deflation; even wins T11303); submodular_fl = per-scope A/B
option, not a replacement; drop greedy_geom; don't use dpp/softmax_mc
for book-gaps. Caveats: proxy not LLM-coverage; T10462 spread
saturates (~ln of few held-out books); modest absolute recall —
triangulation not oracle.

**Status: investigation conclusive across 3 topics. Pending: sign-off
to migrate to `docs/gap_deflation_results.md` + transition this entry
per the validated/refuted convention.**
