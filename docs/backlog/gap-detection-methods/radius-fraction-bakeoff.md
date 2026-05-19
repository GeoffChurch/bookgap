# radius_fraction bake-off

## Idea

The entire deflation investigation (and the paper's k-balls) used a
single fixed `radius_fraction = 0.75`, `k = 1` — i.e. each article's
k-ball radius = 3/4 × (L2-on-unit-sphere ≈ cosine distance to its
nearest book), ball over the article index. That value was never
swept; it was inherited as a default. Sweep it.

## What to run

Reuse the deflation bake-off harness
(`artifacts_scripts/gap_deflation_bakeoff.py`) — `build_kball_candidates`
already takes `radius_fraction` / `k`. Hold deflation =
`fl_bookdemand_resid` (the validated winner) and the corrected
**genuine recall@20** metric fixed; sweep `radius_fraction ∈
{0.4, 0.5, 0.6, 0.75, 0.9, 1.0}` (± a `k ∈ {1,2,3}` arm) across the
4 regime-spanning topics (T11303/T10462/T10320/T10820). The
`--rank`/`--scope`/`--out` plumbing exists; only a `--radius-fraction`
CLI passthrough to `build_kball_candidates`/`kball_seeds_from_state`
is missing (small add).

## Why it might matter

`radius_fraction` directly sets candidate ball size → trades gap
granularity vs coverage. The paper flags "choice of the k-ball
radius" as future work; held-out recall@20 (now sound) gives an
objective way to pick it instead of inheriting 0.75. Production
`GapConfig.radius_fraction` default is also 0.75 — a better value
would propagate to the production opt-in resid path too.

## Data-driven radius arm (2026-05-19 extension)

The constant-fraction sweep above only moves a *global* knob.
**Also add a data-driven per-seed radius arm**, because the
recall@K curve (`docs/gap_deflation_results.md` § "recall@K curve")
left an open question worth a direct test: **is `effsize` (n_eff)
underperforming on book-rich topics because the fixed 0.75
candidate-radius is the wrong scale for it, not because the n_eff
formula is wrong?** n_eff already uses a **data-driven kernel
bandwidth** internally (`state["sigma"]`, per-article self-tuning),
but it inherits a *fixed-0.75* candidate set — a bandwidth/radius
mismatch. Closing that mismatch is the principled test.

Concrete data-driven radius candidates (each replaces the
`radius_i = 0.75 × d(i, nearest_book)` rule in
`build_kball_candidates` / `kballs_with`):

- **σ-tied (cleanest):** `radius_i = α · sigma_i` (or a small fixed
  multiple of the article's self-tuning bandwidth). Ties the ball
  scale to the same per-point local scale n_eff uses → eliminates
  the bandwidth/radius mismatch by construction.
- **Margin-based:** `radius_i = ½ · (d(i, nearest_book) +
  d(i, 2nd-nearest-book))` — the local Voronoi-margin scale; uses
  no fixed fraction.
- **Local-density adaptive fraction:** keep `radius_i = c_i ·
  d(i, nearest_book)` but let `c_i` adapt to local article/book
  density (e.g. via a k-NN distance ratio) instead of a global 0.75.

Run as a small **factorial**: data-driven radius arm ∈ {fixed-0.75
(control), σ-tied, margin-based, density-adaptive} × rank ∈
{raw_kball, effsize, effsize_chapters} × 4 topics, recall@K curve
primary. Deflation stays `fl_bookdemand_resid`. Hypothesis-test
read: if `effsize`'s book-rich top-K shortfall closes (or
reverses) under σ-tied radius while raw_kball is roughly unchanged,
the fixed-radius IS the handicap, not the formula. If both move
similarly, the formula vs raw is the more fundamental story.

## Caveats

Low absolute recall@20 (0.10–0.25) → read the *relative* curve +
qualitative, not as an oracle (the standing posture). Keep the
byte-stable index config (`_build_topic_index` efc=512/efs=2000 or
`--index-type flat`). The data-driven radius arm changes
`build_kball_candidates`'s contract — implement as a new helper
(`build_kball_candidates_adaptive(...)` or a radius-strategy
parameter) so the validated constant-fraction path stays
byte-anchored.
