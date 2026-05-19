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

## Caveats

Low absolute recall@20 (0.10–0.25) → read the *relative* curve +
qualitative, not as an oracle (the standing posture). Keep the
byte-stable index config (`_build_topic_index` efc=512/efs=2000 or
`--index-type flat`).
