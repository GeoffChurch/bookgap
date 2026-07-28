# Rank-1 (teleport) vs diagonal (`+I`) regularisation — audit the graph operators

**Status:** backlog (opened 2026-07-28)
**Origin:** measured in the curriculum repo while designing a citation/phrase
interpolation; see `~/src/curriculum/docs/findings/2026-07-28-selfloop-laziness-in-the-citation-operator.md`
for the full numbers. Filed here because bookgap builds the graph these operators
run on, and the same question applies to the denoisers.

**This is a question to measure, not a fix to apply.** The right answer differs
per call site, and at least one legitimate use of the diagonal exists.

## What was measured

The `cyberbullying_ai1702` bridge npz stores `A = A_real + I` — a self-loop on
**100%** of 1,901,568 works (60% of them have *only* the self-loop, i.e. all their
references fall outside the slice). Row-normalised, that gives:

```
diag(C)  mean 0.6935   median 1.0000
```

The median work's entire out-edge mass is its own self-loop.

## Why it is not free

A uniform lazy walk `(1-c)P + cI` with **constant** `c` has the same stationary
distribution as `P` — this is the intuition that makes `+I` look harmless. It does
not hold when `I` is added **before** row-normalisation, which yields

```
c_i = 1 / (1 + outdeg_i)
```

— degree-dependent laziness, which *does* move the fixed point, retaining mass at
low-out-degree nodes. In a citation slice, low out-degree means "most of my
references are outside the corpus": recent work, books, cross-domain work.

**Measured cost on personalised PageRank** (RL seed, tau=0.15, 40 iterations),
rank of canonical papers with `+I` vs with it removed and dangling rows routed to
the teleport vector:

| paper | `A_real + I` | de-lazied |
|---|---|---|
| Trust Region Policy Optimization | 120 | **16** |
| Playing Atari with Deep RL | 150 | **21** |
| Soft Actor-Critic | 273 | **43** |
| Reinforcement Learning: A Survey | 124 | **12** |

A 6–8x improvement in canonical recovery from removing the diagonal alone.

## The distinction to audit for

The two regularisers solve different problems and are not substitutes:

| | `+ cI` (diagonal) | `+ tau * 1 r^T` (rank-1) |
|---|---|---|
| aperiodicity | yes | yes |
| **irreducibility** | **no** | yes (`r > 0`) |
| preserves stationary vector | only if `c` is constant AND post-normalisation | n/a — defines it |
| carries a query/seed | no | yes (personalisation) |

A citation graph at this scale is not strongly connected, so the diagonal alone
does not guarantee a unique Perron vector.

**The diagonal is still correct for one job**: deliberate laziness for spectral
conditioning — keeping eigenvalues in `[0,1]`, damping period-2 structure — as
used in heat-kernel / spectral-clustering settings. When that is the job, apply a
**constant** after row-normalisation, where it provably does not move the fixed
point.

## Suggested audit

For each graph operator in bookgap (PageRank paths, the denoisers, any
Tikhonov-style regularisation), classify what the regulariser is standing in for:

1. **dangling nodes** -> should be rank-1 (route dangling rows to the teleport
   vector), not a self-loop;
2. **irreducibility** -> must be rank-1; a diagonal cannot do it;
3. **spectral conditioning / laziness** -> a diagonal is right, but as a constant
   applied *after* normalisation;
4. **shrinkage toward a prior** -> note *which* prior: `I` shrinks toward "stay
   put" (adds no information and is degree-coupled if applied pre-normalisation);
   `1 r^T` shrinks toward a chosen distribution and is where a prior belongs.

Where (1) or (2) turns out to be the actual intent, expect the fixed point to
move, so re-measure whatever the operator feeds rather than assuming the change
is cosmetic.

## Related observation

The phrase-side operator (`work -> phrase -> work` over the field-term index) has
`diag(RQ)` mean **0.0043** — essentially non-lazy — so mixing the two operators
without first de-lazying the citation side compares a nearly-absorbing walk
against a freely-mixing one.
