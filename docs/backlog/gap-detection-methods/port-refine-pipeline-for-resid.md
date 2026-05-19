# Port the refine pipeline for production resid

## Context

`fl_bookdemand_resid` is now selectable in the production `bookgap/`
pipeline as an opt-in `GapConfig.deflation_strategy`
(`DeflationStrategy.FL_BOOKDEMAND_RESID`); the default remains the
byte-identical greedy hard-subtraction. See the design spec
`docs/superpowers/specs/2026-05-19-promote-resid-production-optin-design.md`.

That promotion was a **raw deflation swap only**. Production has **no
refine stage**, but resid's validated bake-off recall@20 numbers were
measured **post-`run_refine_pipeline`** (LLR-refine → dedup →
effective_size). So the production resid path currently emits the raw
deflated K-ball candidates, not the refined gaps the bake-off scored.

## Idea

If the deferred qualitative review + the ollama LLM-target-gap-coverage
check (see `MEMORY.md` → "LLM-coverage eval via ollama") shows that
**raw** production resid underperforms its refined bake-off numbers,
port the refine pipeline into production, gated behind the resid
option only:

1. LLR-refine of each resid gap's member set
   (`artifacts_scripts/gap_kball_llr_refine.py` is the validated
   reference).
2. Containment + Jaccard dedup of the refined gaps.
3. effective-size recomputation / re-ranking.

Keep it strictly opt-in and behind `FL_BOOKDEMAND_RESID`; greedy_hard
must stay byte-identical. Do NOT port refine if raw resid already
matches its refined numbers on the production corpus — the simpler
path wins.

## Status

Not started. Blocked on the deferred ollama LLM-coverage comparison of
raw production resid vs. refined bake-off resid.
