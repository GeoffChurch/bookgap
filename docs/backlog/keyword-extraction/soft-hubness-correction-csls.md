# Soft hubness correction (CSLS-style) for keyword extraction

(Migrated from `artifacts_scripts/curriculum/TODO.md`. Source: `mycooc/soft_coocmap.py:_gep_csls`.)

Iterative residual penalty for tokens that appear as candidates in too many topics' top-K simultaneously. Generalises bookgap's marginal-bg + freq-weighting:

- **marginal-bg + freq** (current default): demotes globally-frequent words *once*
- **CSLS-style**: demotes them *iteratively*, with soft (not hard) cutoffs that preserve borderline genuine domain words

Probably overkill given marginal+freq already drives cross-topic crowding to ~zero. Worth knowing about for future scenarios where cross-topic crowding becomes a problem (e.g., much larger topic counts).

**Priority**: low.
