# Soft hubness correction (CSLS-style) for keyword extraction

(Migrated from `artifacts_scripts/curriculum/TODO.md`. Source: `mycooc/soft_coocmap.py:_gep_csls`.)

Iterative residual penalty for tokens that appear as candidates in too many topics' top-K simultaneously. Generalises bookgap's marginal-bg + freq-weighting:

- **marginal-bg + freq** (current default): demotes globally-frequent words *once*
- **CSLS-style**: demotes them *iteratively*, with soft (not hard) cutoffs that preserve borderline genuine domain words

Probably overkill given marginal+freq already drives cross-topic crowding to ~zero. Worth knowing about for future scenarios where cross-topic crowding becomes a problem (e.g., much larger topic counts).

**Priority**: low.

## Related: CSLS in cross-corpus gap detection (distinct application)

The CSLS hubness diagnostic was implemented for the OpenAlex↔Wikipedia
cross-corpus NN check (`artifacts_scripts/wiki/csls_nn_check.py`, results in
`docs/wikipedia-openalex-shared-space.md` §"CSLS + topic-relevant reverse NN").
There it reduced NN top-1 hub share (0.040→0.015 OA→wiki, 0.030→0.010 wiki→OA) but
the absolute hubness was already low in the mean-centered topic-relevant regime.

A **different** application of the same idea would touch the cross-corpus gap
k-ball: an *anti-hub* ambient point (far from every landmark *book*) gets a large
nearest-book distance → large ball radius → sweeps in many neighbors → flagged as a
large gap. A CSLS-style local-scaling correction to the **query→landmark distance**
(not the cross-corpus NN) would shrink the ball for points whose large
nearest-book distance is an anti-hubness artifact rather than a real coverage hole.

Not recommended on current evidence (see the doc's verdict): the k-ball does not
use the cross-corpus NN score CSLS was measured on, and the observed all-Wikipedia
language-family gaps are a genuine *topical* book-coverage hole, not a hubness
artifact. The radius-correction variant is plausible but **unmeasured**, changes
gap *sizes* (not just orderings), and risks suppressing the real cross-corpus holes
we want to surface. Validate it standalone before touching the gap algorithm.

**Priority**: low (radius variant: speculative, validate-first).
