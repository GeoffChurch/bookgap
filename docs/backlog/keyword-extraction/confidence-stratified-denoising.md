# Confidence-stratified denoising for keyword candidates

(Migrated from `artifacts_scripts/curriculum/TODO.md`. Source: `mycooc/docs/backlog/hubness-and-gep/gep-confidence-stratified.md`.)

Apply aggressive correction only to *low-confidence* (low-HHI) tokens; leave *high-confidence* (peaked, well-supported) tokens untouched.

**For bookgap**: stronger boilerplate/noise filtering on diffuse candidates (scattered across many docs weakly), conservative on focused candidates (concentrated in a few high-relevance docs). Naturally pairs with branching-entropy as the confidence signal.

**Priority**: medium. Could meaningfully improve keyword quality on noisy corpora; lower priority for the current curated scopes.
