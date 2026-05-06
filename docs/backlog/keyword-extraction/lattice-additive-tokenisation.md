# Lattice / additive tokenisation (semtok-style)

(Migrated from `artifacts_scripts/curriculum/TODO.md`. Source: `mycooc/vocab/semtok_vocab.py`.)

Both `un` and `unhappy` coexist as tokens; conductance-based forward-backward weighting picks among overlapping segmentations without a count floor.

More complex than greedy left-to-right merging and addresses the order-dependence we accepted in the iterative phrase miner. Rarely needed for canonical scientific phrases but useful for less canonical text (e.g., scraped page content, multilingual abstracts).

**Priority**: low. Current production iterative `parse_with_phrases` (longest-match) handles the common cases well.
