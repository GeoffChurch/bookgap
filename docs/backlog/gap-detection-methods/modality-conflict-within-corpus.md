# Within-corpus modality conflict — the decisive test that gap *modality* (not domain) sets the detector

**Status: not started.** Follow-on to the completed code triangulation
(`docs/code-gap-triangulation.md`), which showed across 6 domains that the **graph** detector finds
structural gaps and the **embedding** detector finds semantic-void gaps — but using *different gap
definitions per domain*, leaving the objection "the definitions differ, not the domains."

## The test
In ONE corpus (Python, the existing `cache/code_gap/python/data`), plant BOTH gap types and show each
detector is *blind* to the other:

- **Structural gap** = the existing `--region graph` plant (a class structurally embedded but
  under-exercised; Δ = log(struct+1) − log(oper+1)). Graph M1 should recover it; embedding M_emb ≈ chance.
- **Semantic-void gap** = a dense region of F-nodes (functions) in embedding space with NO B-node
  (class) nearby — the article *book-surprise* analogue *inside code*. Plant by selecting an
  embedding-dense function cluster with no covering class (or delete a class, leaving its functions an
  orphaned semantic void). The embedding density-ratio / level-set should recover it; M1 ≈ chance (the
  graph has no node where there's a void).

Run the full 2×2 (each detector × each gap type) on the same substrate, embeddings, and graph.

## Why
**Expected:** a clean diagonal — graph finds only the structural gap, embedding only the
semantic-void — proving the off-modality detector is *structurally blind*, not merely weaker, with
**zero cross-domain confounds**. Reframes the headline from "domain X wants detector Y" into the
sharper, non-circular **"the gap *definition* sets the modality; the off-modality detector is blind"**,
and demonstrates code hosts *both* gap modalities (unifying articles and code).

## Reuses / difficulty
`gap_recover` harness + Python data. New work: a semantic-void plant/region definition (mirror the
article book-surprise: F-density high, B-density low — reuse `artifacts_scripts/gap_detection/detectors.py`
level-set). **Difficulty: medium.**
