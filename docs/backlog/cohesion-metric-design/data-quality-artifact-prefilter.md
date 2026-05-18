# Data-quality-artifact prefilter for refined gaps

## Idea

Before ranking refined gaps, drop the ones that are structurally
"data debris" — clusters of records that are dense in embedding
space because they share a fixed metadata template, not because they
represent a real research subarea. A cheap structural-fingerprint
check on member titles catches most known classes.

## Why

`effective_size` ranking does well at the top of the leaderboard
(0 leaks in top-10 across T11303, T10181, T10462 — see the M9
discussion in the cell-type integration design doc), and an
`effective_size >= 30` floor handles the smallest artifact clusters.
But:

- **Medium-sized artifact clusters slip through the floor.** The
  most striking example is the N3C drug-ontology cluster on T10181
  (`effective_size = 76`), which ranks #1 under `median_logit`
  and #2 under `median_logit × effective_size`. Posterior-based
  signals can't distinguish "76 distinct drug-concept records that
  share a title prefix" from "76 distinct papers about a real
  subarea."
- **Adding preprints to the ambient set introduces new artifact
  classes.** The May 14 chapter-only experiment surfaced a
  taxonomic-species-description cluster on T11303
  (`Spathelia simplex Linnaeus 1762`, `Olallamys albicaudus`,
  `Trichrysis tonkinensis`, ...) that wasn't visible when the
  ambient set was `article`-only. This was a preprint-deposit
  pattern (Zenodo-style taxonomy publications). Other topics may
  have other preprint-specific artifact classes we haven't seen yet.

The structural fingerprint is the same across known artifact
classes: a fixed prefix or grammatical template shared by most
members. That's directly testable on titles without invoking the
posterior.

## Concrete artifact classes seen

Documented for grounding the detector design:

| Class | Example titles | Pattern |
|---|---|---|
| Benchmark records | "Benchmark run results by Pratanu Mandal, on benchmark context Tutorial: Static Causal Discovery v2" (×5) | Fixed long prefix |
| Bilingual-dictionary metadata | "French-Catalan LMF Apertium Bilingual dictionary", "Esperanto-Spanish LMF Apertium..." | Fixed suffix + structured permutation |
| Concept-set dataset | "N3C Concept Set - 423850600 (sulfonylureas)", "N3C Concept Set - 189643838 ([N3C] REMDESIVIR)" | Fixed prefix `N3C Concept Set -` |
| Curriculum solution dumps | "[PDF] NCERT Solutions for Class 3 English Unit-7 Poem Puppy and I" | Fixed prefix `[PDF] NCERT Solutions for Class` |
| Bare person-name records | "Marquez, Gelacio", "Cagaoan, Luis", "Ridoloso, Venancio" | Comma-separated surnames, no real title text |
| Serial-content series | "Crazy Sequential Representations: Fill the Gaps (03)", "...Base 17 up to 36" | Fixed prefix + numbered variant |
| Taxonomic descriptions | "Spathelia simplex Linnaeus 1762", "Thaumapsylla wilsoni Hastriter 2009, sp. n." | Grammatical template `<Genus species> <Author> <Year>` |

## Sketch — cheap title-structure signal

Per gap, compute one or both of:

1. **Top-3-token shared-prefix fraction.** For each member title,
   take the first 3 tokens (lowercased, punctuation stripped). The
   fraction of members whose first-3 tokens match the gap's modal
   first-3-token tuple. Catches prefix-templated classes (Pratanu
   Mandal, N3C, NCERT, Crazy Sequential).
2. **Median pairwise title-token Jaccard.** Compute Jaccard on
   tokenised member titles between random pairs; take the median.
   Real research subareas score ~0.05-0.2; templated classes score
   > 0.4. Catches structural-template classes (Apertium dictionaries,
   taxonomic descriptions) that don't share a literal prefix but
   share most tokens.

A gap is flagged as an artifact if either signal exceeds a
threshold (e.g. prefix-fraction > 0.5 or median-pairwise-Jaccard
> 0.4). Costs: O(k\* · 3) for prefix extraction; O(k\*²) per gap
for pairwise Jaccard, but k\* is small (mostly < 200) and the
operation is per-gap-not-per-member so it's cheap relative to
phase-3 NN queries.

Person-name records (T10181's Filipino-surnames cluster) need a
different signal — maybe "average title token count < 3 and no
title contains a common-word verb form." Or just live with them
since they're rare.

## Out of scope (vs the LLR_REFINED integration)

The integration design at
`docs/superpowers/specs/2026-05-13-cell-type-llr-refined-integration-design.md`
keeps this prefilter **out of scope** for the initial integration.
Rationale: M9 ranking with `effective_size >= 30` handles the
visible artifact classes on the three evaluated topics
(0 leaks in top-10 in two of three; 1 in the third). The prefilter
is the long-term principled answer for medium-sized artifact
clusters but isn't required for the integration to deliver value.

Promote to in-scope once a concrete case at the official-pipeline
scale demonstrates the eff_size floor isn't enough.

## Open questions

- **Per-class thresholds vs single threshold?** Different artifact
  classes have different fingerprints (prefix-fraction is high for
  N3C but moderate for taxonomy; pairwise-Jaccard is the reverse).
  Probably easier to OR over a handful of class-specific signals
  than to tune one global threshold.
- **Should the prefilter run before or after dedup?** Before is
  cheaper (fewer gaps to fingerprint); after is more correct (dedup
  may have merged artifact clusters into real gaps as a minority).
  Tentatively: after dedup, so the prefilter sees the final shape.
- **Filtered or flagged?** Flagging (`is_data_artifact: bool` field
  on `LLRRefinedGap`) is cheaper and reversible; filtering removes
  them from downstream LLM batch entirely. Probably flagging at the
  algorithm layer and filtering at the `GapWithContext` layer.

## Related

- Examples surfaced during the May 14 ranking-variant evaluation
  (`artifacts_scripts/gap_llr_rank_variants.py`).
- The taxonomic-species class surfaced specifically when including
  preprints in the ambient set; see the chapter-only experiment in
  `artifacts/gap_kball_llr_refine/chapter_only/`.
- Conceptually related to `boundary-centroid-cosine` in this
  directory (also a per-gap quality signal computed from member
  geometry) but operates on titles rather than embeddings.

## Status / evidence (2026-05-16, denoiser bake-off)

Reinforced, not yet implemented. The denoiser bake-off
(`docs/denoiser_bakeoff_results.md`) showed `pca`/`cclip_*` arms
repeatedly surface the **N3C Concept Set** dataset-record cluster and
monolingual (Korean/French) language clusters in their top-10 by
`effective_size`. Two findings:

- It is **genuine tight data-debris, not a recall artifact**: the
  finding was byte-stable under the corrected high-recall HNSW params
  and under exact-FlatL2.
- The **binomial depletion p-value is a usable interim discriminator**
  — N3C clusters score p≈0.3 (non-significant) even when their
  `effective_size` is top-ranked. So ranking/filtering by the binomial
  p (or this prefilter) works; ranking by raw `sig_count` /
  `effective_size` does not. This is the strongest argument yet for
  the prefilter (or at minimum a p-gate) before any size/significance
  ranking is trusted.
