# Multi-cluster softmax membership

Replace per-seed soft-EM (each basin has its own pairwise posterior `K_G/(K_G+K_B^local)`) with a single global model where every article's basin membership is a proper distribution from a softmax over its k nearest basin centroids — including book-anchored basins as fixed competitors. Designed to subsume most of the post-hoc enrichment pipeline.

## Why

The current pipeline accumulated post-hoc machinery because each enrichment step fixed something the discovery step couldn't address:

- `tau_supp = 0.5` because pairwise posteriors don't directly produce membership; we threshold them
- `--min-final-size 5` because soft-EM doesn't have a way to flag weak basins
- `gap_book_existence_filter` because soft-EM doesn't see books during expansion
- `gap_multi_scale_coverage` because soft-EM doesn't compare basins to each other
- `effective_uncovered_articles` because we need a coverage-aware ranking that combines two separately-computed quantities

Each delta makes sense independently, but the stack as a whole is duplicated work — discovery and labeling solve overlapping problems with disjoint machinery. A multi-cluster softmax with book-anchored basins as native competitors collapses most of these into direct readouts of one model's E-step.

## The model

For each article `a`, soft membership is a softmax over its k nearest basin centroids:

```
p(a ∈ G_k) = exp(log K̃_k(a) / τ) / Σ_j exp(log K̃_j(a) / τ)
```

over `j ∈ kNN(a) ⊂ {discovered basins} ∪ {book-anchored basins} [∪ {∅}]`.

**Per-basin density** (size-fair normalization — required to prevent bigger basins from systematically out-claiming better-fitting smaller ones):

```
K̃_k(a) = (1 / Σ_x w_k(x)) · Σ_x w_k(x) · κ(a, x)
```

— "weighted average kernel similarity to a member of basin `k`." Independent of `|G_k|`. `κ` is the existing self-tuning Gaussian kernel `exp(-‖a-x‖² / (σ_a σ_x))`; `w_k(x)` is the soft membership of `x` in basin `k`.

**Temperature** set by target perplexity: solve for τ such that average per-article entropy `H ≈ log P_target`. `P_target = 1.5–2` (each article commits to ~1.5–2 basins on average, allowing softness for genuinely disputed articles between adjacent basins). Same trick t-SNE uses; cheap 1D root-find.

**Book-anchored basins as fixed competitors.** Their `K̃_H` is computed from current member weights and stays in the softmax denominator for every `a`. Articles geometrically near a book-anchored basin softmax onto it preferentially; a candidate basin `G` overlapping a book-covered region naturally loses members to the book-anchored competitor and shrinks to its uncovered subset. Coverage handling is built into the model.

## Two variants: with ∅ vs. without

Both worth testing.

### Variant A: ∅-free

Drop the empty/background cluster. Softmax is over kNN basins only (discovered + book-anchored).

**Argument for:** every article was a seed; per-seed bootstrap covers article space by construction; even with dynamic centroids, an evicted article redistributes to its nearest basin rather than becoming homeless. Templated-leak detection, min-size filter, and outlier rejection are handled by post-hoc filters that already exist or aren't ∅'s job anyway.

**What you accept:** every article gets assigned somewhere, even articles in the marginal density of the topic that don't really "belong" to any specific basin. They get flat membership distributions across their kNN basins, but the assignment is forced. Whether this contaminates basins with noise is the empirical question.

**Pros:** simplest possible model. No ∅-kernel hyperparameter. Fewer moving parts to debug.

### Variant B: with ∅

Include a background cluster with kernel proportional to **marginal article density**:

```
K̃_∅(a) = (1/N) · Σ_y κ(a, y)
```

over all articles `y` in the topic. Interpretation: "the average kernel similarity of `a` to a randomly-drawn article in the corpus." A basin only earns `a` if its per-member kernel similarity exceeds the global marginal — i.e., it picks `a` out at above-chance probability. This is the standard log-likelihood-ratio-against-marginal that `compute_gap_keywords` already uses for keyword scoring (see `_keyword_helpers.py`, `BackgroundMode.MARGINAL`).

Articles with no specific-basin claim above marginal go to ∅ at high probability. Weak basins lose members to ∅ and shrink. Outlier articles are correctly absorbed by the null hypothesis.

**Pros:** principled treatment of "doesn't belong anywhere"; weak-basin pruning is automatic; templated-leak basins MIGHT lose density to ∅ if their per-member κ is low (artifacts have tight κ to OTHER members of the artifact, but possibly low κ to the rest of the corpus — empirical question whether this catches them; probably not, since artifacts are coherent in embedding space).

**Cons:** one extra computation per article (the marginal background, but cacheable since it doesn't depend on basins); subtle calibration (marginal density depends on topic size and density, may need normalization).

**Cheaper ∅-kernel alternatives** if marginal-density turns out to be expensive or unstable:
- Constant `log K̃_∅ = -C` with C tuned so ∅ absorbs the bottom ~5–10% of articles by best-basin fit. Loses spatial adaptivity.
- Self-tuning per-article density (use `a`'s own k-NN kernel mass as ∅'s claim). Per-article reasonable; less interpretable as a generative null.

Default to marginal density.

## What this obsoletes

| Current mechanism | Status under multi-cluster softmax |
|---|---|
| `tau_supp = 0.5` | Replaced by softmax distribution; argmax for crisp support, soft membership for everything else |
| `--min-final-size 5` | Variant B: ∅ absorbs weak basin members, basins shrink and vanish naturally. Variant A: still needed (small filter). |
| `book_anchored` flag | Implicit — basins win their own books in the softmax; the flag is just `argmax_a includes a book` |
| `inherited_coverage_median_post` | Direct readout: `median over G's members of Σ_H p(a ∈ H)` for `H ∈ book-anchored` |
| `effective_uncovered_articles` | Direct readout: `Σ_a p(a ∈ G) · (1 − Σ_H p(a ∈ H))`, both quantities from same softmax |
| `coverage_label_v2` (tier system) | Continuous, derived from softmax; tiers still useful for human-readable bucketing |
| `gap_book_existence_filter` (Phase 1) | Subsumed — book-anchored basins are first-class competitors, no separate filter |
| `gap_multi_scale_coverage` (Phase 3) | Subsumed — softmax IS the cross-basin reasoning |
| Per-seed `claimed` set / frozenset cache / claimed-overlap early-abort | Obsoleted (no per-seed loop in lockstep EM) |

## What stays orthogonal

| Mechanism | Why it remains |
|---|---|
| Templated-leak detector | Catches data-quality artifacts (Filipino name lists, N3C concept entries, etc.). These are GENUINELY tight clusters in embedding space — high κ to their members. Neither ∅ nor multi-cluster softmax catches them; the signal is title-pattern repetition / source homogeneity, not geometric. |
| Keyword pipeline (`compute_gap_keywords`) | Independent — operates on basin members' titles/abstracts |
| Self-tuning kernel σ (k-sigma) | Underlying kernel; reused as-is |
| HNSW / FlatL2 indices | Reused; the kNN-basin lookup is a new use of the same machinery |

## What the simplified pipeline looks like

**Current (8 stages):**
1. Per-seed soft-EM discovery (`gap_soft_em.py`)
2. `tau_supp` threshold for support
3. `min_final_size` filter
4. `gap_book_existence_filter` (Phase 1 enrichment)
5. Templated-leak detection
6. `gap_multi_scale_coverage` (Phase 3 inheritance)
7. Compute `effective_uncovered_articles` and `coverage_label_v2`
8. Rank by `effective_uncovered_articles`

**With multi-cluster softmax (3 stages):**
1. Lockstep multi-cluster EM with kNN softmax over `{discovered basins} ∪ {book-anchored basins} [∪ {∅}]`, per-member-normalized KDE, target-perplexity τ
2. Templated-leak detector (still post-hoc, reads basin members)
3. Rank basins by effective uncovered size — direct softmax readout

## Connections to prior work

- **Subsumes the lockstep-expansion backlog item** ([article-as-seed-lockstep-expansion](../gap-detection-methods/article-as-seed-lockstep-expansion.md)). That doc describes lockstep with same-cardinality competitor background; this is the same idea but with the natural softmax formulation rather than ad-hoc same-cardinality matching.
- **Generalizes the k-nearest-book-anchored counterfactual idea** (proposed in conversation 2026-05-08, not yet a backlog file). That's the k=2 special case: one G and one nearest-book-anchored H*. The full multi-cluster softmax handles arbitrary k and arbitrary basins as competitors.
- **Avoids the full-data soft-mixture posterior dead_end** ([full-data-soft-mixture-degenerate-bootstrap](../../dead_ends/cohesion-metric-design/full-data-soft-mixture-degenerate-bootstrap.md)). That failure mode used `K_G / K_total` over a fixed K-NN window without size-matching — the iteration became power iteration on the random-walk matrix and diffused to uniform. The fix in that doc is "size-matched K_B"; here we use per-member normalization on K_k for every basin, achieving the same anti-diffusion effect under the multi-cluster framing.
- **Avoids the high-dim-Gaussian-fit dead_end** ([centroid-bayesian-likelihood-high-dim-degenerate](../../dead_ends/cohesion-metric-design/centroid-bayesian-likelihood-high-dim-degenerate.md)). We use KDE likelihoods, not parametric Gaussians, so the `−(d/2) log σ²` normalizing-constant pathology doesn't apply.

## Test plan

### Phase 1: post-hoc reassignment experiment (~2 days)

Goal: empirically check whether the multi-cluster softmax reassignment materially differs from the current pipeline's basin set, before committing to lockstep restructuring.

1. Keep current per-seed soft-EM discovery on T11303 + T10181.
2. Identify book-anchored basins (existing `book_anchored == 1` flag).
3. For each article, compute the kNN basin centroids (k=10) over `{discovered basins} ∪ {book-anchored basins} ∪ {∅ if Variant B}`.
4. Compute `K̃_k(a)` per-member-normalized for each kNN basin.
5. Softmax with τ set by target perplexity (try `P_target ∈ {1.5, 2.0, 3.0}`).
6. Reassign each article's membership; rebuild basins from new soft memberships (basin = articles with argmax membership in this basin, OR all articles with `p(a ∈ G) > some threshold`).
7. Compare basin set to current pipeline:
   - Frozenset overlap (exact + Jaccard)
   - Coverage label distribution shift
   - "Most uncovered research" leaderboard differences
   - Any disputed-article cases that get resolved
8. Run both Variant A and Variant B; compare.

If Phase 1 shows the principled formulation either (a) produces materially different basins, OR (b) reproduces the current basins under simpler logic, proceed to Phase 2. If basins are identical, the current pipeline is a justified approximation and we keep it.

### Phase 2: lockstep EM from scratch (~1–2 weeks)

Replace `gap_soft_em.py` with `gap_lockstep_em.py`:

1. Initial basin seeds: every article (per-seed bootstrap, just like soft-EM).
2. E-step: each article's softmax membership over kNN basins (computed via FAISS centroid index on basin centroids).
3. M-step: each basin's centroid = weighted mean of soft members; weights = soft membership.
4. Confluence handling: at each step, check pairwise basin centroid distance and frozenset overlap; merge basins whose argmax-supports are equal or nearly so. Replaces the per-seed claimed-set / frozenset-cache machinery.
5. Repeat until basin centroids stabilize.
6. Output: basins with soft memberships, ready for templated-leak filter and ranking.

Drop `gap_book_existence_filter` and `gap_multi_scale_coverage` (subsumed). Keep templated-leak detector and keyword pipeline.

## Theoretical concerns to watch for

1. **Bootstrap dependence on book-anchored basin set.** Pass 1 needs book-anchored basins to define competitors, but book-anchored basins are themselves discovered. Solution: 2-pass scheme — pass 1 uses the current soft-EM book-anchored basins; pass 2 re-runs with the lockstep-discovered book-anchored basins; check stability across passes. Probably stable since big book-anchored basins are robust.

2. **Diffusion / runaway in book-poor regions.** Variant A (no ∅): in a region far from any book-anchored basin, an article-only basin can grow without geometric constraint. The local size-matched CSLS-balanced boundary in the current pipeline acts as a regularizer; we'd lose it. Variant B (∅) provides regularization via marginal density. If Variant A turns out unstable, Variant B is the intended fix.

3. **Multi-modal disputes.** When 3+ basins are roughly equidistant from an article, softmax membership is spread thin (low confidence each). Whether this is correct ("genuinely disputed") or pathological ("we don't know how to assign") depends on downstream use. Target-perplexity τ controls this — increase P_target to allow soft assignment.

4. **Per-seed bootstrap may not be needed.** If every basin is just a centroid that gets refined by E/M, you could initialize with a sparse set (e.g., HDBSCAN max-bookless cluster centers + book centroids) rather than every article. Cheaper. Worth testing.

5. **Templated-leak basins may not lose to ∅ even in Variant B.** Artifact clusters are GENUINELY tight in embedding space — they have high mutual κ. The ∅ kernel (marginal density) won't outclaim them. So the templated-leak detector remains necessary. This is fine; the design above explicitly notes templated-leak handling stays orthogonal.

## Open questions

- Is per-member normalization the right size-fairness, or should it be `1/sqrt(|support|)` (effective-size normalization, geometric-mean intuition)? Probably linear is fine.
- Does τ need to be per-article (each article has its own target perplexity tuned from local kernel mass), or is global τ enough? t-SNE uses per-point; we may not need it.
- For the kNN-basin lookup, what's `k`? `k=10` is a reasonable default; could be tuned. Larger k → more accurate softmax (less truncation bias), more compute.
- Does the templated-leak detector work better against the new basin set (cleaner tight clusters) or worse (basins are now coverage-aware so artifacts may be more visible, OR less so if artifacts get merged with real research)? Empirical question.

## Decision criteria

Phase 1 is cheap; run it first. Decide on Phase 2 based on:

- **Phase 1 basin overlap with current pipeline**: if ≥95% Jaccard match across most basins, the principled formulation reproduces existing results — no architectural change needed; maybe just adopt the cleaner ranking metric. If <80% overlap, the formulations are genuinely different and Phase 2 is justified.
- **Disputed-article resolution**: count articles where current pipeline assigns to multiple basins inconsistently vs. multi-cluster softmax assigns coherently. If this is a meaningful number (e.g., >5% of articles), Phase 2 is worth it.
- **Coverage label distribution**: does the new formulation produce a more interpretable real-gap / inherited-* distribution? Currently 0 real-gap basins on T10181 because everything inherits — a sharper threshold under the principled model might restore real-gap as a useful tier.
- **Implementation cost vs. simplification benefit**: Phase 2 is 1–2 weeks of work. Justified only if the simplification is real (drops ≥3 of: `tau_supp`, `min_final_size`, Phase 1 enrichment, Phase 3 enrichment).

## Files affected

If Phase 2 ships:

- New: `bookgap/gap_lockstep_em.py` or `artifacts_scripts/gap_lockstep_em.py`
- Deprecated: `artifacts_scripts/gap_book_existence_filter.py`, `artifacts_scripts/gap_multi_scale_coverage.py` (functionality moves into lockstep EM and the templated-leak detector)
- Updated: `artifacts_scripts/gap_method_comparison_table.py` (read direct softmax outputs instead of multi-stage enriched columns)
- `docs/coverage_design_tradeoffs.md` becomes a historical doc; new design replaces it
- `gap_soft_em.py` retained for comparison / fallback during transition

## Phase 1 result (2026-05-08)

**Verdict: Phase 2 justified by formulation.** See
[`docs/multicluster-softmax-phase1-results.md`](../../multicluster-softmax-phase1-results.md)
for the full report (12 configs × 3 crisp rules on T10181).

Headline numbers:
- Mean best-match Jaccard (old → new) is **0.146–0.172** across all
  one-shot configs, with **422/423 old basins below the 0.80 threshold**.
  The spec's "<80% Jaccard for ≥20% of basins" criterion clears by an
  order of magnitude.
- Variant A vs Variant B is essentially indistinguishable (Jaccard
  differs by < 0.005 across all cells). Perplexity sweep
  (P ∈ {1.5, 2.0, 3.0}) and crisp rule sweep (argmax / p > 0.5 /
  confidence-margin) are also second-order.
- All 6 iterated configs hit `max_iters=20` without converging
  (final_delta 0.75–0.99). The parent doc's "diffusion / runaway"
  prediction is confirmed for Variant A, but Variant B fails too.
- Top-20 leaderboard symmetric difference is **20/20** — completely
  disjoint. Caveat: old uses Phase-3 cross-basin posterior, new uses
  direct softmax readout — formulas differ by design.

**Refined readout from semantic spot-check.** The Jaccard 0.15
headline understates structural relatedness. Per-old-basin recall
against best-match new basin: median 0.72, ≥0.50 for 70% of old
basins. The new pipeline mostly **expands** old tight basins (~80
members, tau_supp ≥ 0.5) into broader natural partition basins
(~210 mean, up to 2698 articles), claiming all 88k articles vs
the old's 21k. The largest new basins are topically coherent
(formal linguistics, corpus linguistics, Japanese/Korean
linguistics), not garbage. For ~30% of old basins, members do
scatter into related-but-distinct topical framings (e.g.,
"LLM reasoning" → "combinatorial optimization") — real divergence,
but adjacent.

Phase 2 design notes derived from this run:

1. **Pick a basin granularity explicitly.** This is the most
   important design decision. The new pipeline's argmax-partition
   gives broader basins; the old's `tau_supp` filter gives narrow
   seeds. Phase 2 should expose this as a tunable
   (e.g., post-hoc `tau_supp` on softmax probabilities) rather
   than implicitly choose.
2. ~~**Keep ∅ active across iterations** in Variant B.~~ Re-examination
   showed ∅ is already active in every iteration (`log_K_empty` is a
   per-topic constant computed from the marginal density and passed
   unchanged into each E-step; dropping the ∅ column from M before
   re-feeding it is correct because ∅ has no embedding location).
   Not the bug. *(Removed.)*
3. ~~**Hold book-anchored basin centroids fixed** during EM~~
   Tested via the `--fix-book-anchored` flag and a 6-config rerun:
   final_delta still 0.93–1.00 (essentially scrambled), and at
   P=3.0 the fix slightly hurts (0.79 → 0.93). T10181 has only 63
   / 423 book-anchored basins, so 360 non-anchored basins still
   drift freely. *(Refuted.)*
4. **Alternatives Phase 2 should test for iteration convergence:**
   (a) hold *every* basin's core (M0 nonzero) members fixed across
   iterations — anchor every centroid to its soft-EM origin while
   allowing softmax to expand the periphery;
   (b) damped updates (M_iter ← α·M_softmax + (1-α)·M_iter_prev
   for α<1) to slow centroid drift;
   (c) accept that pure unsupervised lockstep EM doesn't converge
   in this formulation and ship the **one-shot variant only**
   (which converges trivially and produces sensible basins per the
   Phase 1 spot-check).

T11303 was **not run** — verdict on T10181 was unambiguous and the
sweep across all dimensions agreed. A second topic would be redundant
for the Phase 2 go/no-go decision.

The Phase 1 implementation lives in
`artifacts_scripts/gap_softmax_reassign.py` and
`artifacts_scripts/gap_softmax_compare.py` and is retained for use as
a reference / regression test against the Phase 2 lockstep EM.


## Phase 2 shipped (2026-05-09)

Implemented in `artifacts_scripts/gap_softmax_enrich.py`. The new
driver replaces `gap_book_existence_filter` + `gap_multi_scale_coverage`
with a single one-shot multi-cluster softmax refinement layer over
soft-EM's basin set. Soft-EM (per-seed bootstrap discovery) is kept
unchanged as the upstream input.

Key implementation choices, all derived from Phase 1 evidence:

- One-shot only (no iteration; Phase 1 showed pure unsupervised
  lockstep EM diverges).
- Variant A only (no ∅ class; Phase 1 showed Variant B is
  indistinguishable from A in one-shot).
- `tau_supp` exposed as a tunable knob, default 0.5 (narrow seeds,
  backwards-compatible).
- 7-tier `coverage_label_v2` retired; downstream consumers compute
  their own categorical buckets from the raw flags + continuous
  `inherited_coverage_median_post`.
- Templated-leak signal kept as a tagger (flagging, not dropping).

See [`docs/superpowers/specs/2026-05-09-multicluster-softmax-phase2-design.md`](../../superpowers/specs/2026-05-09-multicluster-softmax-phase2-design.md)
for the spec and [`docs/superpowers/plans/2026-05-09-multicluster-softmax-phase2.md`](../../superpowers/plans/2026-05-09-multicluster-softmax-phase2.md)
for the implementation plan.
