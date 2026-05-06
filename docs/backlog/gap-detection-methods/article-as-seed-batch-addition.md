# Batch addition (multi-add per step) for cohesion expansion

**Idea**: at each expansion step, add ALL candidates with `post(G ∪ {c}) ≥ τ` rather than just the single nearest one. The strict moving-centroid policy currently adds at most one candidate per step (top=1).

**Why this might help**:

1. **More deterministic given G**: with single-add, two seeds whose centroids slightly differ at step k will add slightly different next-articles, then drift further. With batch-add, both seeds adding *all* qualifying candidates at step k means their G's grow much more similarly, which dramatically increases the chance of confluence (set equality) at later steps.

2. **Faster convergence**: a batch-add expansion takes O(natural_size / batch_size) steps instead of O(natural_size). Per-step cost rises (more candidates to evaluate) but total iteration count drops.

3. **Resolves the path-dependence issue more directly than [canonicalization](article-as-seed-canonicalization.md)**: instead of post-processing fix-points, prevent path divergence at the source by removing the artificial single-pick ordering.

**Mechanics**:

```python
# Current strict-single:
j = candidates[0]
G_test = G + [j]
if min_post(G_test) >= τ: G = G_test else stop

# Batch:
qualifying = [c for c in candidates if min_post(G ∪ {c}) >= τ]
if not qualifying: stop
G = G ∪ qualifying  # add all at once
```

**Subtlety**: when adding multiple candidates simultaneously, evaluating `min_post(G ∪ qualifying)` (with all of them added) is different from each candidate's individual `min_post(G ∪ {c})`. Some candidates that pass individually might collectively cause min_post to drop below τ. Three resolution strategies:
- Add all individual-passing candidates greedily, accept collective drop
- Add only the subset whose collective min_post stays ≥ τ (combinatorial; needs a heuristic)
- Add candidates ordered by their individual post, stopping when collective min_post would drop

**Pairs with**:
- [article-as-seed-better-inclusion-policy](article-as-seed-better-inclusion-policy.md) — batch addition is a member of that family of tweaks.
- [article-as-seed-canonicalization](article-as-seed-canonicalization.md) — batch addition makes paths more deterministic, which makes canonicalization converge faster (or potentially makes it unnecessary if confluence becomes near-perfect).

**Test plan**:
1. Implement the simplest variant (greedy add-all-individual-passing).
2. Run on T11303 unfiltered; count distinct exact-match basins. If 789 → much smaller (closer to 254), that's a strong validation.
3. Compare wall time and per-step cost vs single-add baseline.
