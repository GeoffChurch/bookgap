# Curriculum clustering notes

Lessons from the 2026-04-25 series of full-corpus runs (`degree_epsilon=0.0,desired_n_topics=15__22..__24`).

## The singleton-blob failure mode

Default `linkage_method=bisecting_kmeans` on the post-PageRank top-5000
produces an output dominated by:

- **One giant residual cluster** holding 60-100% of the population
  (e.g. 3,684 of 5,000 in `__23`, 4,957 of 5,000 in `__22`).
- **Several singletons** corresponding to high-relevance works
  (Shannon's secrecy paper, EPR, von Neumann, Hilbert-space text) —
  not because they're the curriculum's best topic anchors, but because
  the bisection's variance criterion peels them off one by one as
  the most-distant-from-the-residual works.
- **A handful of small / medium clusters** in between.

This isn't a useful topic structure: most of the curriculum is in
the undifferentiated blob, and the singletons are isolated rather
than anchoring meaningful topic neighbourhoods.

## Why it happens

`bisecting_kmeans` is unweighted: it splits to minimise unweighted
variance. A high-relevance "outlier" in embedding space (Shannon's
1949 paper sits far from quantum-cryptography neighbours) drives the
bisection criterion, getting peeled off into its own one-element
cluster. The bulk that's left has no remaining outliers strong
enough to drive further useful splits, so subsequent bisections just
shave off small clusters from the surface.

The PageRank step *already produces a relevance vector* — the
default bisection ignores it.

## Fix: `weighted_bisecting_kmeans`

`scalable_clustering.bisecting_kmeans_linkage(sample_weights=…)`
plumbs sklearn's `sample_weight` through KMeans / MiniBatchKMeans
and uses weighted centroids + weighted "size" in the Ward-like merge
distance. `build_curriculum.py` exposes this as
`linkage_method="weighted_bisecting_kmeans"`; weights are
`full_point_masses[top_k_indices]`.

Effect: high-relevance works anchor cluster centres instead of being
peeled off, and the bisection criterion subdivides the residual
blob around its internal relevance peaks.

## Empirical comparison

Same forward-only run on 3 quantum-crypto target works, no
`source_work_ids`, top-5000 ranked, 15 topics:

| metric | unweighted (`__23`) | weighted (`__24`) |
|---|---|---|
| singletons | 5 / 15 | **4 / 15** |
| max cluster size | 3,684 | **2,493** |
| std dev of size | 902.2 | **629.4** |
| largest two clusters | 3,684 + 344 | 2,493 + 930 |
| effective # works | 275 | 275 |
| entropy (% of max) | 32.0% | 32.0% |

Same overall mass distribution; the difference is *how cluster
boundaries are drawn*, not how mass concentrates.

### Earned vs unearned singletons

Unweighted's 5 singletons were a mix of "earned" (target works) and
"unearned" (peeled-off seminal anchors that should have anchored
clusters around them). Weighted's 4 singletons are all earned:

1. **EPR paper** — the foundational philosophical paper, sufficiently
   distinct in embedding space that no in-corpus neighbour sits
   close enough to cluster with it. Sim→target = 0.762.
2. **Cryptography in an Algebraic Alphabet (1949)** — a unique
   classical-crypto thread; no clustering opportunity in top-5000.
3. **"Quantum cryptography" review** — sim→target = 1.000 (this *is*
   one of the input target works). Tautological singleton.
4. **Communication via one- and two-particle operators** —
   sim→target = 1.000. Another input target.

### Cluster anchors that did emerge under weighting

Works that the unweighted run had as either singletons or buried in
the residual blob, now anchoring meaningful clusters:

- **Mathematical Foundations of QM** (von Neumann) — 2,493 works
- **Hot Electrons in Germanium and Ohm's Law** — 930 works (whole
  solid-state-physics cluster previously absent)
- **AIP Handbook** — 483 works (physics reference cluster)
- **Introduction to Hilbert Space** — 412 works (was 344 unweighted)
- **Communication Theory of Secrecy Systems** (Shannon) — 45 works
  (was a singleton in `__22` bidirectional run)

## Open follow-ups (not yet tried)

- **Leiden on citation subgraph of top-5000.** Citations are the
  semantic relation curriculum *means*, not embedding distance.
  Should be the highest-quality topic structure but requires
  ~30-50 lines + `igraph` / `leidenalg` integration. Worth doing
  as a parallel approach for comparison; see if the seminal
  works become natural community hubs.
- **HDBSCAN on top-5000 with relevance-weighted point replication.**
  Density-based, naturally variable cluster sizes, no forced
  cluster count. Would put high-mass works in dense regions.
- **Hierarchical agglomerative average-linkage.** Different shape
  bias than bisecting; control experiment.
- **Re-examine `min_cluster_size>1` under weighted_bisecting.**
  Originally rejected because raw `min_cluster_size=20` polluted
  earned singletons with irrelevant neighbours. With weighted
  bisecting the singleton-vs-cluster decisions are different;
  the original objection may not hold.

## Linearity caveat (already documented in `pagerank.py`)

The PR step uses sink-mass redistribution proportional to seeds,
which makes power iteration bilinear in (v, seeds). At full corpus
scale `n_sinks=0` (citation graph has self-loops), so the deviation
is zero in practice; relevant only on artificial sub-graphs.
