# Boundary-centroid cosine as a quality axis

One of the orthogonal "semantic" axes proposed during the cohesion-axis discussion: `cos(centroid(gap_articles), centroid(boundary_books))`. High cosine = boundary books are topically aligned with the gap's topic; low cosine = boundary books are random NLP titles (corpus-noise signature).

**Direction matters** (correction made during conversation): high boundary_centroid_cos means books exist on the topic, which is *less* gap-like, not more. So multiplying by `boundary_centroid_cos` would punish real gaps. Correct usage: a *separator* axis between "real gap with adjacent book" (high) and "templated-data leak" (low) — but in opposite directions for the same observation.

For corpus-QC: low boundary_centroid_cos + high cohesion + low topic_purity = strong signal of templated-data leak (N3C, Filipino names, Tibetan folktales, LMF dictionaries on T10181).

**Subsumed by** the discrete [book-seeded confluence filter](../gap-detection-methods/article-as-seed-with-book-seeds.md) when that's available — a cluster matching a book-seeded cluster *necessarily* has high boundary-cos. Worth comparing the continuous heuristic against the discrete filter to see how much they agree.

**Implementation**: cheap. For each gap, compute `centroid(emb[gap_members])` and `centroid(emb[boundary_books])`, then cosine. Boundary books are already 1-NN per gap article in the local-bg construction.
