# Article-as-seed with book-seeded clusters as filter

Expand from every article AND every book as a seed; allow confluence (set equality merging) across both. Tag each canonical cluster as "book-anchored" if any of its merged seeds was a book; "article-only" otherwise. Drop book-anchored clusters from the gap pool — these are precisely the topics where a book canonically anchors the cluster.

**Why this is elegant**:
1. Unified treatment: gaps and book-territories are the same kind of object, just labeled differently. Same algorithm, same confluence, same metric.
2. **Discrete book filter** instead of the continuous "vote-concentration" heuristic ([see vote-concentration backlog](../cohesion-metric-design/vote-concentration-as-book-exists-filter.md)): you get a clean binary "this cluster has a canonical book" or it doesn't.
3. **Subsumes the boundary-book signal**: the local CSLS-balanced background was an *indirect* way to ask "what books are nearby"; this asks the question by giving each book its own expansion. If book b's natural cluster IS the same as article-cluster X, b is X's anchor — explicit.
4. **Catches the K-ball false-gaps**: the ~25% of K-ball top-20 with directly-titled boundary books (Reordering → "Distortion Constraints in SMT", Dependency parsing → "Dependency Parsing", Morphology → "Morphology", MT Evaluation → "Machine Translation Evaluation", WSD → "Word-Sense Disambiguation") would all collapse via book-seed confluence into the corresponding book's cluster and be filtered out.

**Mechanics**:
- Book seeds expand to clusters of *articles only* (the book itself isn't a member of G; it's just the seed that initializes G with its nearest article)
- Article seeds expand the same way
- Confluence: set equality (or fixed-point equality if combined with [canonicalization](article-as-seed-canonicalization.md)) merges clusters
- Final classification: any cluster whose confluence chain includes a book-seed is "book-anchored" → not a gap

**Inherits path-dependence**: without [canonicalization](article-as-seed-canonicalization.md), exact-match book-vs-article merging fails just as the article-vs-article version does. So this proposal *needs* canonicalization to work cleanly.

**Test plan**:
1. Prototype canonicalization first ([backlog item](article-as-seed-canonicalization.md))
2. Add book-seeds to the same pipeline
3. Compare: which K-ball false-gap-by-book entries (Reordering, Dependency Parsing, Morphology, MT Evaluation, WSD) get filtered out via book-seed confluence?
4. How many additional clusters are discovered as "book-anchored" beyond the K-ball top-20?

Cost: adds N_books seeds (~1,250 for T10181) — small overhead, ~1.4× more seeds. Books with no real article-territory expand to tiny clusters and are ignored.
