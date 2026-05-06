# Vote-concentration as a "book already exists" filter

For each gap, compute the most-voted nearest book (by counting 1-NN-book votes across gap articles). The combination `(top_book_votes / |G|, top_book_title_topical_alignment)` acts as a corpus-QC + textbook-existence detector:

- **High vote concentration (>50%) + topically-aligned book** = topic exists, has a book → filter as "false gap"
- **High vote concentration + tangential book** = topic has an adjacent book (e.g. CJK seg → Myanmar seg) → keep with caveat
- **High vote concentration + random book** = templated-data leak (N3C concepts → "vocabulary pocket guide")
- **Low vote concentration** = diffuse subfield; legitimately fragmented landscape

Empirical evidence (T10181, top-20 K-ball): 5/20 gaps have a directly-titled book covering the topic (Reordering, Dependency Parsing, Morphology, MT Evaluation, WSD), with vote concentrations 25-65%. These are exactly the "false gaps where a textbook exists" cases.

**Implementation**: already computed in `gap_method_comparison_table.py` (column "top books (×votes)"). Could become a CSV column on every gap and a default sort/filter knob.

**Comparison to [book-seeded confluence filter](../gap-detection-methods/article-as-seed-with-book-seeds.md)**: that's a discrete binary "book covers this cluster" detector based on cluster-set equality. The vote-concentration approach is a continuous heuristic that doesn't require expanding from book seeds. They're complementary — vote-concentration is cheaper but coarser; book-seed confluence is structurally cleaner but needs canonicalization.

**Title-topical-alignment** is the missing piece: requires either a small LLM check ("does book title 'Word-Sense Disambiguation' cover topic represented by keywords {sense, disambiguation, word, wsd}?") or a heuristic like "do top-3 keywords of the gap appear in the top book's title?" A simple substring/keyword-overlap heuristic would catch most cases.
