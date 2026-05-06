# OpenAlex topic purity as a corpus-QC axis

For multi-topic scopes: each gap's articles have OpenAlex `primary_topic_id` labels. A gap's `topic_purity = max_count / |G|` (concentration on the dominant OpenAlex topic) catches misclassification.

For single-topic scopes (T10181, T11303, T13734 in our experiments) this is degenerate (all members have the same topic_id by construction). It becomes useful at field/subfield scope where multiple topics coexist.

**Already computed for max-bookless** (`maximal_bookless_extractor.py` outputs `dominant_topic_id`, `dominant_topic_count`, `n_topics_spanned`). Trivial to add to `gap_article_as_seed.py` and the K-ball CSV.

**Why useful**:
- N3C clinical concept set cluster on T10181 spans multiple non-NLP topics that just happen to be misclassified into T10181; topic_purity catches it.
- Cross-domain leaks (signature verification appearing in NLP scope) get caught.

**Combine with** [vote-concentration filter](vote-concentration-as-book-exists-filter.md) for a multi-axis "is this a real research gap?" detector.
