# Cache frozenset(members) for Jaccard dedup

Small optimization: `gap_article_as_seed.py`'s Jaccard dedup loop currently does `set(r["members"])` on every check, reconstructing the set from the list each time. Caching `frozenset(members)` once per accepted gap drops per-acceptance cost from `O(|results| × |G|)` to `O(|results|)` hash comparisons.

Time saving: ~3-5% of T10181 wall time (~24s of 640s). Not huge. Keep this as a cleanup item rather than a priority.

**Note**: currently the script switched to exact-match dedup (`gap_hashes: set[frozenset]`). If we switch back to Jaccard for the "true uniqueness" guarantee, this caching becomes relevant. See [exact-match-dedup-misses-near-duplicates](../../dead_ends/gap-detection-methods/exact-match-dedup-misses-near-duplicates.md) for the tradeoff.
