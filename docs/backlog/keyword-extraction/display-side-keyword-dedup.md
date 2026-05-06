# Display-side dedup of overlapping shorter-phrase keywords

In keyword extraction output, the production log-odds + lattice-merging pipeline can produce overlapping keywords like `{chinese_word_segmentation, segmentation, word_segmentation}` in the top-5 for a single gap. The token positions are not double-counted (lattice longest-match guarantees exclusive parsing at each position), so the underlying counts are correct. But the *display* is redundant — three keywords that all refer to the same concept at three granularities.

**Note**: this is a *display polish*, NOT a fix to the underlying scoring. Removing keywords would mean lying about the rankings the actual log-odds produced.

**Approach**:

```python
def deduplicate_overlapping_keywords(kws):
    selected = []
    for w, score in kws:
        # skip if w is a token-subset of any already-selected term
        if any(set(w.split('_')) <= set(s.split('_')) for s, _ in selected):
            continue
        selected.append((w, score))
    return selected
```

Then `chinese_word_segmentation` wins (highest log-odds), and `word_segmentation` and `segmentation` get dropped because their token sets are subsets.

**When NOT to do this**: when the redundant display is informative — e.g., when the user wants to see all three because each represents a different distribution of contexts.

**Priority**: low. Cosmetic. Keep as a per-display option, not as a default in the production pipeline.
