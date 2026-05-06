# Higher max_iters for book-poor topics

The default `max_iters=500` translates to a maximum gap size of 501 articles. T13734 (Applied AI, 0.25% books) hit this cap on every top-30 gap because cohesive regions are genuinely larger than 500. T10181 and T11303 rarely hit it.

**Why this matters**: when a gap is truncated at the cap, downstream metrics misrepresent it (the *natural* cluster size and its true effective_size are larger). Cross-topic comparisons become apples-to-oranges.

**Cost**: each iteration's `compute_state` call scales linearly with |G|, so total iteration cost is O(|G|²). Doubling max_iters quadruples compute. K_max=1000 → ~4× wall time on book-poor topics.

**Options**:
- Adaptive cap based on book density: `max_iters = max(500, 0.05 × N_articles)`. Book-poor topics get more headroom.
- Hard cap kept; report whether each gap hit the cap (already in output) and surface that as a quality flag.
- Two-pass: run with max_iters=500 to discover most basins; for those that hit the cap, optionally re-run with max_iters=2000 to find natural sizes.

**Test plan**: run T13734 with `--max-iters 1500`. Confirm whether top gaps stop at natural sizes < 1500 or continue to hit the new cap.
