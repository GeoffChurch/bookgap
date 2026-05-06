# K-ball vs HDBSCAN under statistical scoring

(Migrated from `artifacts_scripts/curriculum/TODO.md` "Gap detection — state of the evidence" section.)

`bookgap/depletion_test.py` factored binomial-p / Benjamini-Hochberg FDR / effect-size scoring out of `hdbscan_gap_probe.py`. HDBSCAN probe and K-ball probe both write per-gap CSVs at `artifacts/{hdbscan,kball}_gap_probe/{scope}__{denoiser}__drop{drop_k}.csv`.

**K-ball's caveat**: K-ball gaps are by construction book-free spheres (radius stops at the nearest book), so binomial-on-occupancy is degenerate ("always extreme"). K-ball's native gap-significance is the percentile/zscore/Shapiro framework against the book-as-query null distribution. Cross-algorithm comparison happens at "qualifying gap counts" rather than per-gap p-values.

**Status**: code shipped 2026-04-29, sweep partially run on subfield 17/1702 (results in TODO.md). Open items:
- Cross-scope replication (current results are 17/1702 only).
- HDBSCAN `min_cluster_size` tuning (default 50 is high for the scope; lower may surface more clusters and change the picture).

**See also**: [cohesion-cut on SLT](cohesion-cut-on-slt.md) — an alternative HDBSCAN-style approach using cohesion as the cut criterion rather than statistical depletion.
