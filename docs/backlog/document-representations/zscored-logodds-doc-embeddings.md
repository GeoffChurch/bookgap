# Z-scored log-odds document embeddings (Fightin' Words per-doc)

A principled sparse alternative to TF-IDF derived from the same statistic that makes `keyword_extraction.py`'s `LOG_ODDS` work so well at the topic level.

## Idea

Treat each document $d$ as a "topic of size 1." For every word $w$ in $d$, compute the Dirichlet-smoothed log-odds-ratio vs the corpus prior:
$$\delta_{d,w} = \log\frac{P(w|d)}{1-P(w|d)} - \log\frac{P(w|\text{corpus})}{1-P(w|\text{corpus})}$$
Z-score by the analytic binomial variance, threshold at $|z| > \tau$, output the sparse vector. That's the document embedding.

## Why it might work

The variance normalization is the killer feature of Fightin' Words — it downweights low-count terms statistically rather than via TF-IDF's heuristic "discount common words" logic. Per-document, the same correction would give a sparse vector that's already IDF-adjusted *and* reliability-adjusted, in one closed-form step.

TF-IDF can be reframed as a crude version of this against a uniform prior, without variance correction. Bookgap's empirical finding that log-odds beats TF-IDF for topic keyword extraction is direct evidence that the same fix should pay off per-document.

## Concrete variants

- **Direct sparse embedding**: cosine on the z-vector. Compare against TF-IDF cosine for nearest-neighbor / clustering / gap-detection-input baselines.
- **CCA bridge to GIST/Gemma**: learn a linear map from the sparse z-vector to the dense embedding (or vice versa). Lets us combine an interpretable Fightin'-Words signal with the dense transformer signal. Falls in the same family as SPECTER's citation-text bridge but with log-odds-text instead of citations.
- **Background choice**: corpus marginal (default) vs same-topic background (in-topic discrimination) vs same-field background. Mirrors the existing `BackgroundMode` enum in `keyword_extraction.py`.

## Connection to existing work

The continuous-Gaussian analog is **Fisher's linear discriminant** (LDA): $v_{\text{Fisher}} = \Sigma^{-1}(\mu_+ - \mu_-)$. Difference-of-means is the log-odds direction; $\Sigma^{-1}$ is the variance whitening. The fact that ANTI_LDA is among the worst denoisers in the bake-off and Fightin' Words is among the best keyword extractors is the same observation viewed twice: Fisher's direction is where the class-discriminative signal lives — throw it away for gap detection, keep it for characterization. See [[topic-fisher-signatures]] for the embedding-space sibling.

## Cost

~50 LOC for the sparse-vector path (reuse `compute_log_odds_ratio_docs` from `keyword_extraction.py`). ~150 LOC for the CCA bridge variant. Cheap to prototype.

## Priority

Medium-high. The empirical evidence from keyword extraction is unusually strong, and replacing TF-IDF in any bookgap baseline that currently uses it would be a clean improvement with a defensible motivation.
