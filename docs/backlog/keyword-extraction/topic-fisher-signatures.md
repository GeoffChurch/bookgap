# Topic-Fisher signatures (continuous Fightin' Words for embedded docs)

Continuous-embedding analog of `keyword_extraction.py`'s topic-level log-odds-ratio. Uses the Fisher discriminant direction between in-topic and out-of-topic document embeddings as a topic signature; attributes back to words via word-embedding cosines for an interpretable rendering.

## Idea

For each topic $t$:
1. Let $X_t$ = embedded docs in topic $t$, $X_{\neg t}$ = docs outside.
2. Compute Fisher direction $v_t = \Sigma^{-1}(\mu_t - \mu_{\neg t})$, where $\Sigma$ is the pooled (or ridge-regularized) within-topic covariance.
3. Project documents onto $v_t$ → continuous "characteristic-of-$t$" score per doc.
4. Optionally render as keywords: project the existing word-embedding matrix onto $v_t$, take top-$k$.

## Why it might work

The current `summarize_topics.py` works on word-counts via Fightin' Words. That captures statistical reliability for words that *literally appear* but misses synonymy / paraphrase. The Fisher direction in embedding space inherits the same "discriminative axis with variance correction" structure as Fightin' Words (see [[zscored-logodds-doc-embeddings]] for the discrete-Gaussian connection), but operates on continuous semantic content — so "deep learning" and "neural networks" pull together rather than competing for keyword slots.

## Concrete variants

- **Vanilla Fisher**: as above.
- **Ridge-regularized**: $\Sigma + \varepsilon I$, mirroring the smoothing that makes Fightin' Words behave for small counts. Standard regularized LDA.
- **Multi-topic**: stacked $S_B / S_W$ over all topics simultaneously → multiclass LDA projection; first $k$ directions are a low-dim topic-signature space.
- **Word attribution via integrated gradients** instead of direct cosine — handles cases where the embedded representation is non-linear in token presence.

## What's notable

- ANTI_LDA is among the worst denoisers in the bake-off precisely because Fisher's direction has so much class-discriminative juice. **Use it here, don't project it out.**
- Doesn't require any new ML infrastructure — GIST/Gemma embeddings already cached; LDA is closed-form.
- Could replace or supplement the existing word-level log-odds in `summarize_topics.py`; orthogonal data sources (lexical vs semantic) so a *combined* score is also worth testing.

## Cost

~80 LOC for the Fisher computation + scoring; ~30 LOC for word-attribution rendering. Reuses `bookgap.cloud_denoiser.LDADenoiser` plumbing (or write a direct version since we want the projection *kept*, not discarded).

## Priority

Medium. Mostly an improvement to existing curriculum output rather than a new capability, but the underlying connection (Fishers ↔ Fightin' Words) is theoretically pretty.
