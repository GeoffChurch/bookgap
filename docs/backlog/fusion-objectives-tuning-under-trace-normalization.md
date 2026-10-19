# Do trace-normalized fusion objectives, once tuned, still tend towards pure locality?

The paper used to say that "the fusion objectives tended towards LPP under hyperparameter optimization", and built an interpretation on it: that citation locality handles the other unwanted dimensions best inside a single GEP quotient, rather than in a linear combination or a JAD problem.
Both were cut in October 2026 (GeoffChurch/bookgap-paper 7b39e93), because no sweep behind the claim is reported anywhere, and because it is probably an artifact.

## Why it may be an artifact

Before the trace-normalization fixes (`normalize_traces` in `bookgap/denoisers.py`; the 2026-05-26 sweep in [denoiser-bakeoff-expansion](denoiser-bakeoff-expansion.md) is the first run after them), the raw tr(S_L) was about six orders of magnitude larger than tr(S_B), tr(S_T) or a class's scatter.
Any weighted combination was then dominated by the Laplacian term whatever its nominal weights, so every fusion behaved like a locality objective, and a tuner would find "LPP-like" optima for free.
The bake-off doc records the same effect for one arm: CCLIP at α = 0.01 was a historical Gemma top performer "under the bug, so α was effectively 1".

The October 2026 regeneration, with trace normalization, shows no such tendency on its coarse α grid: CCLIP's best α is 0.5 for GIST's proposal gap percentile and 0 for Gemma's target gap coverage (the paper's Table II, `artifacts/latex_tables/gap0/articles_10_books_5/combined_best_denoisers.tex` in GeoffChurch/bookgap-paper).
No denoiser is significantly better than LPP there, but nor does locality win the tuning.

## What to run

- Sweep CCLIP's α over a fine grid (0, 0.1, …, 1) with `normalize_traces=True`, on both embedders, at the paper's locked k (10 for GIST, 394 for Gemma) and default 10 articles + 5 books.
- Do the same for JAD's weights (`JADDenoiser(wt, wl, w_conflate, normalize_traces=True)`), on a small simplex grid.
- Choose α on one set of topics and score it on another (not `TOPIC_SUBSET_RANDOM_20` alone), so "where the tuner lands" is not fitted to the scoring topics.
- Report the paired intervals against LPP with `artifacts_scripts/paper/denoiser_difference_cis.py`.

Cost: each cell is about 20–30 GPU-minutes on 20 topics (from the October runs), so an 11-point α grid is about 4–6 GPU-hours per embedder.

## Outcomes

- **Optima near α = 1, held out:** the claim can return to the paper, with this sweep as its evidence.
- **Optima spread across α, or near 0:** move this file to `docs/dead_ends/` with the numbers, so the claim is not re-asserted from memory.
