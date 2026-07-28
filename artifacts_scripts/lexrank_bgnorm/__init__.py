"""bg-norm LexRank extractive-summarization experiment (CPU-only, self-contained).

Tests the falsifiable hypothesis from ``docs/backlog/curriculum-applications.md``
("The bipartite sentence-word construction"): Fightin'-Words-corrected sentence
centrality (``p^0.5 * log((p+eps)/(g+eps))`` on the two-step sentence-word
Gramian walk) beats plain LexRank at extractive summarization, scored by ROUGE
against Wikipedia lead sections.

Modules:
- ``textproc``  — wikitext finishing cleaner, lead/body splitter, sentence
  splitter, tokenizers (vendored; the host env has no mwparserfromhell/nltk).
- ``rouge``     — vendored ROUGE-1/2/L F1 (the common simplification: no Porter
  stemmer, flat-sequence LCS; the env has no rouge_score package).
- ``engine``    — sparse B matrix, two-step stochastic Gramian walk, centrality
  via ``curriculum.core.pagerank.compute_pagerank``, bg-norm variants, baselines.
- ``run_eval``  — the experiment CLI (data prep, methods, scoring, bootstrap,
  report). Outputs land in ``artifacts/lexrank_bgnorm_eval/``.

This package is deliberately separate from ``curriculum_lean`` (the Lean
experiment); the only cross-import is the tracked, frozen
``curriculum.core.pagerank``.
"""
