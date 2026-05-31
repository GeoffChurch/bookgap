"""Minimal curriculum core: rank (PageRank) -> select (top-N) -> order.

Depends only on numpy / scipy / scikit-learn. Works with NO embeddings — the
clustering/keyword/quality/judge layers under ``curriculum.layers`` are
optional add-ons for embedding-rich domains. The Lean-extractive use needs only
this core.
"""
