"""Test the weighted-doc-relevance path through compute_gap_keywords."""

import pytest

# Reaches the private sibling `curriculum` via artifacts_scripts.keyword_helpers,
# which imports curriculum.layers._keyword_extraction. Transitive: neither this
# file nor the module under test mentions curriculum. See pyproject.toml.
pytest.importorskip("curriculum", reason="sibling curriculum checkout not installed")

import pytest


def test_compute_gap_keywords_with_weights_passes_relevance_to_extractor():
    """When weights is provided, articles with higher weight should
    contribute more to the basin's keyword profile.

    Setup: two basins, each with two docs.
      basin 0:  doc 0 ("alpha alpha alpha alpha alpha") weight 0.9
                doc 1 ("beta beta beta beta beta") weight 0.1
      basin 1:  doc 0 weight 0.1
                doc 1 weight 0.9

    Basin 0's top keyword should be alpha (mostly weighted alpha doc);
    basin 1's should be beta."""
    from artifacts_scripts.keyword_helpers import compute_gap_keywords

    titles = [
        "alpha alpha alpha alpha alpha",
        "beta beta beta beta beta",
    ]
    abstracts = ["", ""]
    gaps = [[0, 1], [0, 1]]
    weights = [
        [0.9, 0.1],
        [0.1, 0.9],
    ]
    keywords = compute_gap_keywords(
        gaps=gaps, titles=titles, abstracts=abstracts,
        weights=weights, top_k=3, bigram_iterations=0, verbose=False,
    )
    basin0_top = [kw for kw, _ in keywords[0]]
    basin1_top = [kw for kw, _ in keywords[1]]
    assert "alpha" in basin0_top
    assert "beta" in basin1_top
    assert basin0_top[0] == "alpha"
    assert basin1_top[0] == "beta"


def test_compute_gap_keywords_without_weights_unchanged():
    """When weights is None (default), behavior matches the original
    EQUAL doc weighting — backward-compatible."""
    from artifacts_scripts.keyword_helpers import compute_gap_keywords

    titles = ["alpha alpha", "beta beta", "gamma gamma", "delta delta"]
    abstracts = [""] * 4
    gaps = [[0, 1], [2, 3]]
    keywords = compute_gap_keywords(
        gaps=gaps, titles=titles, abstracts=abstracts,
        weights=None, top_k=2, bigram_iterations=0, verbose=False,
    )
    assert len(keywords) == 2
    assert all(len(kw_list) <= 2 for kw_list in keywords)
