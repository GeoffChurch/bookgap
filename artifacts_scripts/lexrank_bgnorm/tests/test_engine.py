"""B construction, Gramian walk, bg-norm demotion toy, baselines, budget selection."""

import numpy as np
import pytest

from artifacts_scripts.lexrank_bgnorm.engine import (
    bgnorm_score,
    build_sentence_word_matrix,
    centrality,
    g1_global_frequency,
    g2_generic_walk,
    gramian_walk,
    rankable_mask,
    ranking_from_scores,
    select_by_budget,
    silence_pagerank_progress,
    tfidf_degree,
)

silence_pagerank_progress()


class TestBMatrix:
    def test_counts_and_vocab(self):
        B, vocab = build_sentence_word_matrix([["cat", "sat", "cat"], ["dog", "sat"]])
        assert B.shape == (2, 3)
        assert B[0, vocab["cat"]] == 2.0
        assert B[0, vocab["sat"]] == 1.0
        assert B[0, vocab["dog"]] == 0.0
        assert B[1, vocab["dog"]] == 1.0

    def test_empty_sentence_row_is_zero(self):
        B, _ = build_sentence_word_matrix([["a"], []])
        assert B[1].nnz == 0

    def test_walk_rows_stochastic(self):
        B, _ = build_sentence_word_matrix([["cat", "sat"], ["dog", "sat"], ["dog", "ran"]])
        P = gramian_walk(B)
        sums = np.asarray(P.sum(axis=1)).ravel()
        assert np.allclose(sums, 1.0)

    def test_walk_connectivity(self):
        # sentences sharing no words have zero transition both ways
        B, _ = build_sentence_word_matrix([["alpha"], ["beta"]])
        P = gramian_walk(B).toarray()
        assert P[0, 1] == 0.0 and P[1, 0] == 0.0
        assert P[0, 0] == 1.0  # all mass returns to self


class TestCentrality:
    def test_uniform_on_symmetric_clique(self):
        toks = [["x", "a"], ["x", "b"], ["x", "c"]]
        B, _ = build_sentence_word_matrix(toks)
        p = centrality(gramian_walk(B))
        assert p == pytest.approx(np.full(3, 1 / 3), abs=1e-6)

    def test_hub_sentence_most_central(self):
        # s0 shares a word with everyone; s1..s3 only with s0
        toks = [["a", "b", "c"], ["a", "z1"], ["b", "z2"], ["c", "z3"]]
        B, _ = build_sentence_word_matrix(toks)
        p = centrality(gramian_walk(B))
        assert int(np.argmax(p)) == 0
        assert p.sum() == pytest.approx(1.0)


class TestBgNorm:
    def test_formula_matches_rank_py(self):
        p = np.array([0.4, 0.6])
        g = np.array([0.5, 0.5])
        eps = 1e-12
        expected = np.power(p, 0.5) * np.log((p + eps) / (g + eps))
        assert bgnorm_score(p, g) == pytest.approx(expected)

    def test_generic_sentence_demoted_on_toy(self):
        """4-sentence toy: the generic sentence must drop under both g variants.

        s0 is glue: built ONLY of corpus-common words ('city', 'century',
        'world') that the other sentences also touch -> plain LexRank makes it
        the most central. s1-s3 share the topical words 'matroid'/'lattice',
        globally rare. bg-norm must demote s0 below the topical core.
        """
        toks = [
            ["city", "century", "world", "city"],            # generic glue
            ["matroid", "lattice", "city"],                  # topical + 1 common
            ["matroid", "duality", "century"],               # topical + 1 common
            ["lattice", "duality", "world"],                 # topical + 1 common
        ]
        global_freq = {
            "city": 0.05, "century": 0.04, "world": 0.05,
            "matroid": 1e-5, "lattice": 2e-5, "duality": 1e-5,
        }
        B, vocab = build_sentence_word_matrix(toks)
        p = centrality(gramian_walk(B))
        assert int(np.argmax(p)) == 0  # plain LexRank crowns the generic sentence

        g1 = g1_global_frequency(toks, global_freq)
        s_g1 = bgnorm_score(p, g1)
        assert int(np.argmax(s_g1)) != 0
        assert int(np.argsort(-s_g1).tolist().index(0)) == 3  # demoted to last

        g2 = g2_generic_walk(B, vocab, global_freq)
        s_g2 = bgnorm_score(p, g2)
        assert int(np.argmax(s_g2)) != 0
        assert np.argsort(-s_g2).tolist().index(0) > np.argsort(-p).tolist().index(0)

    def test_g1_normalized_distribution(self):
        toks = [["rare"], ["common", "common2"]]
        g1 = g1_global_frequency(toks, {"rare": 1e-6, "common": 0.1, "common2": 0.2})
        assert g1.sum() == pytest.approx(1.0)
        assert g1[1] > g1[0]

    def test_g2_is_distribution_and_favors_common_vocab(self):
        """Generic-channel sentences get high g2; the topical hub gets low g2.

        s3 shares a (rare) word with every other sentence — the TOPICAL hub.
        s0-s2 interconnect only through 'common'. Under the global-frequency
        word-hop, nearly all mass flows through 'common', so s0-s2 carry high
        expected-generic centrality while s3 (reachable only through rare
        words) carries little — despite being the best-connected sentence.
        """
        toks = [["common", "t1"], ["common", "t2"], ["common", "t3"], ["t1", "t2", "t3"]]
        B, vocab = build_sentence_word_matrix(toks)
        g2 = g2_generic_walk(B, vocab, {"common": 0.2, "t1": 1e-6, "t2": 1e-6, "t3": 1e-6})
        assert g2.sum() == pytest.approx(1.0)
        assert all(g2[i] > g2[3] for i in range(3))

    def test_rankable_mask_excludes_empty(self):
        mask = rankable_mask([["a"], [], ["b"]])
        assert mask.tolist() == [True, False, True]
        ranking = ranking_from_scores(np.array([0.1, 9.9, 0.2]), mask)
        assert 1 not in ranking


class TestDegreeBaseline:
    def test_hub_has_highest_degree(self):
        toks = [["a", "b"], ["a", "c"], ["b", "c"], ["zz", "yy"]]
        B, _ = build_sentence_word_matrix(toks)
        deg = tfidf_degree(B)
        assert deg[3] == pytest.approx(0.0, abs=1e-12)  # isolated sentence
        assert min(deg[0], deg[1], deg[2]) > deg[3]


class TestBudgetSelection:
    def test_fill_until_budget_crossing_included(self):
        sel = select_by_budget(ranking=[2, 0, 1], sent_lengths=[10, 5, 7], budget=12)
        assert sel == [0, 2]  # 7 then 10 -> 17 >= 12, doc order

    def test_exact_budget_stops(self):
        sel = select_by_budget([0, 1], [12, 5], budget=12)
        assert sel == [0]

    def test_document_exhausted(self):
        sel = select_by_budget([1, 0], [3, 4], budget=100)
        assert sel == [0, 1]

    def test_zero_length_skipped(self):
        sel = select_by_budget([0, 1, 2], [0, 4, 4], budget=8)
        assert sel == [1, 2]
