"""ROUGE on hand-computed toy cases."""

import pytest

from artifacts_scripts.lexrank_bgnorm.rouge import lcs_length, rouge_l, rouge_n, score_all

REF = "the cat sat on the mat".split()
HYP = "the cat on the mat".split()


class TestRougeN:
    def test_rouge1_hand_computed(self):
        # ref counts: the:2 cat:1 sat:1 on:1 mat:1 (total 6)
        # hyp counts: the:2 cat:1 on:1 mat:1 (total 5); clipped overlap = 5
        s = rouge_n(REF, HYP, 1)
        assert s.precision == pytest.approx(1.0)
        assert s.recall == pytest.approx(5 / 6)
        assert s.f1 == pytest.approx(10 / 11)

    def test_rouge2_hand_computed(self):
        # ref bigrams: the-cat cat-sat sat-on on-the the-mat (5)
        # hyp bigrams: the-cat cat-on on-the the-mat (4); overlap 3
        s = rouge_n(REF, HYP, 2)
        assert s.precision == pytest.approx(3 / 4)
        assert s.recall == pytest.approx(3 / 5)
        assert s.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)

    def test_clipping(self):
        # hyp repeats "the" 4x but ref has only 2 -> clipped to 2
        s = rouge_n("the the".split(), "the the the the".split(), 1)
        assert s.precision == pytest.approx(2 / 4)
        assert s.recall == pytest.approx(1.0)

    def test_identical_and_empty(self):
        assert rouge_n(REF, REF, 1).f1 == pytest.approx(1.0)
        assert rouge_n(REF, [], 1).f1 == 0.0
        assert rouge_n([], HYP, 2).f1 == 0.0


class TestRougeL:
    def test_lcs_hand_computed(self):
        assert lcs_length(REF, HYP) == 5  # "the cat on the mat"
        assert lcs_length("a b c d".split(), "b d".split()) == 2
        assert lcs_length("a b".split(), "c d".split()) == 0
        assert lcs_length([], REF) == 0

    def test_rouge_l_hand_computed(self):
        s = rouge_l(REF, HYP)
        assert s.precision == pytest.approx(1.0)
        assert s.recall == pytest.approx(5 / 6)
        assert s.f1 == pytest.approx(10 / 11)

    def test_order_sensitivity(self):
        # same unigrams, scrambled order -> R1 perfect, RL lower
        ref = "a b c d e f".split()
        hyp = "f e d c b a".split()
        assert rouge_n(ref, hyp, 1).f1 == pytest.approx(1.0)
        assert rouge_l(ref, hyp).f1 == pytest.approx(2 * (1 / 6) * (1 / 6) / (2 / 6))


def test_score_all_keys():
    out = score_all(REF, HYP)
    assert set(out) == {"rouge1", "rouge2", "rougeL"}
    assert out["rouge1"] == pytest.approx(10 / 11)
