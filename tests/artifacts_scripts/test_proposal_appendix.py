"""Appendix F's text escaping: what pdflatex can set is set, the rest is a visible box."""
from __future__ import annotations

from artifacts_scripts.paper.generate_proposal_appendix import tex


def test_typographic_punctuation_becomes_latex_commands():
    # Gemma's proposals use em dashes; OpenAlex titles carry curly quotes and the rest.
    assert tex("a—b ‘c’ “d” 5° ¿e? 1–2") == (
        r"a\textemdash{}b \textquoteleft{}c\textquoteright{} "
        r"\textquotedblleft{}d\textquotedblright{} 5\textdegree{} \textquestiondown{}e? 1\textendash{}2"
    )


def test_scripts_pdflatex_cannot_set_are_boxed_one_per_character():
    assert tex("Обзор x") == r"\unkchar{}" * 5 + " x"
