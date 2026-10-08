"""Appendix F's text escaping: what pdflatex can set is set, a script it cannot is
named, and anything else is a visible box."""
from __future__ import annotations

from artifacts_scripts.paper.generate_proposal_appendix import tex


def test_typographic_punctuation_becomes_latex_commands():
    # Gemma's proposals use em dashes; OpenAlex titles carry curly quotes and the rest.
    assert tex("a—b ‘c’ “d” 5° ¿e? 1–2") == (
        r"a\textemdash{}b \textquoteleft{}c\textquoteright{} "
        r"\textquotedblleft{}d\textquotedblright{} 5\textdegree{} \textquestiondown{}e? 1\textendash{}2"
    )


def test_a_title_wholly_in_another_script_becomes_one_label():
    # T14064's prompt titles are Arabic: the spaces between words stay inside the run
    assert tex("تحليل العلاقة") == "[Arabic script]"


def test_a_script_run_ends_at_latin_text_and_keeps_what_separates_them():
    assert tex("Обзор методов: XYZ-2000") == (
        "[Cyrillic script]: XYZ-2000")
    assert tex("ქართული Русский") == (
        "[Georgian script] [Cyrillic script]")


def test_a_latin_letter_outside_t1_stays_a_box():
    assert tex("Azərbaycan") == r"Az\unkchar{}rbaycan"
