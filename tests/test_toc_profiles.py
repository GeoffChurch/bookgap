from bookgap.toc.profiles import _clean_title
from bookgap.toc.profiles import chapter_n_profile, PROFILES


def test_clean_title_strips_trailing_page_number():
    assert _clean_title("Introduction 4") == "Introduction"


def test_clean_title_strips_html_and_missing_and_nbsp():
    assert _clean_title("Special Relativity<br/>") == "Special Relativity"
    assert _clean_title("World Building [[MISSING]]") == "World Building"
    assert _clean_title("Coming clean:\xa0Framing") == "Coming clean: Framing"


def test_clean_title_strips_trailing_part_marker():
    assert _clean_title(
        "Checking Out Visitors: Meteors, Comets, and Man-Made Moons PART II: On"
    ) == "Checking Out Visitors: Meteors, Comets, and Man-Made Moons"


_PEARSON = ("INTRODUCTION: Writing in English PART I: PARAGRAPH ORGANIZATION "
            "CHAPTER 1 The Topic Sentence CHAPTER 2 Supporting Sentences "
            "CHAPTER 3 The Concluding Sentence PART II: BASIC TYPES")

def test_chapter_n_extracts_ordered_chapters():
    p = chapter_n_profile(_PEARSON)
    assert p is not None and p.profile == "chapter-n"
    assert [n.title for n in p.nodes] == [
        "The Topic Sentence", "Supporting Sentences", "The Concluding Sentence"]

def test_chapter_n_rejects_fewer_than_three():
    assert chapter_n_profile("CHAPTER 1 Alpha CHAPTER 2 Beta") is None

def test_chapter_n_rejects_out_of_order():
    assert chapter_n_profile("CHAPTER 1 A CHAPTER 3 B CHAPTER 2 C CHAPTER 4 D") is None

def test_chapter_n_registered():
    assert PROFILES["chapter-n"] is chapter_n_profile
