from bookgap.toc.profiles import _clean_title


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
