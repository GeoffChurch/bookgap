from bookgap.toc.profiles import _clean_title
from bookgap.toc.profiles import chapter_n_profile, PROFILES
from bookgap.toc.profiles import arabic_list_profile
from bookgap.toc.profiles import roman_list_profile


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


_POWER = ("1. Power Electronics 2. Power Diodes 3. Power Transistors "
          "4. Thyristor Devices 5. Single-Phase Uncontrolled Rectifiers")

def test_arabic_list_extracts_ordered_prefix():
    p = arabic_list_profile(_POWER)
    assert p is not None and p.profile == "arabic-list"
    assert [n.title for n in p.nodes] == [
        "Power Electronics", "Power Diodes", "Power Transistors",
        "Thyristor Devices", "Single-Phase Uncontrolled Rectifiers"]

def test_arabic_list_rejects_prose_enumeration_below_five():
    # A three-item "contributions" enumeration must not parse.
    assert arabic_list_profile(
        "We make three contributions: 1. a dataset 2. a model 3. results") is None

def test_arabic_list_ignores_decimal_subsections():
    # "1.1"/"1.2" are decimals, not top-level items; no ordered 1..5 list here.
    assert arabic_list_profile("Section 1.1 intro 1.2 setup 1.3 method") is None

def test_arabic_list_rejects_sentence_length_titles():
    long = " ".join(
        f"{i}. " + ("word " * 30) for i in range(1, 6))  # 5 items but ~150-char titles
    assert arabic_list_profile(long) is None


_ROUTLEDGE = ("I. Newtonian Mechanics<br/> II. Special Relativity<br/> "
              "III. General Relativity<br/> IV. Non-Relativistic Quantum Theory<br/> "
              "V. Quantum Field Theory<br/>")

def test_roman_list_extracts_ordered_and_cleans_html():
    p = roman_list_profile(_ROUTLEDGE)
    assert p is not None and p.profile == "roman-list"
    assert [n.title for n in p.nodes] == [
        "Newtonian Mechanics", "Special Relativity", "General Relativity",
        "Non-Relativistic Quantum Theory", "Quantum Field Theory"]

def test_roman_list_rejects_below_five():
    assert roman_list_profile("I. A II. B III. C IV. D") is None


def test_chapter_n_handles_comma_before_label():
    # Real Task-8 false-split: no whitespace, a comma directly precedes the label.
    # Old boundary class excluded ',', so all three labels were missed -> None.
    ab = "Signal Processing,Chapter 1: Fundamentals,Chapter 2: Systems,Chapter 3: Advanced"
    p = chapter_n_profile(ab)
    assert p is not None
    assert [n.title for n in p.nodes] == ["Fundamentals", "Systems", "Advanced"]


def test_clean_title_strips_trailing_comma():
    assert _clean_title("Concept of the PC,") == "Concept of the PC"

def test_clean_title_strips_stuck_page_number():
    assert _clean_title("Evolve or Die11") == "Evolve or Die"      # glued to lowercase
    assert _clean_title("Systems Integration 27") == "Systems Integration"
    assert _clean_title("CO2") == "CO2"                            # uppercase-preceded digit KEPT

def test_clean_title_strips_trailing_chapter_header():
    assert _clean_title("Authority Rules Chapter") == "Authority Rules"
    assert _clean_title("Misconduct Chapter 3") == "Misconduct"
    assert _clean_title("A New Chapter in History") == "A New Chapter in History"  # mid-title KEPT

def test_chapter_n_rejects_prose_in_chapter_enumeration():
    # "In Chapter N, <lowercase prose>" -> fragments start lowercase -> rejected as prose.
    ab = ("In Chapter 1, a microbial fuel cell was studied. "
          "In Chapter 2, an autonomous model was built. "
          "In Chapter 3, studies of dehumidification are presented.")
    assert chapter_n_profile(ab) is None
