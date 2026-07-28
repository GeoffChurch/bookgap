"""Splitter / cleaner / tokenizer edge cases."""

from artifacts_scripts.lexrank_bgnorm.textproc import (
    content_tokens,
    finish_clean_wikitext,
    split_lead_body,
    split_sentences,
    word_tokens,
)


class TestSentenceSplitter:
    def test_basic_two_sentences(self):
        s = split_sentences("The cat sat on the mat. The dog barked at the moon.")
        assert s == ["The cat sat on the mat.", "The dog barked at the moon."]

    def test_abbreviations_do_not_split(self):
        s = split_sentences("Dr. Smith went to Washington with Mr. Jones yesterday.")
        assert len(s) == 1

    def test_month_abbreviation(self):
        s = split_sentences("He arrived on Jan. 5 and left the city soon after.")
        assert len(s) == 1

    def test_initials_do_not_split(self):
        s = split_sentences(
            "J. R. R. Tolkien wrote the famous book. It sold very well worldwide."
        )
        assert len(s) == 2
        assert s[0].startswith("J. R. R. Tolkien")

    def test_dotted_acronym(self):
        s = split_sentences("He lived in the U.S. for nearly twenty years before moving.")
        assert len(s) == 1

    def test_decimal_numbers_intact(self):
        s = split_sentences("The ratio was about 3.5 to 1 overall. It rose again later that year.")
        assert len(s) == 2
        assert "3.5" in s[0]

    def test_exclamation_and_question(self):
        s = split_sentences("Why did the army retreat so quickly? Nobody could explain it at all!")
        assert len(s) == 2

    def test_quote_then_capital(self):
        s = split_sentences('He said it was "finished and done." Then he left the building forever.')
        assert len(s) == 2

    def test_no_split_before_lowercase(self):
        s = split_sentences("The site dates to c. 500 BC according to several modern estimates.")
        assert len(s) == 1

    def test_paragraph_break_is_boundary(self):
        s = split_sentences("First paragraph has no final period\n\nSecond paragraph is right here.")
        assert len(s) == 2

    def test_short_fragments_dropped(self):
        s = split_sentences("See above. The actual sentence has more than four tokens in it.")
        assert len(s) == 1
        assert s[0].startswith("The actual")

    def test_ie_eg_do_not_split(self):
        s = split_sentences(
            "The treaty covered minor states, e.g. Baden and Hesse, without naming them."
        )
        assert len(s) == 1


class TestCleaner:
    def test_nested_templates_dropped(self):
        out = finish_clean_wikitext("Start {{outer|x={{inner|y}}|z}} end of the line here.")
        assert "{{" not in out and "}}" not in out
        assert "Start" in out and "end of the line" in out

    def test_convert_template_keeps_value(self):
        out = finish_clean_wikitext("The wall is {{convert|14|km|mi}} long.")
        assert "14 km" in out

    def test_lang_template_keeps_text(self):
        out = finish_clean_wikitext("Known as {{lang|fr|le chat}} in French.")
        assert "le chat" in out

    def test_wikilinks_resolved(self):
        out = finish_clean_wikitext("She met [[Anne Boleyn]] and [[Henry VIII|the king]].")
        assert "Anne Boleyn" in out
        assert "the king" in out
        assert "[[" not in out and "|" not in out

    def test_tables_removed(self):
        out = finish_clean_wikitext("Before.\n{| class=x\n|-\n| cell {{tmpl}} \n|}\nAfter.")
        assert "cell" not in out
        assert "Before." in out and "After." in out

    def test_bold_italic_quotes_stripped(self):
        out = finish_clean_wikitext("'''Elizabeth I''' was ''queen'' of England.")
        assert "'''" not in out and "''" not in out
        assert "Elizabeth I" in out

    def test_math_content_dropped_other_tags_keep_text(self):
        out = finish_clean_wikitext("Euler proved <math>e^{i\\pi}=-1</math> here. <small>note</small>")
        assert "e^" not in out
        assert "note" in out

    def test_comments_and_refs_removed(self):
        out = finish_clean_wikitext("Real text<!-- hidden --> stays. <ref name=a>cite</ref>Done.")
        assert "hidden" not in out and "cite" not in out
        assert "Real text stays." in out

    def test_external_links(self):
        out = finish_clean_wikitext("See [https://example.com the site] and [https://x.org].")
        assert "the site" in out
        assert "https" not in out


class TestLeadBodySplit:
    WIKI = (
        "Lead sentence one is here. Lead sentence two follows it.\n\n"
        "==History==\nBody sentence in history section right here.\n"
        "===Early===\nSubsection sentence also in the body text.\n"
        "==References==\nNot prose at all.\n"
        "==Legacy==\nLegacy body sentence is here too.\n"
    )

    def test_split_and_section_drop(self):
        lead, body = split_lead_body(self.WIKI)
        assert "Lead sentence one" in lead
        assert "History" not in body  # heading line dropped
        assert "Body sentence in history" in body
        assert "Subsection sentence" in body
        assert "Not prose" not in body  # References section dropped
        assert "Legacy body sentence" in body  # resumes after blocklisted section

    def test_no_heading_means_all_lead(self):
        lead, body = split_lead_body("Only a lead here. Nothing else at all.")
        assert "Only a lead" in lead
        assert body.strip() == ""

    def test_list_lines_dropped(self):
        lead, body = split_lead_body("Lead text here.\n==S==\nProse line stays.\n* bullet goes\n")
        assert "Prose line stays" in body
        assert "bullet" not in body


class TestTokenizers:
    def test_word_tokens_rouge_style(self):
        assert word_tokens("Dr. Smith's 2-year plan!") == ["dr", "smith", "s", "2", "year", "plan"]

    def test_content_tokens_drop_stopwords_and_singletons(self):
        toks = content_tokens("The battle of Hastings was fought in 1066.")
        assert toks == ["battle", "hastings", "fought", "1066"]
