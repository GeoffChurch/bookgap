"""Golden-text test pinning the current prompt format so refactoring to
the shared render function can't drift."""
from bookgap.llm.prompt_record import (
    ArticleEntry, BookEntry, PromptRecord,
)
from bookgap.llm.prompt_render import render_prompts


_BASE_RECORD = PromptRecord(
    custom_id="gap_0_topic_T1",
    topic_id="T1", topic_name="Test", gap_idx=0,
    requested_num_prompt_articles=2,
    requested_num_prompt_books=1,
    sampled_articles=[
        ArticleEntry(openalex_id="W1", display_name="Art 1",
                     abstract="Abs 1"),
        ArticleEntry(openalex_id="W2", display_name="Art 2",
                     abstract="Abs 2"),
    ],
    covering_books=[
        BookEntry(openalex_id="B1", display_name="Book 1",
                  abstract="BAbs 1", articles_covered=10,
                  fraction_covered=0.5),
    ],
    backend="openai_batch",
    model="gpt-5",
    generation_params={},
)


def test_openai_user_prompt_format() -> None:
    pp = render_prompts(_BASE_RECORD)
    assert pp.system_prompt == "You are an academic researcher and educator."
    expected = (
        "Based on the research articles and related books provided "
        "below, please propose a title and abstract for a textbook that "
        "would prepare students to understand research in this area.\n\n"
        "## Sample Articles\n\n"
        "**Article 1**: Art 1\n\t*Abstract*: Abs 1\n\n"
        "**Article 2**: Art 2\n\t*Abstract*: Abs 2\n\n"
        "## Related Existing Books\n\n"
        "**Book 1**: Book 1\n\t*Abstract*: BAbs 1\n\n"
    )
    assert pp.user_prompt == expected


def test_ollama_appends_json_instruction() -> None:
    from dataclasses import replace
    rec = replace(_BASE_RECORD, backend="ollama_local", model="gemma4:31b")
    pp = render_prompts(rec)
    assert pp.user_prompt.startswith("Based on the research articles")
    assert "JSON" in pp.user_prompt


def test_no_articles_or_books_falls_back_to_generic_instruction() -> None:
    from dataclasses import replace
    rec = replace(_BASE_RECORD, sampled_articles=[], covering_books=[])
    pp = render_prompts(rec)
    assert pp.user_prompt == (
        "Please propose a title and abstract for a textbook that would "
        "prepare students to understand research.\n\n"
    )


def test_articles_only_intro() -> None:
    from dataclasses import replace
    rec = replace(_BASE_RECORD, covering_books=[])
    pp = render_prompts(rec)
    assert pp.user_prompt.startswith(
        "Based on the research articles provided below, please propose"
    )
    assert "## Related Existing Books" not in pp.user_prompt
    assert "## Sample Articles" in pp.user_prompt
