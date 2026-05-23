"""Smoke test for OpenAIBatchGenerator request construction.

Prompt-format coverage lives in tests/llm/test_prompt_render.py.
"""
import os
from unittest.mock import patch

import pytest

from bookgap.llm.protocols import (
    Article, Book, OpenAIModel, ProposalContext, ProposalCounts, PromptPair,
)


@pytest.fixture(autouse=True)
def _openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


def test_build_requests_emits_one_per_pair():
    from bookgap.llm.generators.openai_batch import OpenAIBatchGenerator
    g = OpenAIBatchGenerator(
        model=OpenAIModel.GPT_5, temperature=0.2, max_tokens=4096,
        use_structured_output=False,
    )
    ctx = ProposalContext(
        topic_name="T", topic_id="T1", gap_idx=0,
        requested_num_prompt_articles=1, requested_num_prompt_books=1,
        sampled_articles=[Article(display_name="A", abstract="X")],
        covering_books=[Book(display_name="B", abstract="Y")],
        counts=ProposalCounts(num_articles_for_llm=1, num_books_for_llm=1),
    )
    prompts = PromptPair(system_prompt="S", user_prompt="U")
    requests = g.build_requests([(ctx, prompts), (ctx, prompts)])
    assert len(requests) == 2
    assert requests[0].custom_id == "gap_0_topic_T1"
    assert requests[0].body.messages[0].content == "S"
    assert requests[0].body.messages[1].content == "U"
