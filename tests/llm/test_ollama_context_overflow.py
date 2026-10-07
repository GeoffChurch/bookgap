"""A prompt never reaches the model truncated.

Ollama, by default, fits an over-long prompt into ``num_ctx`` by silently
dropping its start, and shifts the context the same way when prompt plus output
outgrow it. Either way the model answers a different prompt from the one that
was sent. OllamaProposalGenerator asks Ollama to refuse the first
(``truncate: false``) and refuses the second itself; both raise
ContextOverflowError.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

import bookgap.llm.generators.ollama_local as ollama_local
from bookgap.llm.generators.ollama_local import ContextOverflowError, OllamaProposalGenerator
from bookgap.llm.protocols import PromptPair

PROMPTS = PromptPair(system_prompt="SYS", user_prompt="USR")


def _generator(num_ctx: int) -> OllamaProposalGenerator:
    return OllamaProposalGenerator(model="m", temperature=0.2, num_ctx=num_ctx, num_predict=64)


def _reply(prompt_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content='{"title": "X", "abstract": "Y"}', thinking=None),
        prompt_eval_count=prompt_tokens, eval_count=output_tokens,
        total_duration=1, eval_duration=1, done_reason="stop",
    )


def test_every_request_forbids_truncation(monkeypatch):
    sent = []
    monkeypatch.setattr(ollama_local, "_chat", lambda body: sent.append(body) or _reply(10, 10))
    _generator(num_ctx=100)._chat_once(PROMPTS)
    assert sent[0]["truncate"] is False
    assert sent[0]["options"]["num_ctx"] == 100


def test_reply_that_fills_the_window_exactly_is_accepted(monkeypatch):
    monkeypatch.setattr(ollama_local, "_chat", lambda body: _reply(60, 40))
    assert _generator(num_ctx=100)._chat_once(PROMPTS).eval_count == 40


def test_reply_that_overflowed_the_window_is_refused(monkeypatch):
    monkeypatch.setattr(ollama_local, "_chat", lambda body: _reply(60, 41))
    with pytest.raises(ContextOverflowError, match="overflowed num_ctx=100"):
        _generator(num_ctx=100)._chat_once(PROMPTS)


def test_overflow_stops_generate_instead_of_retrying(monkeypatch):
    calls = []
    monkeypatch.setattr(ollama_local, "_chat", lambda body: calls.append(1) or _reply(90, 20))
    with pytest.raises(ContextOverflowError):
        _generator(num_ctx=100).generate(context=None, prompts=PROMPTS)
    assert len(calls) == 1


def test_server_refusal_of_a_long_prompt_raises(monkeypatch):
    # The body Ollama 0.32.14 returns for a 6405-token prompt with num_ctx=2048.
    refusal = json.dumps({"error": json.dumps({"error": {
        "code": 400, "type": "exceed_context_size_error",
        "message": "request (6405 tokens) exceeds the available context size (2048 tokens), try increasing it",
        "n_prompt_tokens": 6405, "n_ctx": 2048}})})

    def post(url, json, timeout):
        return httpx.Response(400, text=refusal, request=httpx.Request("POST", url))

    monkeypatch.setattr(ollama_local.httpx, "post", post)
    with pytest.raises(ContextOverflowError, match="longer than num_ctx=2048"):
        ollama_local._chat({"options": {"num_ctx": 2048}})


def test_other_server_errors_are_not_mistaken_for_overflow(monkeypatch):
    def post(url, json, timeout):
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(ollama_local.httpx, "post", post)
    with pytest.raises(httpx.HTTPStatusError):
        ollama_local._chat({"options": {"num_ctx": 2048}})
