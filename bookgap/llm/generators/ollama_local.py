"""Ollama-based proposal generator for local LLM inference.

Parallels :class:`OpenAIBatchGenerator` but uses a synchronous local call
to an Ollama model (e.g. ``gemma4:31b``) instead of the OpenAI Batch API.
Inherits the prompt-building logic from :class:`OpenAIBatchGenerator` so
prompts are identical to what the paper used. The only deviations from
the OpenAI version:

- One-at-a-time synchronous calls (no batch); suitable for offline eval
  runs but not for the production pipeline.
- No structured-output API; we instruct the model to emit JSON in the
  user prompt and then parse the response.
- Ollama models that produce a ``thinking`` block (e.g. gemma4) have
  their thoughts captured separately for transparency but only the
  final ``content`` is parsed as the proposal.

Use case: the K-ball-vs-HDBSCAN comparison experiment that needs many
proposals generated cheaply, or any other offline eval where keeping
generations local matters more than throughput.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import ollama

from ..protocols import (
    ChatMessage,
    PromptPair,
    Proposal,
    ProposalContext,
)
from .openai_batch import OpenAIBatchGenerator


_JSON_INSTRUCTION = (
    "\n\nOutput ONLY a JSON object with this exact schema, and nothing else:\n"
    "{\"title\": \"...\", \"abstract\": \"...\"}\n"
    "No markdown fences, no commentary, no leading or trailing text."
)


@dataclass(frozen=True)
class OllamaProposalRecord:
    """Container for one Ollama-generated proposal + its provenance.

    Distinct from :class:`ProposalResult` (which assumes file-on-disk
    artifacts written by the OpenAI batch flow). Used in-memory by the
    experiment runner so callers decide where to persist.
    """

    proposal: Proposal | None  # None if JSON parsing failed
    raw_content: str           # the model's literal `content` field
    thinking: str | None       # the model's `thinking` field (gemma4-style)
    prompts: PromptPair        # exact prompts sent
    context: ProposalContext   # exact context that fed the prompts
    metadata: dict[str, Any] = field(default_factory=dict)


class OllamaProposalGenerator:
    """Local-Ollama proposal generator.

    Reuses :class:`OpenAIBatchGenerator`'s prompt-building so prompts
    match the published methodology exactly. Differs only in transport
    (synchronous local Ollama call) and output handling (no
    structured-output API; JSON parsed from raw content).
    """

    def __init__(
        self,
        *,
        model: str,
        num_articles: int,
        num_books: int,
        temperature: float = 0.7,
        num_ctx: int = 8192,
        num_predict: int = 8192,
    ):
        """Initialise an Ollama proposal generator.

        Parameters
        ----------
        model
            Ollama model tag (e.g. ``"gemma4:31b"``, ``"qwen3:30b"``).
        num_articles, num_books
            Plumbed to the prompt-builder so context is sized
            consistently with the OpenAI flow.
        temperature
            Sampling temperature.
        num_ctx
            Context window. 8192 is enough for a typical proposal
            context (5 articles + 5 books).
        num_predict
            Max generated tokens. Must be generous for thinking models
            (gemma4 spends most of its budget on thoughts before
            emitting the final JSON content).
        """
        # Reuse OpenAIBatchGenerator's prompt-building by composition,
        # not inheritance — we don't want its OPENAI_API_KEY check or
        # its batch-API surface.
        self._prompt_builder = _PromptBuilder(
            num_articles=num_articles, num_books=num_books,
        )
        self.model = model
        self.num_articles = num_articles
        self.num_books = num_books
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def build_prompts(self, context: ProposalContext) -> PromptPair:
        """Build prompts via the shared OpenAI logic plus a JSON-output
        instruction (since Ollama lacks structured-output APIs)."""
        base = self._prompt_builder.build(context)
        return PromptPair(
            system_prompt=base.system_prompt,
            user_prompt=base.user_prompt + _JSON_INSTRUCTION,
        )

    def generate(self, context: ProposalContext) -> OllamaProposalRecord:
        """Generate one proposal synchronously via Ollama.

        On strict-parse failure (anything other than a single JSON
        object, optionally fenced), we re-call ``ollama.chat`` ONCE more
        — non-zero temperature gives the model another draw at a clean
        response. If the retry also fails strict parsing, we leave
        ``proposal=None`` (the offline recovery utility
        :func:`_try_parse_proposal_lenient` can still extract a
        proposal from the persisted ``raw_content`` if the operator
        opts in).
        """
        prompts = self.build_prompts(context)
        response = self._chat_once(prompts)
        raw = response.message.content or ""
        thinking = getattr(response.message, "thinking", None)
        proposal = _try_parse_proposal(raw)
        metadata = self._build_metadata(response)

        if proposal is None:
            # ONE retry. Same prompt, fresh sampling.
            first_raw = raw
            first_thinking = thinking
            retry_response = self._chat_once(prompts)
            retry_raw = retry_response.message.content or ""
            retry_thinking = getattr(retry_response.message, "thinking", None)
            retry_proposal = _try_parse_proposal(retry_raw)
            metadata["retried"] = True
            metadata["first_raw_content"] = first_raw
            metadata["first_thinking"] = first_thinking
            if retry_proposal is not None:
                # Success on retry: persist the retry as `raw_content`
                # for downstream / re-parse consistency. The first
                # attempt is preserved in metadata for forensics.
                proposal = retry_proposal
                raw = retry_raw
                thinking = retry_thinking
                metadata.update(self._build_metadata(retry_response))
            else:
                # Both failed: persist the retry as `raw_content` (most
                # recent attempt) and the first in metadata.
                raw = retry_raw
                thinking = retry_thinking
                metadata.update(self._build_metadata(retry_response))

        return OllamaProposalRecord(
            proposal=proposal,
            raw_content=raw,
            thinking=thinking,
            prompts=prompts,
            context=context,
            metadata=metadata,
        )

    def _chat_once(self, prompts: PromptPair):
        return ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.system_prompt},
                {"role": "user", "content": prompts.user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        )

    def _build_metadata(self, response) -> dict[str, Any]:
        return {
            "model": self.model,
            "implementation": "ollama_local",
            "eval_count": response.eval_count,
            "prompt_eval_count": response.prompt_eval_count,
            "total_duration_ns": response.total_duration,
            "eval_duration_ns": response.eval_duration,
            "done_reason": response.done_reason,
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _PromptBuilder(OpenAIBatchGenerator):
    """Compositional shim: subclass OpenAIBatchGenerator just to access
    its private prompt-building methods, but skip the OPENAI_API_KEY
    requirement that the parent's __init__ enforces."""

    def __init__(self, num_articles: int, num_books: int) -> None:  # noqa: D401
        # Skip the parent __init__ entirely — we don't need the OpenAI
        # API key, model enum, or temperature/max_tokens for prompt
        # building. Only num_articles / num_books are referenced inside
        # the prompt logic, and those are stored as plain attributes.
        self.num_articles = num_articles
        self.num_books = num_books

    def build(self, context: ProposalContext) -> PromptPair:
        return self._build_prompt(context)


_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


def _try_parse_proposal(raw: str) -> Proposal | None:
    """Strict JSON → Proposal parsing.

    Accepts only a single JSON object, optionally wrapped in **one**
    enclosing markdown code fence. Rejects lists, multi-block outputs,
    and any output with stray commentary before/after the JSON. The
    intent is to fail loudly when the model deviates from the
    ``{title, abstract}`` contract so that :meth:`OllamaProposalGenerator.generate`
    can do a one-shot retry. For offline recovery from past JSONL files
    that contain such deviations, use :func:`_try_parse_proposal_lenient`.
    """
    raw = raw.strip()
    m = _MARKDOWN_FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        return Proposal.model_validate(parsed)
    except Exception:
        return None


def _try_parse_proposal_lenient(raw: str) -> Proposal | None:
    """Lenient JSON → Proposal parsing, for offline recovery only.

    Scans ``raw`` for every top-level balanced JSON block (``{...}``
    or ``[...]``), parses each, and returns the first ``Proposal``
    that validates. For list blocks, returns the first list item that
    validates. Strings (including escaped quotes) are skipped so
    brackets inside string literals don't confuse the depth counter.

    NOT used during fresh generation — that path uses
    :func:`_try_parse_proposal` + one retry. This is reserved for
    recovering data from JSONL files written before the strict + retry
    machinery was in place.
    """
    raw = raw.strip()
    for block in _iter_balanced_json_blocks(raw):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        try:
            return Proposal.model_validate(parsed)
        except Exception:
            pass
        if isinstance(parsed, list):
            for item in parsed:
                try:
                    return Proposal.model_validate(item)
                except Exception:
                    continue
    return None


def _iter_balanced_json_blocks(text: str):
    """Yield every top-level balanced ``{...}`` or ``[...]`` substring.

    Tracks bracket depth (counting both ``{`` and ``[``) while respecting
    JSON string literals — brackets inside a `"..."` string are ignored.
    Backslash escapes inside strings are honored so ``"a\\"b"`` is one string.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif c in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1
