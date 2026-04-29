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
        """Generate one proposal synchronously via Ollama."""
        prompts = self.build_prompts(context)
        response = ollama.chat(
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

        raw = response.message.content or ""
        thinking = getattr(response.message, "thinking", None)
        proposal = _try_parse_proposal(raw)

        metadata: dict[str, Any] = {
            "model": self.model,
            "implementation": "ollama_local",
            "eval_count": response.eval_count,
            "prompt_eval_count": response.prompt_eval_count,
            "total_duration_ns": response.total_duration,
            "eval_duration_ns": response.eval_duration,
            "done_reason": response.done_reason,
        }

        return OllamaProposalRecord(
            proposal=proposal,
            raw_content=raw,
            thinking=thinking,
            prompts=prompts,
            context=context,
            metadata=metadata,
        )


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


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_proposal(raw: str) -> Proposal | None:
    """Best-effort JSON → Proposal parsing.

    Tries strict json first, then falls back to extracting the largest
    {...} block (handles cases where the model wraps the JSON in extra
    text despite the instruction).
    """
    raw = raw.strip()
    for candidate in (raw, _extract_json_object(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return Proposal.model_validate(parsed)
        except Exception:
            # Maybe it's a list with one item.
            if isinstance(parsed, list) and len(parsed) == 1:
                try:
                    return Proposal.model_validate(parsed[0])
                except Exception:
                    pass
    return None


def _extract_json_object(text: str) -> str | None:
    """Greedy extraction of the first {...} block in text. Returns None
    if no balanced block is found."""
    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None
