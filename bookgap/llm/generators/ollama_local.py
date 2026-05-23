"""Ollama-based proposal generator for local LLM inference.

Parallels :class:`OpenAIBatchGenerator` but uses a synchronous local call
to an Ollama model (e.g. ``gemma4:31b``) instead of the OpenAI Batch API.
Prompt construction is the orchestrator's responsibility: callers pass
an already-rendered :class:`PromptPair` (typically from
:func:`bookgap.llm.prompt_render.render_prompts`). This keeps the prompt
format DRY across backends and lets the persisted ``PromptRecord`` serve
as the single source of truth for what was actually sent.

Deviations from the OpenAI batch path:

- One-at-a-time synchronous calls (no batch); suitable for offline eval
  runs but not for the production pipeline.
- No structured-output API; the orchestrator-rendered user prompt
  carries a JSON-output instruction (see ``prompt_render`` for the
  exact text) and we parse the response.
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
    PromptPair,
    Proposal,
    ProposalContext,
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
    """Synchronous Ollama proposal generator.

    Prompt construction is the orchestrator's responsibility: callers
    pass an already-rendered :class:`PromptPair` (typically from
    :func:`bookgap.llm.prompt_render.render_prompts`). This keeps prompt
    format DRY across backends and lets the persisted ``PromptRecord``
    serve as the single source of truth for what was actually sent.
    """

    def __init__(
        self,
        *,
        model: str,
        num_articles: int,
        num_books: int,
        temperature: float,
        num_ctx: int,
        num_predict: int,
    ) -> None:
        self.model = model
        # num_articles/num_books are retained for metadata only; the
        # actual sample sizes are bounded by the orchestrator before
        # prompts are built.
        self.num_articles = num_articles
        self.num_books = num_books
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(
        self, context: ProposalContext, prompts: PromptPair,
    ) -> OllamaProposalRecord:
        """Generate one proposal via Ollama. ``prompts`` is rendered by
        the orchestrator using the shared ``render_prompts`` function.

        On strict-parse failure (anything other than a single JSON
        object, optionally fenced), we re-call ``ollama.chat`` ONCE more
        — non-zero temperature gives the model another draw at a clean
        response. If the retry also fails strict parsing, we leave
        ``proposal=None`` (the offline recovery utility
        :func:`_try_parse_proposal_lenient` can still extract a
        proposal from the persisted ``raw_content`` if the operator
        opts in).
        """
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
