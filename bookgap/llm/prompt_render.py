"""Pure function: render the system+user prompt for one proposal from a
PromptRecord. Single source of truth for prompt format across all
backends (currently openai_batch + ollama_local) and any consumer that
needs to recompute the exact text the LLM saw.

Logic ported verbatim from the prior OpenAIBatchGenerator._build_*
methods and ollama_local._JSON_INSTRUCTION.
"""
from __future__ import annotations

from bookgap.llm.prompt_record import PromptRecord
from bookgap.llm.protocols import LLMBackend, PromptPair


_SYSTEM_PROMPT = "You are an academic researcher and educator."

_CORE_INSTRUCTION = (
    "propose a title and abstract for a textbook that would prepare "
    "students to understand research"
)

# Ollama lacks a structured-output API, so we instruct the model to
# emit a single JSON object directly. Kept byte-identical to the
# legacy ``_JSON_INSTRUCTION`` constant in ``ollama_local.py``.
_OLLAMA_JSON_INSTRUCTION = (
    "\n\nOutput ONLY a JSON object with this exact schema, and nothing else:\n"
    "{\"title\": \"...\", \"abstract\": \"...\"}\n"
    "No markdown fences, no commentary, no leading or trailing text."
)


def render_prompts(record: PromptRecord) -> PromptPair:
    user = _build_user_prompt(record)
    if record.backend == LLMBackend.OLLAMA_LOCAL.value:
        user = user + _OLLAMA_JSON_INSTRUCTION
    return PromptPair(system_prompt=_SYSTEM_PROMPT, user_prompt=user)


def _build_user_prompt(record: PromptRecord) -> str:
    intro_parts: list[str] = []
    if record.sampled_articles:
        intro_parts.append("research articles")
    if record.covering_books:
        intro_parts.append("related books")

    parts: list[str] = []
    if intro_parts:
        parts.append("Based on the ")
        parts.append(" and ".join(intro_parts))
        parts.append(
            " provided below, please " + _CORE_INSTRUCTION + " in this area.\n\n"
        )
    else:
        parts.append("Please " + _CORE_INSTRUCTION + ".\n\n")

    if record.sampled_articles:
        parts.append("## Sample Articles\n\n")
        for i, a in enumerate(record.sampled_articles, 1):
            parts.append(f"**Article {i}**: {a.display_name}\n")
            parts.append(f"\t*Abstract*: {a.abstract}\n\n")

    if record.covering_books:
        parts.append("## Related Existing Books\n\n")
        for i, b in enumerate(record.covering_books, 1):
            parts.append(f"**Book {i}**: {b.display_name}\n")
            parts.append(f"\t*Abstract*: {b.abstract}\n\n")

    return "".join(parts)
