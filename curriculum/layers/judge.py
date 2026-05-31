"""Extrinsic LLM-as-judge for curriculum quality (local ollama).

Lifted from ``artifacts_scripts/curriculum/curriculum_llm_judge.py``: the robust
parse (tolerant brace-extraction + persist-raw on failure), the neutral
structured-output prompts/schemas, and the three judged dimensions (topic
pedagogical validity, prerequisite-ordering correctness, keyword
meaningfulness). Decoupled to operate on in-memory :class:`TopicCard` lists
(rather than parsing an on-disk markdown report) and model-pluggable.

``ollama`` is imported lazily inside :class:`JudgeClient.ask` so importing this
module never requires ollama to be installed — only running the judge does.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TopicCard:
    """One topic's identity card for the judge (curriculum order = ``rank``)."""

    rank: int
    size: int
    keywords: List[str]
    titles: List[str]


# --- structured-output schemas (ollama format=) ---------------------------

_TOPIC_VALIDITY_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "is_grab_bag": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["rating", "is_grab_bag", "reason"],
}

_ORDER_PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["agree", "reason"],
}

_KEYWORD_SCHEMA = {
    "type": "object",
    "properties": {
        "noise_keywords": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["noise_keywords", "reason"],
}

# --- neutral prompts (describe both poles; never lead the judge) ----------

_TOPIC_VALIDITY_SYSTEM = (
    "You are an expert curriculum reviewer evaluating automatically-discovered "
    "topics for a self-taught course. Judge each topic on its own merits and "
    "respond ONLY with JSON matching the requested schema."
)
_KEYWORD_SYSTEM = (
    "You are an expert reviewer auditing the keyword labels that were "
    "automatically mined for a topic in a curriculum. Respond ONLY with JSON "
    "matching the requested schema."
)
_ORDER_SYSTEM = (
    "You are an expert curriculum reviewer evaluating the ORDER in which topics "
    "are taught. Respond ONLY with JSON matching the requested schema."
)


def _topic_validity_user(topic: TopicCard) -> str:
    kw = ", ".join(topic.keywords[:10]) or "(none)"
    titles = "; ".join(topic.titles[:6]) or "(none)"
    return (
        "Here is one discovered topic.\n"
        f"Top keywords: {kw}\n"
        f"Representative member items (titles): {titles}\n\n"
        "Rate, on a scale of 1 to 5, whether this topic is a single coherent, "
        "teachable concept that a textbook chapter could be written about:\n"
        "  5 = a clear, unified, teachable concept.\n"
        "  3 = a loose but recognizable theme.\n"
        "  1 = NOT a teachable concept: a grab-bag of unrelated items, a list of "
        "software tools/libraries/frameworks, an infrastructure cluster, or a "
        "pile of bibliographic/metadata noise.\n\n"
        "Also set is_grab_bag=true if the topic is primarily a tool list, an "
        "infrastructure/metadata cluster, or an incoherent mixture rather than a "
        "teachable concept; otherwise false. Give a one-sentence reason."
    )


def _keyword_user(topic: TopicCard) -> str:
    kw = ", ".join(topic.keywords[:10]) or "(none)"
    return (
        "Here are the automatically-mined top keywords for one topic in a "
        "curriculum:\n"
        f"{kw}\n\n"
        "Some keywords may be genuine domain concepts. Others may be "
        "NON-conceptual noise: specific software tool or product brand names, "
        "bibliographic/metadata tokens, website or service names, OCR garbage, "
        "or proper names that are not themselves a concept being taught.\n\n"
        "List in noise_keywords EXACTLY those keywords from the list above that "
        "are such non-conceptual noise. If none, return an empty list. Give a "
        "one-sentence reason."
    )


def _order_pair_user(earlier: TopicCard, later: TopicCard) -> str:
    e_kw = ", ".join(earlier.keywords[:8]) or "(none)"
    l_kw = ", ".join(later.keywords[:8]) or "(none)"
    e_t = "; ".join(earlier.titles[:4]) or "(none)"
    l_t = "; ".join(later.titles[:4]) or "(none)"
    return (
        "In this curriculum, Topic A is taught EARLIER than Topic B.\n\n"
        f"Topic A (earlier)\n  keywords: {e_kw}\n  examples: {e_t}\n\n"
        f"Topic B (later)\n  keywords: {l_kw}\n  examples: {l_t}\n\n"
        "Question: is it pedagogically sensible to teach Topic A before Topic B "
        "— i.e. is A either a prerequisite/foundation for B, or at least "
        "reasonably placed before it? Set agree=true if the ordering A-then-B is "
        "sensible, false if B should clearly come before A. Give a one-sentence "
        "reason."
    )


def extract_first_json_object(content: str) -> Optional[dict]:
    """Best-effort recover the first balanced ``{...}`` object from ``content``.

    String-literal aware (ignores braces inside JSON strings, honoring ``\\``
    escapes). Tolerates a leading preamble and trailing junk. Returns the parsed
    dict or ``None``.
    """
    start = content.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(content)):
        ch = content[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = content[start:i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


@dataclass
class JudgeClient:
    """Thin, model-pluggable ollama client with the robust parse + raw-dump."""

    model: str
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 1536
    n_calls: int = 0
    n_parse_fail: int = 0
    rawfail_path: Optional[Path] = None
    _ctx: Dict[str, Any] = field(default_factory=dict)

    def set_context(self, **ctx: Any) -> None:
        self._ctx = ctx

    def _dump_rawfail(self, content: str, dimension: str) -> None:
        if self.rawfail_path is None:
            return
        rec = {"model": self.model, "dimension": dimension, "content": content, **self._ctx}
        self.rawfail_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rawfail_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def ask(self, system: str, user: str, schema: dict, dimension: str = "") -> Optional[dict]:
        import ollama  # lazy: only needed when the judge actually runs

        self.n_calls += 1
        try:
            r = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            )
        except Exception as e:  # pragma: no cover - network/runtime
            LOGGER.warning("ollama.chat failed: %s", e)
            self.n_parse_fail += 1
            self._dump_rawfail(f"<ollama.chat exception: {e}>", dimension)
            return None
        content = (r.message.content or "").strip()
        try:
            obj: Any = json.loads(content)
        except json.JSONDecodeError:
            obj = extract_first_json_object(content)
            if obj is None:
                self.n_parse_fail += 1
                self._dump_rawfail(content, dimension)
                return None
        if not isinstance(obj, dict):
            self.n_parse_fail += 1
            self._dump_rawfail(content, dimension)
            return None
        return obj


def _majority_bool(votes: List[bool]) -> Tuple[bool, float]:
    if not votes:
        return False, float("nan")
    n_true = sum(1 for v in votes if v)
    n_false = len(votes) - n_true
    winner = n_true > n_false
    agree = max(n_true, n_false) / len(votes)
    return winner, agree


def judge_curriculum(
    topics: Sequence[TopicCard],
    *,
    model: str,
    repeats: int = 1,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    num_predict: int = 1536,
    n_order_pairs: int = 10,
    seed: int = 0,
    rawfail_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the three judged dimensions over an ordered list of topic cards.

    ``topics`` must be in curriculum (prereq -> frontier) order. Returns a flat
    summary dict with per-topic and aggregate scores plus parse-failure counts.
    """
    client = JudgeClient(
        model=model, temperature=temperature, num_ctx=num_ctx,
        num_predict=num_predict, rawfail_path=rawfail_path,
    )
    rng = random.Random(seed)

    topic_ratings: List[Dict[str, Any]] = []
    for topic in topics:
        client.set_context(rank=topic.rank, dimension="topic_validity")
        ratings: List[int] = []
        grab_flags: List[bool] = []
        for _ in range(repeats):
            obj = client.ask(
                _TOPIC_VALIDITY_SYSTEM, _topic_validity_user(topic),
                _TOPIC_VALIDITY_SCHEMA, dimension="topic_validity",
            )
            if obj is not None:
                ratings.append(int(obj.get("rating", 0)))
                grab_flags.append(bool(obj.get("is_grab_bag", False)))
        is_grab, _ = _majority_bool(grab_flags)
        topic_ratings.append({
            "rank": topic.rank,
            "mean_rating": float(sum(ratings) / len(ratings)) if ratings else float("nan"),
            "is_grab_bag": is_grab,
        })

    noise_counts: List[int] = []
    for topic in topics:
        client.set_context(rank=topic.rank, dimension="keyword")
        per_repeat: List[int] = []
        for _ in range(repeats):
            obj = client.ask(
                _KEYWORD_SYSTEM, _keyword_user(topic), _KEYWORD_SCHEMA, dimension="keyword",
            )
            if obj is not None:
                per_repeat.append(len(obj.get("noise_keywords", []) or []))
        if per_repeat:
            noise_counts.append(int(round(sum(per_repeat) / len(per_repeat))))

    order_votes: List[bool] = []
    n = len(topics)
    pairs: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    for (i, j) in pairs[:n_order_pairs]:
        client.set_context(earlier=topics[i].rank, later=topics[j].rank, dimension="order")
        votes: List[bool] = []
        for _ in range(repeats):
            obj = client.ask(
                _ORDER_SYSTEM, _order_pair_user(topics[i], topics[j]),
                _ORDER_PAIR_SCHEMA, dimension="order",
            )
            if obj is not None:
                votes.append(bool(obj.get("agree", False)))
        if votes:
            winner, _ = _majority_bool(votes)
            order_votes.append(winner)

    valid_ratings = [t["mean_rating"] for t in topic_ratings if t["mean_rating"] == t["mean_rating"]]
    return {
        "model": model,
        "n_topics": n,
        "topic_ratings": topic_ratings,
        "mean_topic_rating": float(sum(valid_ratings) / len(valid_ratings)) if valid_ratings else float("nan"),
        "n_grab_bag_topics": sum(1 for t in topic_ratings if t["is_grab_bag"]),
        "total_noise_keywords": int(sum(noise_counts)),
        "order_agree_rate": float(sum(order_votes) / len(order_votes)) if order_votes else float("nan"),
        "n_calls": client.n_calls,
        "n_parse_fail": client.n_parse_fail,
    }
