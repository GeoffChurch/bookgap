#!/usr/bin/env python
# artifacts_scripts/curriculum/curriculum_llm_judge.py
"""Extrinsic LLM-as-judge for curriculum quality (the *semantic* dashboard).

The intrinsic basket (``curriculum_quality.py``,
``docs/curriculum-quality-metrics.md``) catches STRUCTURAL failures (singletons,
size balance, modularity, ordering geometry) but PROVABLY misses two SEMANTIC
ones it has no metric for:

  (a) v1's **pytorch/keras toolkit grab-bag** topic — embed-coherent, lexically
      distinct, but pedagogically useless (a tool list, not a teachable concept);
  (b) v3-m32's **bibliographic/tool/OCR keyword leak** into foundations
      (``icanhazpdf``, ``polyanalyst``, ``mistral``) — a keyword-meaningfulness
      defect the structural metrics rate as the *cleanest* config.

This layer scores a curriculum on exactly those dimensions, via the project's
local **ollama** adapter (the standing rule: LLM eval goes through local ollama,
NOT the OpenAI Batch API). It is deliberately CHEAP and BOUNDED: ~12-15 topic
judgments + ~10 ordered-pair judgments per curriculum, each optionally repeated a
few times for a noise/agreement read. All outputs are STRUCTURED (ollama
``format=<json schema>``) so we never parse free text, and the prompts are
NEUTRAL (a 1-5 scale with both poles described; we never tell the judge which
answer we want).

Three judged dimensions
========================

1. **Topic pedagogical validity** (per topic). Given its top keywords +
   representative titles: is this a coherent *teachable concept*, or a grab-bag /
   infrastructure / tool-list / bibliographic-noise cluster? Returns a 1-5
   ``rating`` (5 = clean teachable concept, 1 = grab-bag), a boolean
   ``is_grab_bag`` flag, and a one-line ``reason``. The litmus for (a).

2. **Prerequisite-ordering correctness** (per sampled ordered pair i<j). Given
   two topics where i is taught EARLIER than j: is i a sensible prerequisite of
   j (or at least correctly before it)? Returns ``agree`` (bool) + ``reason``.
   Aggregated to an order-sensibility rate.

3. **Keyword meaningfulness** (per topic). Among a topic's top keywords, which
   are bibliographic / tool / brand-name / OCR-noise tokens rather than genuine
   domain concepts? Returns the list of flagged keywords. Summed per curriculum.
   The litmus for (b).

Reads the SAME cached curricula the intrinsic scorer uses
(``cache/.curriculum_quality/``). Per-topic keywords + representative titles are
parsed from each cache's committed ``report*.md`` "Topic Overview" table, which
is already emitted in curriculum (prerequisite -> frontier) order — so we reuse
the pipelines' own ordering rather than re-deriving it.

Usage
=====
    conda run -p .conda python -m artifacts_scripts.curriculum.curriculum_llm_judge \
        --model gpt-oss:20b --repeats 3 \
        --json-out cache/.curriculum_quality/llm_judge.json

By default scores v1 (the toolkit-grab-bag litmus), v2, v3-m32, v3-m16, plus
wiki-native if its cache is present. ``--quick`` does a single repeat for a fast
look; ``--only NAME`` restricts to one curriculum.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ollama

QC_CACHE = Path("cache/.curriculum_quality")


def _log(msg: str) -> None:
    print(f"[curriculum_llm_judge] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Parsing the per-curriculum topic table out of the committed report*.md
# ---------------------------------------------------------------------------

@dataclass
class TopicCard:
    """One topic's identity card for the judge (curriculum order = ``rank``)."""
    rank: int                 # 1-based curriculum position (prereq -> frontier)
    size: int
    keywords: List[str]
    titles: List[str]         # representative member titles ("sample nodes/pages")


@dataclass
class CurriculumCard:
    name: str
    topics: List[TopicCard]   # in curriculum order


def _split_md_row(line: str) -> List[str]:
    """Split a markdown table row ``| a | b | c |`` into stripped cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _clean_keywords(cell: str) -> List[str]:
    """``probability, mathematics, logic`` -> tokens (strip ★ / markdown)."""
    cell = cell.replace("★", " ").replace("*", " ")
    return [w.strip() for w in cell.split(",") if w.strip()]


def _clean_titles(cell: str) -> List[str]:
    """``Parsing; Compiler; ★Foo`` -> titles ('★' marks OA-origin nodes)."""
    cell = cell.replace("★", " ").replace("*", " ")
    return [t.strip() for t in cell.split(";") if t.strip()]


def parse_report_topics(report_path: Path) -> List[TopicCard]:
    """Parse the "Topic Overview" table from a curriculum report markdown file.

    The table is uniform across all caches: a header row containing both ``Rank``
    and ``Size``, a separator row, then one row per topic IN CURRICULUM ORDER.
    The keyword column header contains "keyword"; the representative-title column
    header contains "Sample" (``Sample nodes`` / ``Sample pages``). We locate
    those columns by header text so the joint (``OA frac``/``in-out``) and
    wiki-native (``lift``/``bgnorm``) layouts both parse.
    """
    lines = report_path.read_text().splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("|") and "Rank" in ln and "Size" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no 'Rank|Size' topic table found in {report_path}")

    headers = _split_md_row(lines[header_idx])
    low = [h.lower() for h in headers]

    def _find(substr: str) -> int:
        for j, h in enumerate(low):
            if substr in h:
                return j
        raise ValueError(f"column '{substr}' not in headers {headers}")

    c_rank = _find("rank")
    c_size = _find("size")
    c_kw = _find("keyword")
    c_titles = _find("sample")

    topics: List[TopicCard] = []
    for ln in lines[header_idx + 1:]:
        if not ln.lstrip().startswith("|"):
            if topics:  # table ended
                break
            continue
        cells = _split_md_row(ln)
        if not cells or cells[c_rank] in ("", ":--:", ":--", "---:"):
            continue
        if set(cells[c_rank]) <= set(":- "):  # separator row
            continue
        try:
            rank = int(cells[c_rank])
        except (ValueError, IndexError):
            continue
        try:
            size = int(cells[c_size])
        except (ValueError, IndexError):
            size = -1
        topics.append(TopicCard(
            rank=rank,
            size=size,
            keywords=_clean_keywords(cells[c_kw]) if c_kw < len(cells) else [],
            titles=_clean_titles(cells[c_titles]) if c_titles < len(cells) else [],
        ))
    if not topics:
        raise ValueError(f"parsed zero topics from {report_path}")
    topics.sort(key=lambda t: t.rank)
    return topics


# Curriculum -> report markdown file inside its QC cache. Mirrors the loaders in
# curriculum_quality.py (v2 prefers the emb re-run cache to match the synthesis).
_CURRICULA: List[Tuple[str, str, str]] = [
    # (display name, cache subdir, report filename)
    ("joint-v1", "joint_v1", "report_v1.md"),
    ("joint-v2", "joint_v2", "report_v2.md"),
    ("v3-diffuse-m16", "joint_v3_m16", "report.md"),
    ("v3-diffuse-m32", "joint_v3_m32", "report.md"),
    ("wiki-native", "wiki_native", "report.md"),
]


def load_curriculum_card(name: str, subdir: str, report_name: str) -> CurriculumCard:
    report = QC_CACHE / subdir / report_name
    if not report.exists():
        # tolerate report.md vs report_vN.md drift
        alt = list((QC_CACHE / subdir).glob("report*.md"))
        if not alt:
            raise FileNotFoundError(f"no report markdown in {QC_CACHE / subdir}")
        report = alt[0]
    return CurriculumCard(name=name, topics=parse_report_topics(report))


# ---------------------------------------------------------------------------
# Structured-output schemas (ollama `format=`)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Neutral prompts (describe both poles; never lead the judge to an answer)
# ---------------------------------------------------------------------------

_TOPIC_VALIDITY_SYSTEM = (
    "You are an expert curriculum reviewer evaluating automatically-discovered "
    "topics for a self-taught course on natural language processing and machine "
    "learning. Judge each topic on its own merits and respond ONLY with JSON "
    "matching the requested schema."
)

_KEYWORD_SYSTEM = (
    "You are an expert reviewer auditing the keyword labels that were "
    "automatically mined for a topic in an NLP/ML curriculum. Respond ONLY with "
    "JSON matching the requested schema."
)

_ORDER_SYSTEM = (
    "You are an expert curriculum reviewer evaluating the ORDER in which topics "
    "are taught in a self-taught NLP/ML course. Respond ONLY with JSON matching "
    "the requested schema."
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
        "  5 = a clear, unified, teachable concept (e.g. 'syntactic parsing', "
        "'statistical machine translation').\n"
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
        "Here are the automatically-mined top keywords for one topic in an "
        "NLP/ML curriculum:\n"
        f"{kw}\n\n"
        "Some keywords may be genuine domain concepts (e.g. 'parsing', "
        "'machine_translation', 'probability'). Others may be NON-conceptual "
        "noise that leaked in: specific software tool or product brand names "
        "(e.g. a library or app name), bibliographic/metadata tokens, website or "
        "service names, OCR garbage, or proper names that are not themselves a "
        "concept being taught.\n\n"
        "List in noise_keywords EXACTLY those keywords from the list above that "
        "are such non-conceptual noise (tool/brand names, bibliographic/metadata "
        "tokens, OCR garbage). If none, return an empty list. Give a one-sentence "
        "reason."
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
        "sensible, false if B should clearly come before A (the order is "
        "backwards). Give a one-sentence reason."
    )


# ---------------------------------------------------------------------------
# Ollama judge
# ---------------------------------------------------------------------------

def _extract_first_json_object(content: str) -> Optional[dict]:
    """Best-effort recover the first balanced ``{...}`` object from ``content``.

    Tolerates a leading preamble ("Here is the JSON:\\n{...}") and trailing junk
    after the object, and parses a near-complete object as long as its braces
    balance. Returns the parsed dict, or ``None`` if no balanced object parses.
    Brace-matching is string-literal aware (ignores ``{``/``}`` inside JSON
    strings, honoring ``\\`` escapes) so braces inside ``reason`` text don't
    throw off the balance count.
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
    model: str
    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 1536
    n_calls: int = 0
    n_parse_fail: int = 0
    # context for the model-keyed raw-failure dump (set by the runner)
    rawfail_path: Optional[Path] = None
    _ctx: Dict[str, Any] = field(default_factory=dict)

    def set_context(self, **ctx: Any) -> None:
        """Record the curriculum/topic/dimension context for raw-fail dumps."""
        self._ctx = ctx

    def _dump_rawfail(self, content: str, dimension: str) -> None:
        if self.rawfail_path is None:
            return
        rec = {
            "model": self.model,
            "dimension": dimension,
            "content": content,
            **self._ctx,
        }
        self.rawfail_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rawfail_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def ask(self, system: str, user: str, schema: dict,
            dimension: str = "") -> Optional[dict]:
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
                think=False,  # gemma4:31b's thinking can exhaust num_predict before
                              # emitting the JSON -> empty content (esp. the harder
                              # pairwise prompt). Verified: think=True -> done=length,
                              # content=0; think=False -> done=stop, real answer.
            )
        except Exception as e:  # pragma: no cover - network/runtime
            _log(f"  ollama.chat failed: {e}")
            self.n_parse_fail += 1
            self._dump_rawfail(f"<ollama.chat exception: {e}>", dimension)
            return None
        content = (r.message.content or "").strip()
        try:
            obj: Any = json.loads(content)
        except json.JSONDecodeError:
            # Tolerant fallback: pull the first balanced {...} object out of a
            # preamble / trailing-junk / near-truncated response before counting
            # a true failure. Only failure if the tolerant extraction also fails.
            obj = _extract_first_json_object(content)
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
    """Majority vote + agreement fraction (fraction agreeing with the winner).

    Ties (even number of repeats split 50/50) resolve to ``False`` — the
    conservative default for a positive flag (grab-bag / order-agree): we only
    assert a flag when a strict majority of repeats vote for it.
    """
    if not votes:
        return False, float("nan")
    n_true = sum(1 for v in votes if v)
    n_false = len(votes) - n_true
    winner = n_true > n_false
    agree = max(n_true, n_false) / len(votes)
    return winner, agree


# ---------------------------------------------------------------------------
# Scoring one curriculum
# ---------------------------------------------------------------------------

@dataclass
class TopicJudgment:
    rank: int
    keywords: List[str]
    titles: List[str]
    validity_ratings: List[int] = field(default_factory=list)
    grab_bag_votes: List[bool] = field(default_factory=list)
    validity_reasons: List[str] = field(default_factory=list)
    noise_keyword_runs: List[List[str]] = field(default_factory=list)
    keyword_reasons: List[str] = field(default_factory=list)

    @property
    def mean_rating(self) -> float:
        return float(sum(self.validity_ratings) / len(self.validity_ratings)) \
            if self.validity_ratings else float("nan")

    @property
    def is_grab_bag(self) -> bool:
        win, _ = _majority_bool(self.grab_bag_votes)
        return win

    @property
    def grab_bag_agreement(self) -> float:
        _, ag = _majority_bool(self.grab_bag_votes)
        return ag

    @property
    def noise_keywords(self) -> List[str]:
        """Keywords flagged as noise by a MAJORITY of repeats (stable flags)."""
        if not self.noise_keyword_runs:
            return []
        counts: Counter = Counter()
        for run in self.noise_keyword_runs:
            for kw in set(run):
                counts[kw] += 1
        thresh = len(self.noise_keyword_runs) / 2.0
        return sorted([kw for kw, c in counts.items() if c >= thresh])


@dataclass
class PairJudgment:
    earlier_rank: int
    later_rank: int
    agree_votes: List[bool] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        win, _ = _majority_bool(self.agree_votes)
        return win

    @property
    def agreement(self) -> float:
        _, ag = _majority_bool(self.agree_votes)
        return ag


@dataclass
class CurriculumJudgeResult:
    name: str
    model: str
    repeats: int
    topics: List[TopicJudgment]
    pairs: List[PairJudgment]

    # --- aggregate semantic-dashboard metrics ---
    @property
    def n_topics(self) -> int:
        return len(self.topics)

    @property
    def mean_topic_validity(self) -> float:
        vals = [t.mean_rating for t in self.topics if not math.isnan(t.mean_rating)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @property
    def min_topic_validity(self) -> float:
        vals = [t.mean_rating for t in self.topics if not math.isnan(t.mean_rating)]
        return float(min(vals)) if vals else float("nan")

    @property
    def n_grab_bag_topics(self) -> int:
        return sum(1 for t in self.topics if t.is_grab_bag)

    @property
    def n_noise_keywords(self) -> int:
        """Total noise keywords flagged (majority-stable) across all topics."""
        return sum(len(t.noise_keywords) for t in self.topics)

    @property
    def n_topics_with_noise(self) -> int:
        return sum(1 for t in self.topics if t.noise_keywords)

    @property
    def order_sensibility_rate(self) -> float:
        if not self.pairs:
            return float("nan")
        return float(sum(1 for p in self.pairs if p.agree) / len(self.pairs))

    @property
    def mean_grab_bag_agreement(self) -> float:
        ags = [t.grab_bag_agreement for t in self.topics
               if not math.isnan(t.grab_bag_agreement)]
        return float(sum(ags) / len(ags)) if ags else float("nan")

    @property
    def mean_order_agreement(self) -> float:
        ags = [p.agreement for p in self.pairs if not math.isnan(p.agreement)]
        return float(sum(ags) / len(ags)) if ags else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "repeats": self.repeats,
            "summary": {
                "n_topics": self.n_topics,
                "mean_topic_validity": self.mean_topic_validity,
                "min_topic_validity": self.min_topic_validity,
                "n_grab_bag_topics": self.n_grab_bag_topics,
                "n_noise_keywords": self.n_noise_keywords,
                "n_topics_with_noise": self.n_topics_with_noise,
                "order_sensibility_rate": self.order_sensibility_rate,
                "mean_grab_bag_agreement": self.mean_grab_bag_agreement,
                "mean_order_agreement": self.mean_order_agreement,
            },
            "topics": [
                {
                    "rank": t.rank,
                    "keywords": t.keywords,
                    "titles": t.titles,
                    "mean_rating": t.mean_rating,
                    "ratings": t.validity_ratings,
                    "is_grab_bag": t.is_grab_bag,
                    "grab_bag_votes": t.grab_bag_votes,
                    "grab_bag_agreement": t.grab_bag_agreement,
                    "validity_reasons": t.validity_reasons,
                    "noise_keywords": t.noise_keywords,
                    "noise_keyword_runs": t.noise_keyword_runs,
                    "keyword_reasons": t.keyword_reasons,
                }
                for t in self.topics
            ],
            "pairs": [
                {
                    "earlier_rank": p.earlier_rank,
                    "later_rank": p.later_rank,
                    "agree": p.agree,
                    "agree_votes": p.agree_votes,
                    "agreement": p.agreement,
                    "reasons": p.reasons,
                }
                for p in self.pairs
            ],
        }


def sample_ordered_pairs(
    topics: Sequence[TopicCard], n_pairs: int, rng: random.Random,
) -> List[Tuple[TopicCard, TopicCard]]:
    """Sample ordered pairs (i earlier than j) for the ordering judgment.

    Includes all adjacent consecutive pairs first (the steps the curriculum
    actually makes), then fills the remainder with random non-adjacent i<j pairs
    up to ``n_pairs``. Singleton-only or <2-topic curricula yield fewer pairs.
    """
    ordered = sorted(topics, key=lambda t: t.rank)
    n = len(ordered)
    if n < 2:
        return []
    pairs: List[Tuple[int, int]] = [(i, i + 1) for i in range(n - 1)]
    far: List[Tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(i + 2, n)
    ]
    rng.shuffle(far)
    for pr in far:
        if len(pairs) >= n_pairs:
            break
        pairs.append(pr)
    pairs = pairs[:n_pairs]
    return [(ordered[i], ordered[j]) for i, j in pairs]


def judge_curriculum(
    card: CurriculumCard,
    client: JudgeClient,
    *,
    repeats: int,
    n_pairs: int,
    rng: random.Random,
) -> CurriculumJudgeResult:
    _log(f"judging {card.name}: {len(card.topics)} topics x{repeats}, "
         f"up to {n_pairs} ordered pairs x{repeats}")

    topic_judgments: List[TopicJudgment] = []
    for tc in card.topics:
        tj = TopicJudgment(rank=tc.rank, keywords=tc.keywords, titles=tc.titles)
        for rep in range(repeats):
            client.set_context(curriculum=card.name, topic_rank=tc.rank,
                               keywords=tc.keywords[:10], repeat=rep)
            v = client.ask(_TOPIC_VALIDITY_SYSTEM, _topic_validity_user(tc),
                           _TOPIC_VALIDITY_SCHEMA, dimension="topic_validity")
            if v is not None:
                try:
                    tj.validity_ratings.append(int(v["rating"]))
                    tj.grab_bag_votes.append(bool(v["is_grab_bag"]))
                    tj.validity_reasons.append(str(v.get("reason", "")))
                except (KeyError, ValueError, TypeError):
                    client.n_parse_fail += 1
            k = client.ask(_KEYWORD_SYSTEM, _keyword_user(tc), _KEYWORD_SCHEMA,
                           dimension="keyword_meaningfulness")
            if k is not None:
                raw = k.get("noise_keywords", [])
                if isinstance(raw, list):
                    # keep only flags that are actually among the topic's keywords
                    kwset = {w.lower() for w in tc.keywords}
                    flagged = [str(w) for w in raw
                               if str(w).lower() in kwset]
                    tj.noise_keyword_runs.append(flagged)
                    tj.keyword_reasons.append(str(k.get("reason", "")))
        topic_judgments.append(tj)

    pair_judgments: List[PairJudgment] = []
    for earlier, later in sample_ordered_pairs(card.topics, n_pairs, rng):
        pj = PairJudgment(earlier_rank=earlier.rank, later_rank=later.rank)
        for rep in range(repeats):
            client.set_context(curriculum=card.name,
                               earlier_rank=earlier.rank,
                               later_rank=later.rank, repeat=rep)
            o = client.ask(_ORDER_SYSTEM, _order_pair_user(earlier, later),
                           _ORDER_PAIR_SCHEMA, dimension="order_pair")
            if o is not None:
                try:
                    pj.agree_votes.append(bool(o["agree"]))
                    pj.reasons.append(str(o.get("reason", "")))
                except (KeyError, TypeError):
                    client.n_parse_fail += 1
        pair_judgments.append(pj)

    return CurriculumJudgeResult(
        name=card.name, model=client.model, repeats=repeats,
        topics=topic_judgments, pairs=pair_judgments,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_ROWS = [
    ("n_topics", "topics judged", "{:.0f}"),
    ("mean_topic_validity", "Topic validity (mean 1-5)", "{:.2f}"),
    ("min_topic_validity", "Topic validity (worst topic)", "{:.2f}"),
    ("n_grab_bag_topics", "#grab-bag topics flagged", "{:.0f}"),
    ("n_noise_keywords", "#noise keywords flagged", "{:.0f}"),
    ("n_topics_with_noise", "#topics with noise kw", "{:.0f}"),
    ("order_sensibility_rate", "Order-sensibility rate", "{:.2f}"),
    ("mean_grab_bag_agreement", "[noise] grab-bag vote agreement", "{:.2f}"),
    ("mean_order_agreement", "[noise] order vote agreement", "{:.2f}"),
]


def print_dashboard(results: List[CurriculumJudgeResult]) -> None:
    names = [r.name for r in results]
    summaries = {r.name: r.to_dict()["summary"] for r in results}
    header = f"{'semantic metric':<34}" + "".join(f"{n:>16}" for n in names)
    print()
    print(header)
    print("-" * len(header))
    for key, label, fmt in _DASHBOARD_ROWS:
        row = f"{label:<34}"
        for n in names:
            v = summaries[n].get(key, float("nan"))
            cell = "n/a" if (v is None or (isinstance(v, float) and math.isnan(v))) \
                else fmt.format(v)
            row += f"{cell:>16}"
        print(row)
    print()
    print("Higher=better: topic validity, order-sensibility rate. "
          "Lower=better: #grab-bag topics, #noise keywords, #topics with noise. "
          "[noise]=judge self-consistency across repeats (1.0=unanimous; a "
          "reliability read, not a quality axis).")
    print()
    # Litmus call-outs.
    for r in results:
        gb = [t for t in r.topics if t.is_grab_bag]
        if gb:
            print(f"  {r.name}: grab-bag topics @ ranks "
                  + ", ".join(f"{t.rank}(kw: {', '.join(t.keywords[:4])})" for t in gb))
        noisy = [t for t in r.topics if t.noise_keywords]
        if noisy:
            print(f"  {r.name}: noise keywords -> "
                  + "; ".join(f"rank {t.rank}: {t.noise_keywords}" for t in noisy))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss:20b",
                    help="ollama model to PIN for all judgments (default "
                         "gpt-oss:20b — fast, passes the litmus in testing).")
    ap.add_argument("--repeats", type=int, default=3,
                    help="judge each item this many times for a noise/agreement "
                         "read (default 3; use 1 for a quick look).")
    ap.add_argument("--quick", action="store_true",
                    help="shortcut for --repeats 1.")
    ap.add_argument("--pairs", type=int, default=10,
                    help="number of ordered topic pairs to judge per curriculum "
                         "(adjacent steps first, then random non-adjacent).")
    ap.add_argument("--only", default=None,
                    help="restrict to one curriculum by display name "
                         "(e.g. joint-v1).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-predict", type=int, default=1536,
                    help="max tokens the judge may generate per call (default "
                         "1536). 512 truncated gemma4:31b's verbose structured "
                         "output; 1536 leaves headroom for both judge models.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="dump the full per-item judgments as JSON.")
    args = ap.parse_args()

    repeats = 1 if args.quick else max(1, args.repeats)
    rng = random.Random(args.seed)

    # Verify ollama is reachable + the model is available before spinning.
    try:
        local = {m.model for m in ollama.list().models if m.model}
    except Exception as e:
        _log(f"FATAL: cannot reach ollama server ({e}). Is `ollama serve` up?")
        sys.exit(2)
    if args.model not in local:
        _log(f"FATAL: model '{args.model}' not pulled. Available: "
             f"{sorted(local)}. Run `ollama pull {args.model}`.")
        sys.exit(2)

    model_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    rawfail_path = QC_CACHE / f"llm_judge_{model_slug}_rawfails.jsonl"
    # Fresh dump per run (raw failures are re-parseable from disk afterward).
    if rawfail_path.exists():
        rawfail_path.unlink()
    client = JudgeClient(model=args.model, temperature=args.temperature,
                         num_predict=args.num_predict,
                         rawfail_path=rawfail_path)

    curricula = _CURRICULA
    if args.only:
        curricula = [c for c in curricula if c[0] == args.only]
        if not curricula:
            _log(f"no curriculum named '{args.only}'; "
                 f"choices: {[c[0] for c in _CURRICULA]}")
            sys.exit(2)

    t0 = time.time()
    results: List[CurriculumJudgeResult] = []
    for name, subdir, report_name in curricula:
        cache = QC_CACHE / subdir
        if not cache.exists():
            _log(f"skip {name}: cache {cache} missing")
            continue
        try:
            card = load_curriculum_card(name, subdir, report_name)
        except Exception as e:
            _log(f"skip {name}: could not load ({e})")
            continue
        res = judge_curriculum(card, client, repeats=repeats,
                               n_pairs=args.pairs, rng=rng)
        results.append(res)

    if not results:
        _log("no curricula judged")
        sys.exit(1)

    elapsed = time.time() - t0
    fail_rate = client.n_parse_fail / max(1, client.n_calls)
    _log(f"done: {client.n_calls} ollama calls, "
         f"{client.n_parse_fail} parse/transport failures ({fail_rate:.1%}), "
         f"{elapsed:.0f}s ({elapsed / max(1, client.n_calls):.1f}s/call)")
    if client.n_parse_fail and rawfail_path.exists():
        _log(f"raw failures dumped to {rawfail_path}")

    print_dashboard(results)

    if args.json_out:
        payload = {
            "model": args.model,
            "repeats": repeats,
            "pairs_per_curriculum": args.pairs,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "n_ollama_calls": client.n_calls,
            "n_parse_failures": client.n_parse_fail,
            "parse_failure_rate": client.n_parse_fail / max(1, client.n_calls),
            "elapsed_seconds": elapsed,
            "results": [r.to_dict() for r in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        _log(f"wrote full judgments to {args.json_out}")


if __name__ == "__main__":
    main()
