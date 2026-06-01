#!/usr/bin/env python
# artifacts_scripts/curriculum/eval_pairwise_judge.py
"""Pairwise (A/B/tie) gemma4 LLM-judge comparing TWO built curricula.

Companion to ``eval_book_judge.py`` (the single-curriculum 1-7 rubric judge).
Where that scores one curriculum's book outline on an absolute 1-7 scale, this
asks gemma4:31b to directly COMPARE two curricula head-to-head on the same four
dimensions and pick a winner (or call a tie) on each. Pairwise preference is
often a sharper signal than two independent absolute scores when the two runs
differ only subtly (e.g. a ``base`` vs ``boundary`` selection variant on the
same topic).

Book-level only
===============
The comparison is at the BOOK level (the whole outline), NOT per-chapter:
chapters do not correspond between two independently-built curricula, so a
chapter-by-chapter A/B has no well-defined pairing. We render each curriculum's
single-blind outline (via ``eval_book_judge.render_outline`` — title + ordered
chapters with keywords + representative work titles) and present them as the
neutral **"Curriculum A"** and **"Curriculum B"**.

Single-blind + position-bias control
=====================================
The judge NEVER sees the true config names (``--label-a`` / ``--label-b``, e.g.
``base`` / ``boundary``): those labels are used ONLY to de-randomize the output
back to the real runs. To neutralise position bias (a judge's tendency to favour
whichever outline is shown first), we judge BOTH orders:

  - order1: A = run_a, B = run_b
  - order2: A = run_b, B = run_a

Each order is one judge call returning, for each of the four dimensions, a
verdict in {"A", "B", "tie"} plus a one-line comment (structured JSON via
``format=schema``; tolerant parse + persist-raw-on-failure, all reused from
``curriculum_llm_judge.JudgeClient``). ``--repeats`` per order (default 1) takes
a majority of the per-order verdicts.

De-randomize + combine (position-bias guard)
=============================================
Each order's A/B verdict is mapped back to run_a / run_b:

  - order1: "A" -> run_a, "B" -> run_b
  - order2: "A" -> run_b, "B" -> run_a   (positions are swapped)

A dimension's WINNER is run_a (or run_b) iff that SAME run is preferred in BOTH
orders. If the two orders disagree (a position flip) OR either order says
"tie", the dimension is recorded as a **tie**. This makes a verdict robust to
position bias: we only declare a winner when the preference survives swapping
the presentation order.

The four dimensions (same definitions as ``eval_book_judge.py``)
================================================================
  - **interestingness**   — does it seem an interesting book worth reading
  - **comprehensiveness** — does it seem to cover the full subject
  - **specificity**       — does it maintain focus on the proposed topic
  - **progressive**       — does the material move from easy to hard

Usage
=====
    conda run -p .conda python -m artifacts_scripts.curriculum.eval_pairwise_judge \
        --run-dir-a artifacts/curriculum/gap_curriculum_T10181_base \
        --run-dir-b artifacts/curriculum/gap_curriculum_T10181_boundary \
        --label-a base --label-b boundary

    # custom model / repeats / output path:
    conda run -p .conda python -m artifacts_scripts.curriculum.eval_pairwise_judge \
        --run-dir-a ... --run-dir-b ... --label-a base --label-b boundary \
        --model gemma4:31b --repeats 1 \
        --out artifacts/curriculum_eval_pairwise/T10181.json

Output: one JSON under ``artifacts/curriculum_eval_pairwise/<topic>.json``:
    {topic, run_a, run_b, label_a, label_b,
     dims:{<dim>:{winner, order1, order2, comments}}}
where ``winner`` is one of ``label_a`` / ``label_b`` / ``"tie"``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama

# Reuse the single-blind outline renderer and the ollama judge plumbing wholesale
# (model-keyed structured call, tolerant parse, persist-raw-on-failure counters).
from artifacts_scripts.curriculum.curriculum_llm_judge import (  # noqa: F401
    JudgeClient,
    _extract_first_json_object,
    _log,
)
from artifacts_scripts.curriculum.eval_book_judge import (
    DIMENSIONS,
    _parse_topic_name,
    render_outline,
)

DEFAULT_MODEL = "gemma4:31b"
EVAL_OUT_DIR = Path("artifacts/curriculum_eval_pairwise")
TIE = "tie"
_VERDICTS = ("A", "B", "tie")

# Reuse the exact four dimensions + rubric anchors from the single-curriculum
# judge so the absolute and pairwise layers agree on what each axis means.
_DIM_KEYS = [d[0] for d in DIMENSIONS]


# ---------------------------------------------------------------------------
# Structured-output schema (ollama `format=`): per dim a verdict + comment
# ---------------------------------------------------------------------------

def _pairwise_schema() -> dict:
    """{dim: {verdict: 'A'|'B'|'tie', comment: str}} for all four dims."""
    return {
        "type": "object",
        "properties": {
            dim: {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": list(_VERDICTS)},
                    "comment": {"type": "string"},
                },
                "required": ["verdict", "comment"],
            }
            for dim in _DIM_KEYS
        },
        "required": _DIM_KEYS,
    }


_RUBRIC_TEXT = "\n".join(f"  - {anchor}" for _, anchor in DIMENSIONS)

_PAIRWISE_SYSTEM = (
    "You are an expert book reviewer and curriculum designer comparing two "
    "proposed textbook outlines, 'Curriculum A' and 'Curriculum B', for the same "
    "subject. Judge them on their own merits, with no prior preference for either "
    "label, and respond ONLY with JSON matching the requested schema."
)


def _pairwise_user(outline_a: str, outline_b: str) -> str:
    return (
        "Two textbook outlines are proposed for the same subject. Each is given as "
        "its title, then its chapters in order, each chapter with key topics and a "
        "few representative works.\n\n"
        "===== Curriculum A =====\n"
        f"{outline_a}\n\n"
        "===== Curriculum B =====\n"
        f"{outline_b}\n\n"
        "Compare Curriculum A and Curriculum B on each of the following four "
        "dimensions. For EACH dimension, decide which curriculum is better on that "
        "dimension and give a 'verdict' of exactly \"A\", \"B\", or \"tie\" (use "
        "\"tie\" only when they are genuinely comparable on that dimension), plus a "
        "one-sentence 'comment' justifying the verdict.\n\n"
        f"{_RUBRIC_TEXT}"
    )


def _parse_pairwise(obj: Optional[dict]) -> Optional[Dict[str, Tuple[str, str]]]:
    """Validate a pairwise response -> {dim: (verdict, comment)} or None.

    Verdict is normalised to one of "A"/"B"/"tie"; anything else (or a missing
    cell) fails the whole response so the caller counts a schema-shape miss.
    """
    if not isinstance(obj, dict):
        return None
    out: Dict[str, Tuple[str, str]] = {}
    for dim in _DIM_KEYS:
        cell = obj.get(dim)
        if not isinstance(cell, dict) or "verdict" not in cell:
            return None
        verdict = str(cell["verdict"]).strip()
        # tolerate casing / stray punctuation around the token
        v_norm = verdict.lower()
        if v_norm == "a":
            verdict = "A"
        elif v_norm == "b":
            verdict = "B"
        elif v_norm == "tie":
            verdict = TIE
        else:
            return None
        out[dim] = (verdict, str(cell.get("comment", "")))
    return out


# ---------------------------------------------------------------------------
# One order = one (repeat-voted) judge call; verdicts are in A/B/tie space
# ---------------------------------------------------------------------------

def _majority_verdict(votes: List[str]) -> str:
    """Majority over {"A","B","tie"} per-repeat verdicts.

    A non-tie wins only with a STRICT majority; on a tie of counts (or an empty
    vote list, or "tie" winning) the order's verdict is "tie" — the conservative
    default that the combine step then treats as no-preference.
    """
    if not votes:
        return TIE
    counts = Counter(votes)
    top = counts.most_common()
    best_verdict, best_n = top[0]
    # strict-majority guard: if another verdict ties the top count, no winner
    if len(top) > 1 and top[1][1] == best_n:
        return TIE
    return best_verdict if best_verdict in ("A", "B") else TIE


@dataclass
class OrderResult:
    """One presentation order's per-dimension verdicts + comments.

    ``verdict`` values are in A/B presentation space ("A"/"B"/"tie"); the
    combine step maps them back to run_a/run_b according to which run was shown
    as A in this order.
    """
    verdicts: Dict[str, str]            # dim -> "A"/"B"/"tie" (majority over repeats)
    comments: Dict[str, List[str]]      # dim -> per-repeat comments


def judge_order(
    client: JudgeClient,
    outline_a: str,
    outline_b: str,
    *,
    repeats: int,
    ctx: Dict[str, Any],
) -> OrderResult:
    """Run one presentation order ``--repeats`` times; majority-vote per dim.

    ``outline_a`` is shown to the judge as Curriculum A and ``outline_b`` as
    Curriculum B, regardless of which real run each is — the caller decides the
    mapping per order.
    """
    schema = _pairwise_schema()
    system = _PAIRWISE_SYSTEM
    user = _pairwise_user(outline_a, outline_b)
    per_dim_votes: Dict[str, List[str]] = {d: [] for d in _DIM_KEYS}
    per_dim_comments: Dict[str, List[str]] = {d: [] for d in _DIM_KEYS}
    for rep in range(repeats):
        client.set_context(**ctx, repeat=rep)
        obj = client.ask(system, user, schema, dimension=ctx.get("order", "order"))
        parsed = _parse_pairwise(obj)
        if parsed is None:
            # ask() already counted a transport/parse failure if obj is None;
            # a schema-shape miss on a parsed dict is counted here.
            if obj is not None:
                client.n_parse_fail += 1
            continue
        for dim, (verdict, comment) in parsed.items():
            per_dim_votes[dim].append(verdict)
            per_dim_comments[dim].append(comment)
    verdicts = {d: _majority_verdict(per_dim_votes[d]) for d in _DIM_KEYS}
    return OrderResult(verdicts=verdicts, comments=per_dim_comments)


# ---------------------------------------------------------------------------
# De-randomize + combine the two orders into per-dim run-space winners
# ---------------------------------------------------------------------------

def _verdict_to_run(verdict: str, *, a_is_run_a: bool) -> Optional[str]:
    """Map an A/B/tie presentation verdict back to ``"run_a"``/``"run_b"``/None.

    ``a_is_run_a`` is True for order1 (Curriculum A == run_a) and False for
    order2 (Curriculum A == run_b, i.e. positions swapped). A "tie" verdict
    maps to None (no preferred run).
    """
    if verdict == "A":
        return "run_a" if a_is_run_a else "run_b"
    if verdict == "B":
        return "run_b" if a_is_run_a else "run_a"
    return None  # tie


def combine_dim(
    order1_verdict: str,
    order2_verdict: str,
    *,
    label_a: str,
    label_b: str,
) -> str:
    """Combine the two orders' verdicts for one dim into a run-space winner.

    Returns ``label_a``, ``label_b``, or ``"tie"``. A run wins iff it is the
    preferred run in BOTH orders (after de-randomizing each order's A/B back to
    run_a/run_b). A position flip between orders, or a "tie" in either order,
    yields ``"tie"`` (the position-bias guard).
    """
    run1 = _verdict_to_run(order1_verdict, a_is_run_a=True)
    run2 = _verdict_to_run(order2_verdict, a_is_run_a=False)
    if run1 is None or run2 is None:
        return TIE
    if run1 != run2:
        return TIE  # the two orders disagree -> position flip -> tie
    return label_a if run1 == "run_a" else label_b


# ---------------------------------------------------------------------------
# Top-level pairwise evaluation of two curricula
# ---------------------------------------------------------------------------

def evaluate_pair(
    run_dir_a: Path,
    run_dir_b: Path,
    client: JudgeClient,
    *,
    label_a: str,
    label_b: str,
    repeats: int,
    n_titles: int = 5,
) -> Dict[str, Any]:
    outline_a = render_outline(run_dir_a, n_titles=n_titles)
    outline_b = render_outline(run_dir_b, n_titles=n_titles)
    topic = _parse_topic_name(run_dir_a)

    _log(f"pairwise judging '{topic}': {label_a} vs {label_b}, both orders "
         f"x{repeats} ({client.model})")

    # order1: Curriculum A = run_a, Curriculum B = run_b
    order1 = judge_order(
        client, outline_a, outline_b, repeats=repeats,
        ctx={"topic": topic, "order": "order1"},
    )
    # order2: Curriculum A = run_b, Curriculum B = run_a (positions swapped)
    order2 = judge_order(
        client, outline_b, outline_a, repeats=repeats,
        ctx={"topic": topic, "order": "order2"},
    )

    dims: Dict[str, Any] = {}
    for dim in _DIM_KEYS:
        o1 = order1.verdicts[dim]
        o2 = order2.verdicts[dim]
        winner = combine_dim(o1, o2, label_a=label_a, label_b=label_b)
        dims[dim] = {
            "winner": winner,
            "order1_verdict": o1,
            "order2_verdict": o2,
            "comments": order1.comments[dim] + order2.comments[dim],
        }

    return {
        "topic": topic,
        "run_a": str(run_dir_a),
        "run_b": str(run_dir_b),
        "label_a": label_a,
        "label_b": label_b,
        "model": client.model,
        "repeats": repeats,
        "dims": dims,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(result: Dict[str, Any]) -> None:
    print()
    print(f"Pairwise: {result['topic']}  ({result['label_a']} vs "
          f"{result['label_b']}, model={result['model']}, "
          f"repeats={result['repeats']})")
    hdr = f"{'dimension':<20}{'order1':>10}{'order2':>10}{'WINNER':>16}"
    print(hdr)
    print("-" * len(hdr))
    for dim in _DIM_KEYS:
        d = result["dims"][dim]
        print(f"{dim:<20}{d['order1_verdict']:>10}{d['order2_verdict']:>10}"
              f"{d['winner']:>16}")
    print("\n(WINNER is a config name only when that run is preferred in BOTH "
          "orders; a flip or any tie -> 'tie'. order1/order2 verdicts are in "
          "A/B presentation space.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir-a", type=Path, required=True,
                    help="first built curriculum dir (must hold summary.md).")
    ap.add_argument("--run-dir-b", type=Path, required=True,
                    help="second built curriculum dir (must hold summary.md).")
    ap.add_argument("--label-a", required=True,
                    help="TRUE config name of run A (e.g. 'base'); used ONLY for "
                         "de-randomizing the output, NEVER shown to the judge.")
    ap.add_argument("--label-b", required=True,
                    help="TRUE config name of run B (e.g. 'boundary'); used ONLY "
                         "for de-randomizing the output, NEVER shown to the judge.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"ollama model to PIN for the judge (default "
                         f"{DEFAULT_MODEL}).")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each ORDER's single judge call this many times; "
                         "the per-dimension majority verdict is used (default 1).")
    ap.add_argument("--n-titles", type=int, default=5,
                    help="representative work titles to show per chapter (3-5; "
                         "default 5).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-predict", type=int, default=1536,
                    help="max tokens per judge call (default 1536; 512 truncates "
                         "gemma4:31b's verbose structured output).")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default "
                         "artifacts/curriculum_eval_pairwise/<topic>.json).")
    ap.add_argument("--print-outlines", action="store_true",
                    help="print both rendered single-blind outlines and exit "
                         "(no judging).")
    args = ap.parse_args()

    for run_dir in (args.run_dir_a, args.run_dir_b):
        if not run_dir.exists():
            _log(f"FATAL: run dir {run_dir} does not exist")
            sys.exit(2)

    if args.print_outlines:
        print("===== Curriculum A =====")
        print(render_outline(args.run_dir_a, n_titles=args.n_titles))
        print("\n===== Curriculum B =====")
        print(render_outline(args.run_dir_b, n_titles=args.n_titles))
        return

    repeats = max(1, args.repeats)

    # Verify ollama is reachable + the model is pulled before spinning.
    try:
        local = {m.model for m in ollama.list().models if m.model}
    except Exception as e:
        _log(f"FATAL: cannot reach ollama server ({e}). Is `ollama serve` up?")
        sys.exit(2)
    if args.model not in local:
        _log(f"FATAL: model '{args.model}' not pulled. Available: "
             f"{sorted(local)}. Run `ollama pull {args.model}`.")
        sys.exit(2)

    topic = _parse_topic_name(args.run_dir_a)
    topic_slug = re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_") or "pair"
    out_path = args.out or (EVAL_OUT_DIR / f"{topic_slug}.json")
    model_slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    rawfail_path = EVAL_OUT_DIR / f"{topic_slug}_{model_slug}_rawfails.jsonl"
    if rawfail_path.exists():
        rawfail_path.unlink()

    client = JudgeClient(model=args.model, temperature=args.temperature,
                         num_predict=args.num_predict, rawfail_path=rawfail_path)

    t0 = time.time()
    result = evaluate_pair(
        args.run_dir_a, args.run_dir_b, client,
        label_a=args.label_a, label_b=args.label_b,
        repeats=repeats, n_titles=args.n_titles,
    )
    elapsed = time.time() - t0
    fail_rate = client.n_parse_fail / max(1, client.n_calls)
    _log(f"done: {client.n_calls} ollama calls, {client.n_parse_fail} "
         f"parse/transport failures ({fail_rate:.1%}), {elapsed:.0f}s "
         f"({elapsed / max(1, client.n_calls):.1f}s/call)")
    if client.n_parse_fail and rawfail_path.exists():
        _log(f"raw failures dumped to {rawfail_path}")

    result["n_ollama_calls"] = client.n_calls
    result["n_parse_failures"] = client.n_parse_fail
    result["parse_failure_rate"] = fail_rate
    result["elapsed_seconds"] = elapsed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _log(f"wrote {out_path}")

    _print_summary(result)


if __name__ == "__main__":
    main()
