"""Mock-judge unit tests for the pairwise (A/B/tie) curriculum judge.

NO ollama / GPU: we monkeypatch ``JudgeClient.ask`` to return canned per-order
verdicts and assert the de-randomization + position-bias-guard combine logic.
The three core cases mirror the spec:

  (a) a dim preferred for the SAME run in both orders -> that run wins (verifies
      each order's A/B is mapped back to run_a/run_b correctly: order1 A==run_a,
      order2 A==run_b);
  (b) an order-FLIP (verdict "A" in order1 AND "A" in order2 -> different runs,
      because positions are swapped) -> tie;
  (c) any "tie" verdict in EITHER order -> tie.

``render_outline`` is also stubbed so the test needs no on-disk curriculum dirs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import artifacts_scripts.curriculum.eval_pairwise_judge as epj
from artifacts_scripts.curriculum.eval_pairwise_judge import (
    JudgeClient,
    _DIM_KEYS,
    combine_dim,
    evaluate_pair,
)


# ---------------------------------------------------------------------------
# Pure-function checks on the position-bias-guard combine logic
# ---------------------------------------------------------------------------

def test_combine_same_run_both_orders_wins():
    # run_a preferred in BOTH orders: order1 shows it as "A", order2 shows it as
    # "B" (positions swapped) -> winner is label_a.
    assert combine_dim("A", "B", label_a="base", label_b="boundary") == "base"
    # symmetric: run_b preferred in both orders.
    assert combine_dim("B", "A", label_a="base", label_b="boundary") == "boundary"


def test_combine_position_flip_is_tie():
    # "A" in order1 -> run_a; "A" in order2 -> run_b (swapped). Disagreement -> tie.
    assert combine_dim("A", "A", label_a="base", label_b="boundary") == "tie"
    assert combine_dim("B", "B", label_a="base", label_b="boundary") == "tie"


def test_combine_any_tie_is_tie():
    assert combine_dim("tie", "B", label_a="base", label_b="boundary") == "tie"
    assert combine_dim("A", "tie", label_a="base", label_b="boundary") == "tie"
    assert combine_dim("tie", "tie", label_a="base", label_b="boundary") == "tie"


# ---------------------------------------------------------------------------
# End-to-end with a MOCK JudgeClient (no ollama). evaluate_pair issues exactly
# two ask() calls (order1 then order2); we hand back a scripted response each.
# ---------------------------------------------------------------------------

def _resp(per_dim_verdict: Dict[str, str]) -> dict:
    """A full pairwise response object: every dim gets the given verdict."""
    return {dim: {"verdict": per_dim_verdict[dim], "comment": f"because {dim}"}
            for dim in _DIM_KEYS}


class _MockJudge:
    """Stand-in for JudgeClient: returns queued responses in call order."""

    def __init__(self, responses: List[dict]):
        self._responses = list(responses)
        self.model = "mock:gemma4"
        self.n_calls = 0
        self.n_parse_fail = 0
        self.calls: List[Dict[str, Any]] = []

    def set_context(self, **ctx: Any) -> None:
        self._ctx = ctx

    def ask(self, system: str, user: str, schema: dict,
            dimension: str = "") -> Optional[dict]:
        self.n_calls += 1
        self.calls.append({"system": system, "user": user, "dimension": dimension})
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _stub_render_outline(monkeypatch):
    """Stub render_outline so no curriculum dirs / summary.md are needed."""
    monkeypatch.setattr(
        epj, "render_outline",
        lambda run_dir, n_titles=5: f"OUTLINE[{Path(run_dir).name}]",
    )
    monkeypatch.setattr(epj, "_parse_topic_name", lambda run_dir: "Test Topic")


def _eval(order1_verdicts: Dict[str, str], order2_verdicts: Dict[str, str],
          *, label_a: str = "base", label_b: str = "boundary") -> Dict[str, Any]:
    client = _MockJudge([_resp(order1_verdicts), _resp(order2_verdicts)])
    result = evaluate_pair(
        Path("run_a_dir"), Path("run_b_dir"), client,  # type: ignore[arg-type]
        label_a=label_a, label_b=label_b, repeats=1,
    )
    # Exactly two judge calls: one per order.
    assert client.n_calls == 2
    # The judge must never see the true config labels in the prompt.
    blob = " ".join(c["system"] + " " + c["user"] for c in client.calls)
    assert label_a not in blob and label_b not in blob
    return result


def test_e2e_same_run_both_orders_wins():
    # interestingness: run_a wins both orders (A in order1, B in order2).
    o1 = {d: ("A" if d == "interestingness" else "tie") for d in _DIM_KEYS}
    o2 = {d: ("B" if d == "interestingness" else "tie") for d in _DIM_KEYS}
    res = _eval(o1, o2)
    dim = res["dims"]["interestingness"]
    assert dim["winner"] == "base"
    assert dim["order1_verdict"] == "A"
    assert dim["order2_verdict"] == "B"
    # other dims tied
    assert res["dims"]["specificity"]["winner"] == "tie"
    # de-randomization for the OTHER run too: run_b wins both orders.
    o1b = {d: ("B" if d == "comprehensiveness" else "tie") for d in _DIM_KEYS}
    o2b = {d: ("A" if d == "comprehensiveness" else "tie") for d in _DIM_KEYS}
    res_b = _eval(o1b, o2b)
    assert res_b["dims"]["comprehensiveness"]["winner"] == "boundary"


def test_e2e_position_flip_is_tie():
    # "A" in BOTH orders -> run_a then run_b (positions swap) -> flip -> tie.
    o1 = {d: "A" for d in _DIM_KEYS}
    o2 = {d: "A" for d in _DIM_KEYS}
    res = _eval(o1, o2)
    for d in _DIM_KEYS:
        assert res["dims"][d]["winner"] == "tie", d


def test_e2e_any_tie_in_either_order_is_tie():
    # specificity: tie in order1 (even though order2 prefers a run) -> tie.
    o1 = {d: ("tie" if d == "specificity" else "A") for d in _DIM_KEYS}
    o2 = {d: ("B" if d == "specificity" else "B") for d in _DIM_KEYS}
    res = _eval(o1, o2)
    assert res["dims"]["specificity"]["winner"] == "tie"
    # the non-tie dims here are "A"/"B": run_a both orders -> base wins.
    assert res["dims"]["interestingness"]["winner"] == "base"


def test_e2e_comments_collected_from_both_orders():
    o1 = {d: "A" for d in _DIM_KEYS}
    o2 = {d: "B" for d in _DIM_KEYS}
    res = _eval(o1, o2)
    # one comment per order per dim (repeats=1) -> two comments combined.
    assert len(res["dims"]["interestingness"]["comments"]) == 2


# ---------------------------------------------------------------------------
# Schema/parse helpers and the per-order majority vote.
# ---------------------------------------------------------------------------

def test_parse_pairwise_normalises_and_rejects():
    good = {dim: {"verdict": "a", "comment": "c"} for dim in _DIM_KEYS}
    parsed = epj._parse_pairwise(good)
    assert parsed is not None
    assert parsed["interestingness"][0] == "A"  # casing normalised

    # missing a dim -> reject
    bad = {dim: {"verdict": "A", "comment": ""} for dim in _DIM_KEYS[:-1]}
    assert epj._parse_pairwise(bad) is None

    # junk verdict -> reject
    junk = {dim: {"verdict": "maybe", "comment": ""} for dim in _DIM_KEYS}
    assert epj._parse_pairwise(junk) is None


def test_majority_verdict():
    assert epj._majority_verdict(["A", "A", "B"]) == "A"
    assert epj._majority_verdict(["A", "B"]) == "tie"       # count tie -> tie
    assert epj._majority_verdict(["tie", "tie", "A"]) == "tie"
    assert epj._majority_verdict([]) == "tie"               # no votes -> tie


def test_judgeclient_reused_from_curriculum_llm_judge():
    # Confirms the build reuses the existing plumbing rather than redefining it.
    from artifacts_scripts.curriculum.curriculum_llm_judge import (
        JudgeClient as Base,
    )
    assert JudgeClient is Base
