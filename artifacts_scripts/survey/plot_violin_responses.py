#!/usr/bin/env python3
"""
Plot violin distributions of raw survey responses per book for each question type.
One figure with three subplots: Interesting?, Well-written?, Publishable?.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from verify_survey_alignment import (
    BookRecord,
    QuestionType,
    choose_by_percentile,
    load_selected_books,
    load_survey,
)

QUESTION_TYPES: List[QuestionType] = ["interesting", "well_written", "publishable"]
QUESTION_TITLES = {
    "interesting": "Interesting?",
    "well_written": "Well-written?",
    "publishable": "Publishable?",
}
ROLE_ORDER = ["high", "low", "ours"]


def assign_roles(selected: List[BookRecord]) -> Dict[str, str]:
    """Assign each book title a role of high/low/ours within its topic."""
    by_topic: Dict[str, List[BookRecord]] = {}
    for rec in selected:
        by_topic.setdefault(rec.topic_id, []).append(rec)

    roles: Dict[str, str] = {}
    for records in by_topic.values():
        high, low, generated = choose_by_percentile(records)
        if high:
            roles[high.title] = "high"
        if low:
            roles[low.title] = "low"
        if generated:
            roles[generated.title] = "ours"
    return roles


def gather_responses(
    selected: List[BookRecord],
    questions,
    responses,
) -> Dict[QuestionType, List[Tuple[str, List[float]]]]:
    """Return per-question: list of (label, response values) for each selected book."""
    roles = assign_roles(selected)
    # Map title -> qtype -> column name for quick lookup.
    q_by_title: Dict[str, Dict[QuestionType, str]] = {}
    for q in questions:
        q_by_title.setdefault(q.title, {})[q.question_type] = q.column_name

    # Order books by topic, then role.
    ordered = sorted(selected, key=lambda r: (r.topic_id, ROLE_ORDER.index(roles.get(r.title, "ours")), r.title))

    per_q: Dict[QuestionType, List[Tuple[str, List[float]]]] = {qt: [] for qt in QUESTION_TYPES}
    for rec in ordered:
        role = roles.get(rec.title, "ours")
        label = f"{rec.topic_id}\n{role}"
        title_map = q_by_title.get(rec.title)
        if not title_map:
            continue
        for qtype in QUESTION_TYPES:
            col = title_map.get(qtype)
            if not col:
                continue
            vals: List[float] = []
            for row in responses:
                raw = row.get(col, "").strip()
                if not raw:
                    continue
                try:
                    vals.append(float(raw))
                except ValueError:
                    continue
            per_q[qtype].append((label, vals))
    return per_q


def plot_violins(data: Dict[QuestionType, List[Tuple[str, List[float]]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, qtype in zip(axes, QUESTION_TYPES):
        entries = data[qtype]
        if not entries:
            ax.set_visible(False)
            continue
        labels, dists = zip(*entries)
        parts = ax.violinplot(dists, showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("lightgray")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        ax.set_title(QUESTION_TITLES[qtype])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_xlabel("Topic / role")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Rating (1–7)")
    axes[0].set_ylim(0.75, 7.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-path",
        type=Path,
        default=Path("artifacts_scripts/survey/selected_books.csv"),
        help="Path to selected_books.csv",
    )
    parser.add_argument(
        "--survey-path",
        type=Path,
        default=Path("artifacts_scripts/survey/Survey_+Textbook+Proposal+Evaluation_December+19,+2025_10.55.csv"),
        help="Path to survey results CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts_scripts/survey/violin_responses.png"),
        help="Where to save the plot",
    )
    args = parser.parse_args()

    selected = load_selected_books(args.selected_path)
    questions, responses = load_survey(args.survey_path)
    data = gather_responses(selected, questions, responses)
    plot_violins(data, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
