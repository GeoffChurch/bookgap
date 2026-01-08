#!/usr/bin/env python3
"""
Plot per-topic mean ratings: ours vs. high and ours vs. low (two subplots with y=x).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from verify_survey_alignment import (
    BookRecord,
    QuestionType,
    choose_by_percentile,
    compute_question_means,
    load_selected_books,
    load_survey,
)

QUESTION_TYPES: List[QuestionType] = ["interesting", "well_written", "publishable"]
QUESTION_TITLES = {
    "interesting": "Interesting?",
    "well_written": "Well-written?",
    "publishable": "Publishable?",
}
COLORS = {
    "interesting": "tab:orange",
    "well_written": "tab:green",
    "publishable": "tab:purple",
}
MARKERS = {
    "interesting": "o",
    "well_written": "s",
    "publishable": "^",
}


def collect_pairs(
    selected: List[BookRecord],
    means: Dict[Tuple[str, QuestionType], Optional[float]],
) -> Tuple[
    Dict[QuestionType, List[Tuple[float, float]]],
    Dict[QuestionType, List[Tuple[float, float]]],
    Dict[QuestionType, List[Tuple[float, float]]],
]:
    """Return per-question pairs: ours vs high, ours vs low, and low vs high."""
    by_topic: Dict[str, List[BookRecord]] = {}
    for rec in selected:
        by_topic.setdefault(rec.topic_id, []).append(rec)

    ours_vs_high = {qt: [] for qt in QUESTION_TYPES}
    ours_vs_low = {qt: [] for qt in QUESTION_TYPES}
    low_vs_high = {qt: [] for qt in QUESTION_TYPES}

    for records in by_topic.values():
        high, low, generated = choose_by_percentile(records)
        if not generated:
            continue
        for qtype in QUESTION_TYPES:
            ours = means.get((generated.title, qtype))
            if ours is None:
                continue
            high_val: Optional[float] = None
            low_val: Optional[float] = None
            if high:
                high_val = means.get((high.title, qtype))
                if high_val is not None:
                    ours_vs_high[qtype].append((ours, high_val))
            if low:
                low_val = means.get((low.title, qtype))
                if low_val is not None:
                    ours_vs_low[qtype].append((ours, low_val))
            if high_val is not None and low_val is not None:
                low_vs_high[qtype].append((low_val, high_val))

    return ours_vs_high, ours_vs_low, low_vs_high


def plot_pairs(
    ours_vs_high: Dict[QuestionType, List[Tuple[float, float]]],
    ours_vs_low: Dict[QuestionType, List[Tuple[float, float]]],
    low_vs_high: Dict[QuestionType, List[Tuple[float, float]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, pairs, title, xlabel, ylabel in [
        (axes[0], ours_vs_high, "Ours vs. High", "Ours mean", "High mean"),
        (axes[1], ours_vs_low, "Ours vs. Low", "Ours mean", "Low mean"),
        (axes[2], low_vs_high, "Low vs. High", "Low mean", "High mean"),
    ]:
        for qtype in QUESTION_TYPES:
            pts = pairs[qtype]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, color=COLORS[qtype], marker=MARKERS[qtype], alpha=0.8, label=QUESTION_TITLES[qtype])
        ax.plot([1, 7], [1, 7], color="gray", linestyle="--", linewidth=1, label="y = x")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_xlim(1, 7)
    axes[0].set_ylim(1, 7)
    # Build a unified legend from left axis to avoid duplicates.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left")

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
        default=Path("artifacts_scripts/survey/ours_vs_baselines.png"),
        help="Where to save the plot",
    )
    args = parser.parse_args()

    selected = load_selected_books(args.selected_path)
    questions, responses = load_survey(args.survey_path)
    means = compute_question_means(questions, responses)

    ours_vs_high, ours_vs_low, low_vs_high = collect_pairs(selected, means)
    plot_pairs(ours_vs_high, ours_vs_low, low_vs_high, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
