#!/usr/bin/env python3
"""
Plot survey ratings vs. kernel percentile for the top/bottom (high/low) selected
books per topic. Can plot per-response points, per-topic means, or both.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.artist import Artist

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


def collect_mean_points(
    selected: List[BookRecord],
    means: Dict[Tuple[str, QuestionType], Optional[float]],
) -> Dict[QuestionType, Dict[str, List[Tuple[float, float]]]]:
    """Gather (kernel_percentile, mean_rating) pairs for high/low per question type."""
    points = {qt: {"high": [], "low": []} for qt in QUESTION_TYPES}
    by_topic: Dict[str, List[BookRecord]] = {}
    for rec in selected:
        by_topic.setdefault(rec.topic_id, []).append(rec)

    for records in by_topic.values():
        high, low, _ = choose_by_percentile(records)
        for label, rec in ("high", high), ("low", low):
            if not rec or rec.kernel_percentile is None:
                continue
            for qtype in QUESTION_TYPES:
                mean_val = means.get((rec.title, qtype))
                if mean_val is None:
                    continue
                points[qtype][label].append((rec.kernel_percentile, mean_val))
    return points


def collect_response_points(
    selected: List[BookRecord],
    questions,
    responses,
) -> Dict[QuestionType, Dict[str, List[Tuple[float, float]]]]:
    """Gather per-response (kernel_percentile, rating) pairs for high/low per question."""
    points = {qt: {"high": [], "low": []} for qt in QUESTION_TYPES}

    # Map titles to high/low records with kernel percentiles.
    by_topic: Dict[str, List[BookRecord]] = {}
    for rec in selected:
        by_topic.setdefault(rec.topic_id, []).append(rec)
    title_slot: Dict[str, Tuple[str, BookRecord]] = {}
    for records in by_topic.values():
        high, low, _ = choose_by_percentile(records)
        if high and high.kernel_percentile is not None:
            title_slot[high.title] = ("high", high)
        if low and low.kernel_percentile is not None:
            title_slot[low.title] = ("low", low)

    # Pull raw responses.
    for row in responses:
        for q in questions:
            slot = title_slot.get(q.title)
            if not slot:
                continue
            raw = row.get(q.column_name, "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            label, rec = slot
            points[q.question_type][label].append((rec.kernel_percentile, val))
    return points


def plot_kernel_vs_scores(
    means_points: Optional[Dict[QuestionType, Dict[str, List[Tuple[float, float]]]]],
    response_points: Optional[Dict[QuestionType, Dict[str, List[Tuple[float, float]]]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    legend_handles: Dict[str, Artist] = {}

    mean_style = {
        "high": {"color": "tab:red", "marker": "o", "edgecolor": "k", "linewidth": 0.5, "s": 60, "label": "High mean"},
        "low": {"color": "tab:blue", "marker": "s", "edgecolor": "k", "linewidth": 0.5, "s": 60, "label": "Low mean"},
    }
    resp_style = {
        "high": {"color": "tab:red", "marker": "o", "s": 18, "alpha": 0.35, "label": "High responses"},
        "low": {"color": "tab:blue", "marker": "s", "s": 18, "alpha": 0.35, "label": "Low responses"},
    }

    for ax, qtype, title in zip(
        axes,
        QUESTION_TYPES,
        ["Interesting?", "Well-written?", "Publishable?"],
    ):
        if response_points:
            for label, style in resp_style.items():
                xs, ys = zip(*response_points[qtype][label]) if response_points[qtype][label] else ([], [])
                handle = ax.scatter(xs, ys, **style)
                if label not in legend_handles:
                    legend_handles[label] = handle
        if means_points:
            for label, style in mean_style.items():
                xs, ys = zip(*means_points[qtype][label]) if means_points[qtype][label] else ([], [])
                handle = ax.scatter(xs, ys, **style)
                if label not in legend_handles:
                    legend_handles[label] = handle
        ax.set_title(title)
        ax.set_xlabel("Kernel percentile")
        # ax.set_xlim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Rating (1–7)")

    # Build a combined legend preserving order and uniqueness.
    handles_labels = []
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles.keys()), loc="upper center", ncol=2)

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
        default=Path("artifacts_scripts/survey/kernel_vs_scores.png"),
        help="Where to save the plot",
    )
    parser.add_argument(
        "--plot-means",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot per-topic mean points",
    )
    parser.add_argument(
        "--plot-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot individual response points",
    )
    args = parser.parse_args()

    selected = load_selected_books(args.selected_path)
    survey_questions, survey_responses = load_survey(args.survey_path)
    means = compute_question_means(survey_questions, survey_responses)

    means_points = collect_mean_points(selected, means) if args.plot_means else None
    response_points = (
        collect_response_points(selected, survey_questions, survey_responses)
        if args.plot_responses
        else None
    )

    if not means_points and not response_points:
        raise SystemExit("Nothing to plot: enable --plot-means and/or --plot-responses.")

    plot_kernel_vs_scores(means_points, response_points, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
