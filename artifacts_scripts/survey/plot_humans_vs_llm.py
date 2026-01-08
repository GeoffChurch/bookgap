#!/usr/bin/env python3
"""
Plot human survey means vs. LLM means (batched vs. unbatched) for each question type.
One figure with three subplots (Interesting?, Well-written?, Publishable?), showing
human mean on x-axis and LLM mean on y-axis with y = x reference.

python -m artifacts_scripts.survey.survey_llm artifacts_scripts/survey/survey.qsf --model hf:meta-llama/Llama-3.3-70B-Instruct --hf-4bit --prompt-mode standard --samples=2 --run
python -m artifacts_scripts.survey.plot_humans_vs_llm --llm-json survey_results.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from random import random
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt

from artifacts_scripts.survey.verify_survey_alignment import (
    QuestionType,
    compute_question_means,
    load_llm_results,
    load_survey,
)

QUESTION_TYPES: List[QuestionType] = ["interesting", "well_written", "publishable"]
QUESTION_TITLES = {
    "interesting": "Interesting?",
    "well_written": "Well-written?",
    "publishable": "Publishable?",
}


def format_variant_label(model: str, prompt_mode: str) -> str:
    if model == "Human":
        return "Human"
    if prompt_mode not in ("standard", "batch"):
        return f"{model} [{prompt_mode}]"
    return f"{model}"


def collect_scores(
    means: Dict[Tuple[str, QuestionType], float],
    qtype: QuestionType,
) -> List[float]:
    """Return all scores for a variant for a given question type."""
    vals: List[float] = []
    for (title, qt), val in means.items():
        if qt != qtype:
            continue
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return vals


def plot_humans_vs_llm(
    human_means: Dict[Tuple[str, QuestionType], Optional[float]],
    variants: Dict[Tuple[str, str], Dict[Tuple[str, QuestionType], float]],
    output: Path,
    title: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=False, sharey=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    # Treat humans as another variant cloud on the diagonal.
    human_variant = {
        (title, qtype): val
        for (title, qtype), val in human_means.items()
        if val is not None
    }
    human_entry = (("Human", "human"), human_variant)
    variant_items = [human_entry] + list(variants.items())

    variant_labels = [format_variant_label(model, prompt_mode) for (model, prompt_mode), _ in variant_items]

    # Single row: raw score clouds per variant.
    for ax, qtype in zip(axes, QUESTION_TYPES):
        for idx, ((model, prompt_mode), llm_means) in enumerate(variant_items):
            scores = collect_scores(llm_means, qtype)
            if not scores:
                continue
            jitter_x = [idx + 0.18 * (random() - 0.5) for _ in scores]
            color = color_cycle[idx % len(color_cycle)]
            ax.scatter(
                jitter_x,
                scores,
                color=color,
                edgecolor="k",
                linewidth=0.4,
                alpha=0.85,
                s=32,
                label=format_variant_label(model, prompt_mode) if qtype == QUESTION_TYPES[0] else None,
            )
            # Mean marker per variant.
            ax.hlines(sum(scores) / len(scores), idx - 0.22, idx + 0.22, colors=color, linestyles="-", linewidth=1.2)
        ax.set_xticks(range(len(variant_items)))
        ax.set_xticklabels(variant_labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(QUESTION_TITLES[qtype])
        ax.set_ylabel("Rating (1–7)")
        ax.set_ylim(0.75, 7.25)
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)))

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey-path",
        type=Path,
        default=Path("artifacts_scripts/survey/Survey_+Textbook+Proposal+Evaluation_December+19,+2025_10.55.csv"),
        help="Path to human survey CSV",
    )
    parser.add_argument(
        "--llm-json",
        type=Path,
        required=True,
        help="Path to LLM JSON results (survey_llm output)",
    )
    parser.add_argument(
        "--qsf-path",
        type=Path,
        default=Path("artifacts_scripts/survey/survey.qsf"),
        help="Path to the Qualtrics QSF used for the LLM run",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of model specs to include (e.g., hf:meta-llama/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts_scripts/survey/humans_vs_llm.png"),
        help="Where to save the plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    args = parser.parse_args()

    questions, responses = load_survey(args.survey_path)
    human_means = compute_question_means(questions, responses)

    llm_variants = load_llm_results(
        llm_path=args.llm_json,
        qsf_path=args.qsf_path,
        models=args.models,
    )
    if not llm_variants:
        raise SystemExit("No LLM variants found with provided filters.")

    title = args.title
    if title is None:
        title = "Humans vs. LLM"
        if args.models:
            title += f" ({', '.join(args.models)})"

    plot_humans_vs_llm(human_means, llm_variants, args.output, title=title)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
