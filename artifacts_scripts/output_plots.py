from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, List, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .common import (
    GAP_ZSCORE_COLUMN,
    PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
    RunMetadata,
    find_run_directories,
    load_run_metadata,
    load_results_dataframe,
    relative_run_path,
    valid_rank_fraction_mask,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class RunPlotData:
    label: str
    dataframe: pd.DataFrame
    metadata: Optional[RunMetadata]


def collect_run_points(outputs_dir: Path) -> List[RunPlotData]:
    run_points: List[RunPlotData] = []

    for run_dir in find_run_directories(outputs_dir):
        metadata = load_run_metadata(run_dir)
        df = load_results_dataframe(run_dir)
        if df is None:
            continue

        try:
            mask = valid_rank_fraction_mask(df)
        except KeyError as exc:
            LOGGER.warning("%s in %s", exc, run_dir)
            continue

        required_columns = {GAP_ZSCORE_COLUMN, PROPOSAL_BOOK_RANK_FRACTION_COLUMN}
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            LOGGER.warning("Missing columns %s in %s", ", ".join(missing_columns), run_dir)
            continue

        filtered_df = df.loc[mask, [GAP_ZSCORE_COLUMN, PROPOSAL_BOOK_RANK_FRACTION_COLUMN]].dropna()
        if filtered_df.empty:
            LOGGER.warning("No valid rows to plot in %s", run_dir)
            continue

        run_label = str(relative_run_path(run_dir, outputs_dir))
        run_points.append(RunPlotData(run_label, filtered_df, metadata))

    return run_points


def plot_points(run_points: List[RunPlotData], output_file: Path, dpi: int) -> None:
    if not run_points:
        LOGGER.warning("No points available to plot. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = plt.get_cmap("tab20", len(run_points))
    for idx, run_data in enumerate(run_points):
        label = run_data.label
        df = run_data.dataframe
        color = cmap(idx % cmap.N)
        ax.scatter(
            df[GAP_ZSCORE_COLUMN],
            df[PROPOSAL_BOOK_RANK_FRACTION_COLUMN],
            label=label,
            s=20,
            alpha=0.7,
            color=color,
            edgecolors="none",
        )

    ax.set_xlabel("Gap Z-Score")
    ax.set_ylabel("Proposal Book Rank Fraction")
    ax.set_title("Gap Z-Score vs Proposal Book Rank Fraction by Run")
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize="small", title="Run", frameon=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote plot to %s", output_file)


def plot_llm_counts_scatter(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
) -> None:
    marker_cycle = cycle(["o", "s", "^", "D", "v", "P", "X", "*", ">", "<", "H"])
    marker_for_config: dict[tuple, str] = {}
    legend_handles: dict[tuple, Line2D] = {}
    plotted_points = []

    for run_data in run_points:
        metadata = run_data.metadata
        if metadata is None:
            LOGGER.warning("Skipping %s: missing metadata for LLM scatter", run_data.label)
            continue
        if metadata.num_books is None or metadata.num_articles is None:
            LOGGER.warning(
                "Skipping %s: missing num_books/num_articles in metadata", run_data.label
            )
            continue

        df = run_data.dataframe
        if df.empty:
            continue

        mean_rank_fraction = float(df[PROPOSAL_BOOK_RANK_FRACTION_COLUMN].mean())
        config_key = (
            metadata.denoiser or "UNKNOWN",
            metadata.drop_k,
            metadata.embedder_model or "UNKNOWN",
        )
        plotted_points.append(
            (metadata.num_books, metadata.num_articles, mean_rank_fraction, config_key, run_data.label)
        )

    if not plotted_points:
        LOGGER.warning("No runs available for LLM scatter plot. Skipping generation.")
        return

    fractions = [point[2] for point in plotted_points]
    min_fraction = min(fractions)
    max_fraction = max(fractions)
    if min_fraction == max_fraction:
        min_fraction -= 0.01
        max_fraction += 0.01

    norm = Normalize(vmin=min_fraction, vmax=max_fraction)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 6))

    for num_books, num_articles, fraction, config_key, label in plotted_points:
        marker = marker_for_config.setdefault(config_key, next(marker_cycle))
        ax.scatter(
            num_books,
            num_articles,
            c=[fraction],
            marker=marker,
            cmap=cmap,
            norm=norm,
            s=120,
            edgecolors="black",
            linewidths=0.5,
        )

        if config_key not in legend_handles:
            denoiser, drop_k, embedder = config_key
            label_text = (
                f"denoiser={denoiser}, drop_k={drop_k if drop_k is not None else 'N/A'}, "
                f"embedder={embedder}"
            )
            legend_handles[config_key] = Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markeredgecolor="black",
                linestyle="None",
                markersize=8,
                label=label_text,
            )

    ax.set_xlabel("LLM Number of Books")
    ax.set_ylabel("LLM Number of Articles")
    ax.set_title("Mean Proposal Book Rank Fraction by LLM Counts")
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5)

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            title="Denoiser / drop_k / Embedder",
            fontsize="small",
            loc="best",
        )

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Mean Proposal Book Rank Fraction")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote LLM scatter plot to %s", output_file)


def plot_rank_fraction_violin(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
) -> None:
    plot_entries: List[tuple[float, List[float], Optional[RunMetadata], str]] = []

    for run_data in run_points:
        series = run_data.dataframe[PROPOSAL_BOOK_RANK_FRACTION_COLUMN].dropna()
        if series.empty:
            continue
        values = series.tolist()
        mean_value = float(series.mean())
        metadata = run_data.metadata
        plot_entries.append((mean_value, values, metadata, run_data.label))

    if not plot_entries:
        LOGGER.warning("No proposal book rank fraction data for violin plot. Skipping generation.")
        return

    plot_entries.sort(key=lambda entry: entry[0])

    data: List[List[float]] = [entry[1] for entry in plot_entries]
    labels: List[str] = []

    for _, _, metadata, fallback_label in plot_entries:
        if metadata is None:
            labels.append(fallback_label)
            continue

        denoiser = metadata.denoiser or "UNKNOWN"
        drop_k = metadata.drop_k if metadata.drop_k is not None else "N/A"
        embedder = metadata.embedder_model or "UNKNOWN"
        if isinstance(embedder, str) and "/" in embedder:
            embedder = embedder.split("/")[-1]
        num_books = metadata.num_books if metadata.num_books is not None else "?"
        num_articles = metadata.num_articles if metadata.num_articles is not None else "?"
        label = (
            f"den={denoiser}\n"
            f"k={drop_k}\n"
            f"embed={embedder}\n"
            f"books={num_books}\n"
            f"articles={num_articles}"
        )
        labels.append(label)

    figure_width = max(10, len(data) * 0.6)
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    parts = cast(dict[str, Any], ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False))

    bodies_obj = parts.get("bodies")
    if isinstance(bodies_obj, list):
        bodies = bodies_obj
    elif bodies_obj is None:
        bodies = []
    else:
        bodies = [bodies_obj]

    for body in bodies:
        body.set_facecolor("#5DADE2")
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    for key in ("cmeans", "cmedians"):
        if key in parts:
            collection = parts[key]
            collection.set_edgecolor("black")
            collection.set_linewidth(1.2)

    ax.set_xticks(range(1, len(labels) + 1))
    tick_texts = ax.set_xticklabels(labels)
    for text in tick_texts:
        text.set_fontsize(6)
        text.set_horizontalalignment("center")

    ax.set_ylabel("Proposal Book Rank Fraction")
    ax.set_title("Distribution of Proposal Book Rank Fraction by Run")
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5, axis="y")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Wrote violin plot to %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create scatter plots for BookGap runs: gap_zscore versus proposal_book_rank_fraction, "
            "mean rank fraction by LLM book/article counts, and per-run rank fraction distributions."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the root outputs directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts") / "gap_rank_fraction_scatter.png",
        help="Where to write the scatter plot image.",
    )
    parser.add_argument(
        "--llm-counts-output-file",
        type=Path,
        default=Path("artifacts") / "rank_fraction_by_llm_counts.png",
        help="Where to write the scatter plot colored by rank fraction and keyed by LLM counts.",
    )
    parser.add_argument(
        "--rank-fraction-violin-file",
        type=Path,
        default=Path("artifacts") / "rank_fraction_violin.png",
        help="Where to write the per-run proposal book rank fraction violin plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (dots per inch) for the generated plot.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    run_points = collect_run_points(args.outputs_dir)
    plot_points(run_points, args.output_file, args.dpi)
    plot_llm_counts_scatter(run_points, args.llm_counts_output_file, args.dpi)
    plot_rank_fraction_violin(run_points, args.rank_fraction_violin_file, args.dpi)


if __name__ == "__main__":
    main()
