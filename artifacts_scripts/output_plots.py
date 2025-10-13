from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, List, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

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


@dataclass
class RankFractionEntry:
    mean_value: float
    median_value: float
    values: List[float]
    label: str


def _format_metadata_label(metadata: Optional[RunMetadata], fallback_label: str) -> str:
    if metadata is None:
        return fallback_label

    denoiser = metadata.denoiser or "UNKNOWN"
    drop_k = metadata.drop_k if metadata.drop_k is not None else "N/A"
    embedder = metadata.embedder_model or "UNKNOWN"
    embedder_str = str(embedder)
    if "/" in embedder_str:
        embedder_str = embedder_str.split("/")[-1]
    num_books = metadata.num_books if metadata.num_books is not None else "?"
    num_articles = metadata.num_articles if metadata.num_articles is not None else "?"
    return (
        f"den={denoiser}\n"
        f"k={drop_k}\n"
        f"embed={embedder_str}\n"
        f"books={num_books}\n"
        f"articles={num_articles}"
    )


def _extract_values(series: pd.Series) -> List[float]:
    cleaned = series.dropna()
    return cleaned.tolist()


def build_rank_fraction_entries(
    run_points: List[RunPlotData],
    value_column: str,
) -> List[RankFractionEntry]:
    entries: List[RankFractionEntry] = []
    for run_data in run_points:
        if value_column not in run_data.dataframe.columns:
            continue
        values = _extract_values(run_data.dataframe[value_column])
        if not values:
            continue
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        label = _format_metadata_label(run_data.metadata, run_data.label)
        entries.append(
            RankFractionEntry(
                mean_value=mean_value,
                median_value=median_value,
                values=values,
                label=label,
            )
        )

    entries.sort(key=lambda entry: (entry.median_value, entry.mean_value))
    return entries


def _apply_yscale(ax: Axes, yscale: str, values: List[float], base_title: str) -> str:
    if yscale.lower() == "log":
        if values:
            min_value = min(values)
        else:
            min_value = None
        if min_value is not None and min_value <= 0:
            LOGGER.warning(
                "Encountered non-positive values (minimum %.3g); falling back to symlog scale.",
                min_value,
            )
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_ylim(bottom=0)
            return f"{base_title} (symlog)"
        ax.set_yscale("log")
        return f"{base_title} (log scale)"

    ax.set_yscale("linear")
    return base_title


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

        required_columns = {
            GAP_ZSCORE_COLUMN,
            PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        }
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            LOGGER.warning("Missing columns %s in %s", ", ".join(missing_columns), run_dir)
            continue

        requested_columns = [GAP_ZSCORE_COLUMN, PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
        if "proposal_article_rank_fraction" in df.columns:
            requested_columns.append("proposal_article_rank_fraction")

        filtered_df = df.loc[mask, requested_columns].dropna(
            subset=[col for col in requested_columns if col != GAP_ZSCORE_COLUMN]
        )
        if filtered_df.empty:
            LOGGER.warning("No valid rows to plot in %s", run_dir)
            continue

        run_label = str(relative_run_path(run_dir, outputs_dir))
        run_points.append(RunPlotData(run_label, filtered_df, metadata))

    return run_points


def plot_points(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    if not run_points:
        LOGGER.warning("No points available to plot. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = plt.get_cmap("tab20", len(run_points))
    for idx, run_data in enumerate(run_points):
        label = run_data.label
        df = run_data.dataframe
        color = cmap(idx % cmap.N)
        if value_column not in df.columns:
            continue
        ax.scatter(
            df[GAP_ZSCORE_COLUMN],
            df[value_column],
            label=label,
            s=20,
            alpha=0.7,
            color=color,
            edgecolors="none",
        )

    ax.set_xlabel("Gap Z-Score")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    value_column: str,
    colorbar_label: str,
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

        if value_column not in df.columns:
            continue
        mean_rank_fraction = float(df[value_column].mean())
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
    colorbar.set_label(colorbar_label)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote LLM scatter plot to %s", output_file)


def plot_rank_fraction_violin(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
    value_column: str,
    yscale: str,
    ylabel: str,
    base_title: str,
) -> None:
    entries = build_rank_fraction_entries(run_points, value_column)

    if not entries:
        LOGGER.warning("No proposal book rank fraction data for violin plot. Skipping generation.")
        return

    data: List[List[float]] = [entry.values for entry in entries]
    labels: List[str] = [entry.label for entry in entries]

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

    flattened = [value for entry in data for value in entry]
    flattened = [value for entry in data for value in entry]
    title = _apply_yscale(ax, yscale, flattened, base_title)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5, axis="y")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Wrote violin plot to %s", output_file)


def plot_rank_fraction_scatter(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
    value_column: str,
    yscale: str,
    ylabel: str,
    base_title: str,
    add_summary_markers: bool = False,
) -> None:
    entries = build_rank_fraction_entries(run_points, value_column)
    if not entries:
        LOGGER.warning("No proposal book rank fraction data for scatter plot. Skipping generation.")
        return

    figure_width = max(10, len(entries) * 0.6)
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    all_values: List[float] = []
    mean_handle: Optional[Line2D] = None
    median_handle: Optional[Line2D] = None
    for idx, entry in enumerate(entries):
        values = entry.values
        if not values:
            continue
        all_values.extend(values)
        n = len(values)
        if n > 1:
            jitter = np.linspace(-0.2, 0.2, n)
        else:
            jitter = np.zeros(1)
        x_positions = idx + 1 + jitter
        ax.scatter(
            x_positions,
            values,
            alpha=0.6,
            s=25,
            color="#5DADE2",
            edgecolors="black",
            linewidths=0.3,
        )

        if add_summary_markers:
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            ax.scatter(
                idx + 1,
                mean_value,
                marker="D",
                s=70,
                color="#E74C3C",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
            )
            ax.scatter(
                idx + 1,
                median_value,
                marker="P",
                s=80,
                color="#2ECC71",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
            )
            if mean_handle is None:
                mean_handle = Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="none",
                    markerfacecolor="#E74C3C",
                    markeredgecolor="black",
                    markersize=7,
                    linestyle="None",
                    label="Mean",
                )
            if median_handle is None:
                median_handle = Line2D(
                    [0],
                    [0],
                    marker="P",
                    color="none",
                    markerfacecolor="#2ECC71",
                    markeredgecolor="black",
                    markersize=8,
                    linestyle="None",
                    label="Median",
                )

    labels = [entry.label for entry in entries]
    ax.set_xticks(range(1, len(labels) + 1))
    tick_texts = ax.set_xticklabels(labels)
    for text in tick_texts:
        text.set_fontsize(6)
        text.set_horizontalalignment("center")

    title = _apply_yscale(ax, yscale, all_values, base_title)

    ax.set_ylabel(ylabel)

    if add_summary_markers:
        handles: List[Line2D] = []
        if mean_handle is not None:
            handles.append(mean_handle)
        if median_handle is not None:
            handles.append(median_handle)
        if handles:
            ax.legend(handles=handles, loc="upper right", title="Summary Markers", fontsize="small")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5, axis="y")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Wrote scatter plot to %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots for BookGap runs: gap_zscore versus proposal_book_rank_fraction, "
            "mean rank fraction by LLM book/article counts, and per-run rank fraction distributions "
            "rendered as scatter and violin plots with optional log-scale variants."
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
        default=Path("artifacts") / "gap_book_rank_fraction_scatter.png",
        help="Where to write the scatter plot (books).",
    )
    parser.add_argument(
        "--llm-counts-output-file",
        type=Path,
        default=Path("artifacts") / "book_rank_fraction_by_llm_counts.png",
        help="Where to write the scatter plot colored by book rank fraction and keyed by LLM counts.",
    )
    parser.add_argument(
        "--rank-fraction-scatter-file",
        type=Path,
        default=Path("artifacts") / "book_rank_fraction_scatter_linear.png",
        help="Where to write the per-topic book rank fraction scatter plot (linear y-scale).",
    )
    parser.add_argument(
        "--rank-fraction-scatter-log-file",
        type=Path,
        default=Path("artifacts") / "book_rank_fraction_scatter_log.png",
        help="Where to write the per-topic book rank fraction scatter plot (log y-scale).",
    )
    parser.add_argument(
        "--rank-fraction-violin-file",
        type=Path,
        default=Path("artifacts") / "book_rank_fraction_violin_linear.png",
        help="Where to write the per-run book rank fraction violin plot (linear y-scale).",
    )
    parser.add_argument(
        "--rank-fraction-violin-log-file",
        type=Path,
        default=Path("artifacts") / "book_rank_fraction_violin_log.png",
        help="Where to write the per-run book rank fraction violin plot (log y-scale).",
    )
    parser.add_argument(
        "--article-output-file",
        type=Path,
        default=Path("artifacts") / "gap_article_rank_fraction_scatter.png",
        help="Where to write the article scatter plot.",
    )
    parser.add_argument(
        "--article-llm-counts-output-file",
        type=Path,
        default=Path("artifacts") / "article_rank_fraction_by_llm_counts.png",
        help="Where to write the article scatter plot keyed by LLM counts.",
    )
    parser.add_argument(
        "--article-rank-fraction-scatter-file",
        type=Path,
        default=Path("artifacts") / "article_rank_fraction_scatter_linear.png",
        help="Where to write the per-topic article rank fraction scatter plot (linear y-scale).",
    )
    parser.add_argument(
        "--article-rank-fraction-scatter-log-file",
        type=Path,
        default=Path("artifacts") / "article_rank_fraction_scatter_log.png",
        help="Where to write the per-topic article rank fraction scatter plot (log y-scale).",
    )
    parser.add_argument(
        "--article-rank-fraction-violin-file",
        type=Path,
        default=Path("artifacts") / "article_rank_fraction_violin_linear.png",
        help="Where to write the per-run article rank fraction violin plot (linear y-scale).",
    )
    parser.add_argument(
        "--article-rank-fraction-violin-log-file",
        type=Path,
        default=Path("artifacts") / "article_rank_fraction_violin_log.png",
        help="Where to write the per-run article rank fraction violin plot (log y-scale).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (dots per inch) for the generated plot.",
    )
    return parser.parse_args()


def _generate_book_plots(run_points: List[RunPlotData], args: argparse.Namespace) -> None:
    plot_points(
        run_points,
        args.output_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "Proposal Book Rank Fraction",
        "Gap Z-Score vs Proposal Book Rank Fraction by Run",
    )
    plot_llm_counts_scatter(
        run_points,
        args.llm_counts_output_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "Mean Proposal Book Rank Fraction",
    )
    plot_rank_fraction_scatter(
        run_points,
        args.rank_fraction_scatter_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "linear",
        "Proposal Book Rank Fraction",
        "Proposal Book Rank Fraction by Run",
    )
    plot_rank_fraction_scatter(
        run_points,
        args.rank_fraction_scatter_log_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "log",
        "Proposal Book Rank Fraction",
        "Proposal Book Rank Fraction by Run",
        add_summary_markers=True,
    )
    plot_rank_fraction_violin(
        run_points,
        args.rank_fraction_violin_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "linear",
        "Proposal Book Rank Fraction",
        "Distribution of Proposal Book Rank Fraction by Run",
    )
    plot_rank_fraction_violin(
        run_points,
        args.rank_fraction_violin_log_file,
        args.dpi,
        PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
        "log",
        "Proposal Book Rank Fraction",
        "Distribution of Proposal Book Rank Fraction by Run",
    )


def _generate_article_plots(run_points: List[RunPlotData], args: argparse.Namespace) -> None:
    column = "proposal_article_rank_fraction"
    ylabel = "Proposal Article Rank Fraction"
    title_prefix = "Proposal Article Rank Fraction"
    plot_points(
        run_points,
        args.article_output_file,
        args.dpi,
        column,
        ylabel,
        "Gap Z-Score vs Proposal Article Rank Fraction by Run",
    )
    plot_llm_counts_scatter(
        run_points,
        args.article_llm_counts_output_file,
        args.dpi,
        column,
        "Mean Proposal Article Rank Fraction",
    )
    plot_rank_fraction_scatter(
        run_points,
        args.article_rank_fraction_scatter_file,
        args.dpi,
        column,
        "linear",
        ylabel,
        f"{title_prefix} by Run",
    )
    plot_rank_fraction_scatter(
        run_points,
        args.article_rank_fraction_scatter_log_file,
        args.dpi,
        column,
        "log",
        ylabel,
        f"{title_prefix} by Run",
        add_summary_markers=True,
    )
    plot_rank_fraction_violin(
        run_points,
        args.article_rank_fraction_violin_file,
        args.dpi,
        column,
        "linear",
        ylabel,
        f"Distribution of {title_prefix} by Run",
    )
    plot_rank_fraction_violin(
        run_points,
        args.article_rank_fraction_violin_log_file,
        args.dpi,
        column,
        "log",
        ylabel,
        f"Distribution of {title_prefix} by Run",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    run_points = collect_run_points(args.outputs_dir)
    _generate_book_plots(run_points, args)
    _generate_article_plots(run_points, args)


if __name__ == "__main__":
    main()
