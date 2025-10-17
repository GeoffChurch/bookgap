from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 5,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
    }
)

from .common import (
    GAP_COVERAGE_FRACTION_COLUMN,
    GAP_ZSCORE_COLUMN,
    PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN,
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


@dataclass(frozen=True)
class PlotFilenames:
    zscore_scatter: str
    llm_counts_scatter: str
    scatter_linear: str
    scatter_log: str
    violin_linear: str
    violin_log: str


@dataclass(frozen=True)
class PlotVariantSpec:
    name: str
    title_suffix: str
    sort_key: Callable[[RankFractionEntry], float]
    colorbar_label: str
    color_aggregator: Callable[[pd.Series], float]


@dataclass(frozen=True)
class PlotMetricSpec:
    column: str
    label: str
    filenames: PlotFilenames
    variants: Sequence[PlotVariantSpec]


def _mean_sort_key(entry: RankFractionEntry) -> float:
    return entry.mean_value


def _median_sort_key(entry: RankFractionEntry) -> float:
    return entry.median_value


def _create_variants(label: str) -> Sequence[PlotVariantSpec]:
    def _mean_aggregator(series: pd.Series) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            return 0.0
        return float(np.mean(cleaned))

    def _median_aggregator(series: pd.Series) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            return 0.0
        return float(np.median(cleaned))

    return (
        PlotVariantSpec(
            name="mean",
            title_suffix=" (sorted by mean)",
            sort_key=_mean_sort_key,
            colorbar_label=f"Mean {label}",
            color_aggregator=_mean_aggregator,
        ),
        PlotVariantSpec(
            name="median",
            title_suffix=" (sorted by median)",
            sort_key=_median_sort_key,
            colorbar_label=f"Median {label}",
            color_aggregator=_median_aggregator,
        ),
    )


def _build_metric_specs() -> Sequence[PlotMetricSpec]:
    return (
        PlotMetricSpec(
            column=PROPOSAL_BOOK_RANK_FRACTION_COLUMN,
            label="Proposal Book Rank Fraction",
            filenames=PlotFilenames(
                zscore_scatter="gap_book_rank_fraction_scatter.png",
                llm_counts_scatter="book_rank_fraction_by_llm_counts.png",
                scatter_linear="book_rank_fraction_scatter_linear.png",
                scatter_log="book_rank_fraction_scatter_log.png",
                violin_linear="book_rank_fraction_violin_linear.png",
                violin_log="book_rank_fraction_violin_log.png",
            ),
            variants=_create_variants("Proposal Book Rank Fraction"),
        ),
        PlotMetricSpec(
            column=PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN,
            label="Proposal Article Rank Fraction",
            filenames=PlotFilenames(
                zscore_scatter="gap_article_rank_fraction_scatter.png",
                llm_counts_scatter="article_rank_fraction_by_llm_counts.png",
                scatter_linear="article_rank_fraction_scatter_linear.png",
                scatter_log="article_rank_fraction_scatter_log.png",
                violin_linear="article_rank_fraction_violin_linear.png",
                violin_log="article_rank_fraction_violin_log.png",
            ),
            variants=_create_variants("Proposal Article Rank Fraction"),
        ),
        PlotMetricSpec(
            column=GAP_COVERAGE_FRACTION_COLUMN,
            label="Gap Coverage Fraction",
            filenames=PlotFilenames(
                zscore_scatter="gap_coverage_fraction_gap_zscore_scatter.png",
                llm_counts_scatter="gap_coverage_fraction_by_llm_counts.png",
                scatter_linear="gap_coverage_fraction_scatter_linear.png",
                scatter_log="gap_coverage_fraction_scatter_log.png",
                violin_linear="gap_coverage_fraction_violin_linear.png",
                violin_log="gap_coverage_fraction_violin_log.png",
            ),
            variants=_create_variants("Gap Coverage Fraction"),
        ),
        PlotMetricSpec(
            column=GAP_ZSCORE_COLUMN,
            label="Gap Z-Score",
            filenames=PlotFilenames(
                zscore_scatter="gap_zscore_gap_zscore_scatter.png",
                llm_counts_scatter="gap_zscore_by_llm_counts.png",
                scatter_linear="gap_zscore_scatter_linear.png",
                scatter_log="gap_zscore_scatter_log.png",
                violin_linear="gap_zscore_violin_linear.png",
                violin_log="gap_zscore_violin_log.png",
            ),
            variants=_create_variants("Gap Z-Score"),
        ),
    )


METRIC_SPECS: Sequence[PlotMetricSpec] = _build_metric_specs()


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
        f"{denoiser}\n"
        f"k={drop_k}\n"
        f"{embedder_str}\n"
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


def collect_run_points(outputs_dir: Path, metric: PlotMetricSpec) -> List[RunPlotData]:
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
            metric.column,
        }
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            LOGGER.warning("Missing columns %s in %s", ", ".join(missing_columns), run_dir)
            continue

        requested_columns = [GAP_ZSCORE_COLUMN, PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
        if metric.column not in requested_columns:
            requested_columns.append(metric.column)

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
    aggregator: Callable[[pd.Series], float],
    title: str,
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
        mean_rank_fraction = float(aggregator(df[value_column]))
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
    ax.set_title(title)
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


def _sorted_rank_fraction_entries(
    run_points: List[RunPlotData],
    value_column: str,
    sort_key: Callable[[RankFractionEntry], float],
) -> List[RankFractionEntry]:
    entries = build_rank_fraction_entries(run_points, value_column)
    entries.sort(key=lambda entry: (sort_key(entry), entry.median_value, entry.mean_value, entry.label))
    return entries


def _build_rank_fraction_scatter_plot(
    entries: List[RankFractionEntry],
    ylabel: str,
) -> tuple[Figure, Axes, List[float]]:
    figure_width = max(10, len(entries) * 0.6) if entries else 10
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    all_values: List[float] = []
    for idx, entry in enumerate(entries):
        values = entry.values
        if not values:
            continue
        all_values.extend(values)
        n = len(values)
        jitter = np.linspace(-0.2, 0.2, n) if n > 1 else np.zeros(1)
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
        ax.scatter(
            idx + 1,
            entry.mean_value,
            marker="D",
            s=70,
            color="#E74C3C",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )
        ax.scatter(
            idx + 1,
            entry.median_value,
            marker="P",
            s=80,
            color="#2ECC71",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )

    labels = [entry.label for entry in entries]
    ax.set_xticks(range(1, len(labels) + 1))
    tick_texts = ax.set_xticklabels(labels)
    for text in tick_texts:
        text.set_fontsize(6)
        text.set_horizontalalignment("center")

    ax.set_ylabel(ylabel)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#E74C3C",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="P",
            color="none",
            markerfacecolor="#2ECC71",
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
            label="Median",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="Summary Markers", fontsize="small")
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5, axis="y")

    return fig, ax, all_values


def plot_rank_fraction_scatter_pair(
    run_points: List[RunPlotData],
    output_linear: Path,
    output_log: Path,
    dpi: int,
    value_column: str,
    ylabel: str,
    base_title: str,
    sort_key: Callable[[RankFractionEntry], float],
) -> None:
    entries = _sorted_rank_fraction_entries(run_points, value_column, sort_key)
    if not entries:
        LOGGER.warning("No %s data for scatter plot. Skipping generation.", ylabel)
        return

    fig, ax, all_values = _build_rank_fraction_scatter_plot(entries, ylabel)

    output_linear.parent.mkdir(parents=True, exist_ok=True)
    title_linear = _apply_yscale(ax, "linear", all_values, base_title)
    ax.set_title(title_linear)
    fig.tight_layout()
    fig.savefig(output_linear, dpi=dpi)
    LOGGER.info("Wrote scatter plot to %s", output_linear)

    output_log.parent.mkdir(parents=True, exist_ok=True)
    title_log = _apply_yscale(ax, "log", all_values, base_title)
    ax.set_title(title_log)
    fig.tight_layout()
    fig.savefig(output_log, dpi=dpi)
    LOGGER.info("Wrote scatter plot to %s", output_log)

    plt.close(fig)


def plot_rank_fraction_violin(
    run_points: List[RunPlotData],
    output_file: Path,
    dpi: int,
    value_column: str,
    yscale: str,
    ylabel: str,
    base_title: str,
    sort_key: Callable[[RankFractionEntry], float],
) -> None:
    entries = _sorted_rank_fraction_entries(run_points, value_column, sort_key)
    if not entries:
        LOGGER.warning("No %s data for violin plot. Skipping generation.", ylabel)
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
    title = _apply_yscale(ax, yscale, flattened, base_title)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, linestyle="--", alpha=0.5, axis="y")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    LOGGER.info("Wrote violin plot to %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create plots for BookGap runs. For each metric column, generates gap z-score scatter, "
            "LLM counts scatter, and rank-fraction scatter/violin plots sorted by mean and median."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the root outputs directory containing run subdirectories.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help=(
            "Directory where plot artifacts will be written. Each metric column gets its own "
            "subdirectory containing 'mean' and 'median' folders."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        help="Optional list of metric columns to plot (defaults to all known metrics).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution (dots per inch) for generated plots.",
    )
    return parser.parse_args()


def select_metric_specs(requested_columns: Optional[Sequence[str]]) -> List[PlotMetricSpec]:
    if not requested_columns:
        return list(METRIC_SPECS)

    column_to_spec = {spec.column: spec for spec in METRIC_SPECS}
    selected: List[PlotMetricSpec] = []
    for column in requested_columns:
        spec = column_to_spec.get(column)
        if spec is None:
            LOGGER.warning("Unknown metric column '%s'; skipping.", column)
            continue
        selected.append(spec)
    return selected


def generate_metric_variant_plots(
    run_points: List[RunPlotData],
    metric: PlotMetricSpec,
    variant: PlotVariantSpec,
    metric_dir: Path,
    dpi: int,
) -> None:
    variant_dir = metric_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_label = f"{metric.label}{variant.title_suffix}"
    gap_vs_title = f"Gap Z-Score vs {variant_label}"
    llm_title = f"{variant_label} by LLM Counts"
    scatter_title = f"{variant_label} by Run"
    distribution_title = f"Distribution of {variant_label} by Run"

    plot_points(
        run_points,
        variant_dir / metric.filenames.zscore_scatter,
        dpi,
        metric.column,
        metric.label,
        gap_vs_title,
    )

    plot_llm_counts_scatter(
        run_points,
        variant_dir / metric.filenames.llm_counts_scatter,
        dpi,
        metric.column,
        variant.colorbar_label,
        variant.color_aggregator,
        llm_title,
    )

    plot_rank_fraction_scatter_pair(
        run_points,
        variant_dir / metric.filenames.scatter_linear,
        variant_dir / metric.filenames.scatter_log,
        dpi,
        metric.column,
        metric.label,
        scatter_title,
        variant.sort_key,
    )

    plot_rank_fraction_violin(
        run_points,
        variant_dir / metric.filenames.violin_linear,
        dpi,
        metric.column,
        "linear",
        metric.label,
        distribution_title,
        variant.sort_key,
    )

    plot_rank_fraction_violin(
        run_points,
        variant_dir / metric.filenames.violin_log,
        dpi,
        metric.column,
        "log",
        metric.label,
        distribution_title,
        variant.sort_key,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    specs = select_metric_specs(args.metrics)
    if not specs:
        LOGGER.warning("No valid metrics selected; nothing to plot.")
        return

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    for metric in specs:
        LOGGER.info("Collecting data for metric %s", metric.column)
        run_points = collect_run_points(args.outputs_dir, metric)
        if not run_points:
            LOGGER.warning("No data found for metric %s; skipping.", metric.column)
            continue
        metric_dir = args.artifacts_dir / metric.column
        for variant in metric.variants:
            LOGGER.info(
                "Generating plots for metric %s sorted by %s", metric.column, variant.name
            )
            generate_metric_variant_plots(run_points, metric, variant, metric_dir, args.dpi)


if __name__ == "__main__":
    main()
