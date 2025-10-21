from __future__ import annotations

import argparse
import logging
import numbers
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import pandas as pd
from typing import TYPE_CHECKING, cast

from artifacts_scripts.common import (
    EMBEDDER_DISPLAY_NAME_MAP,
    friendly_column_name,
    friendly_embedder_name,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyalex as _pyalex_module
else:  # pragma: no cover - runtime import
    _pyalex_module = None
try:  # pragma: no cover - import guard
    import pyalex as _pyalex_runtime
except ModuleNotFoundError as exc:  # pragma: no cover - external dependency
    _pyalex_runtime = None
    PYALEX_IMPORT_ERROR = exc
else:
    PYALEX_IMPORT_ERROR = None

pyalex = cast(Any, _pyalex_runtime)


LOGGER = logging.getLogger(__name__)

DEFAULT_PYALEX_EMAIL = "gchurchill@cs.stonybrook.edu"
TOPIC_BATCH_SIZE = 25
TOPIC_DISPLAY_NAME_CACHE: dict[str, str] = {}


SUMMARY_COLUMNS: Iterable[str] = (
    "run",
    "embedder_model",
    "denoiser",
    "drop_k",
    "llm_num_articles",
    "llm_num_books",
    "proposal_book_rank_fraction_mean",
    "proposal_article_rank_fraction_mean",
    "proposal_article_rank_fraction_median",
    "proposal_book_rank_fraction_std",
    "row_count",
    "gap_coverage_fraction_mean",
    "gap_coverage_fraction_median",
)

SUMMARY_FLOAT_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction_mean",
    "proposal_article_rank_fraction_mean",
    "proposal_article_rank_fraction_median",
    "proposal_book_rank_fraction_std",
    "gap_coverage_fraction_mean",
    "gap_coverage_fraction_median",
)

SUMMARY_PERCENTAGE_COLUMNS = SUMMARY_FLOAT_COLUMNS

BEST_RUN_PERCENTAGE_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction",
    "proposal_article_rank_fraction",
    "gap_coverage_fraction",
)

TRIPLET_PERCENTAGE_COLUMNS: Iterable[str] = (
    "gap_coverage_fraction_mean",
)

BEST_RUN_REQUIRED_COLUMNS: Iterable[str] = (
    "topic_id",
    "gap_idx",
    "proposal_book_rank",
    "proposal_book_rank_fraction",
    "proposal_article_rank",
    "proposal_article_rank_fraction",
    "gap_coverage_fraction",
)

GAP_FILE_PATTERN = re.compile(r"(?P<prefix>.+)_gap(?P<idx>\d+)$")


def load_summary_frames(run_summary_file: Path) -> dict[int, pd.DataFrame]:
    if not run_summary_file.exists():
        raise FileNotFoundError(f"Run summary file {run_summary_file} not found")

    match = GAP_FILE_PATTERN.match(run_summary_file.stem)
    candidate_files: list[tuple[int, Path]] = []

    if match:
        prefix = match.group("prefix")
        pattern = f"{prefix}_gap*{run_summary_file.suffix}"
        for candidate in sorted(run_summary_file.parent.glob(pattern)):
            candidate_match = GAP_FILE_PATTERN.match(candidate.stem)
            if not candidate_match:
                continue
            gap_idx = int(candidate_match.group("idx"))
            candidate_files.append((gap_idx, candidate))
    else:
        candidate_files.append((0, run_summary_file))

    if not candidate_files:
        raise FileNotFoundError(
            f"No gap-indexed run summary files found for base {run_summary_file}"
        )

    summary_frames: dict[int, pd.DataFrame] = {}
    for gap_idx, file_path in candidate_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:  # pragma: no cover - defensive IO handling
            LOGGER.error("Unable to read %s: %s", file_path, exc)
            continue
        summary_frames[gap_idx] = df

    if not summary_frames:
        raise RuntimeError(
            f"Failed to load any summary frames from {run_summary_file.parent}"
        )

    return dict(sorted(summary_frames.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LaTeX tables summarizing BookGap runs for each "
            "(embedder_model, llm_num_articles, llm_num_books) triplet."
        )
    )
    parser.add_argument(
        "--run-summary-file",
        type=Path,
        default=Path("artifacts") / "run_gap_summary_gap0.csv",
        help=(
            "Path to one of the gap-specific run summary CSVs produced by output_run_summary.py. "
            "All matching files with the same base name will be processed."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the root outputs directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "latex_tables",
        help="Directory where per-triplet LaTeX tables will be written.",
    )
    parser.add_argument(
        "--max-gap-rows",
        type=int,
        default=None,
        help=(
            "Maximum number of rows to include in the best-run gap analysis table. "
            "Leave unset to include all rows."
        ),
    )
    parser.add_argument(
        "--max-title-length",
        type=int,
        default=0,
        help=(
            "Maximum character length for proposal titles in the LaTeX tables. "
            "Use 0 to omit the title column entirely."
        ),
    )
    parser.add_argument(
        "--topic-name-length",
        type=int,
        default=15,
        help=(
            "Maximum character length for topic display names in best-run tables. "
            "Use a negative value to leave names untrimmed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing LaTeX files if they already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "value"


LATEX_SIMPLE_ESCAPE_CHARS = "&%$#_{}"
LATEX_SPECIAL_ESCAPE_MAP = {
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
LATEX_SIMPLE_ESCAPE_PATTERN = re.compile(
    rf"(?<!\\)([{re.escape(LATEX_SIMPLE_ESCAPE_CHARS)}])"
)


def escape_latex(text: object | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text)
    value = LATEX_SIMPLE_ESCAPE_PATTERN.sub(r"\\\1", value)
    for char, replacement in LATEX_SPECIAL_ESCAPE_MAP.items():
        value = re.sub(rf"(?<!\\){re.escape(char)}", replacement, value)
    return value


def escape_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    text_columns = working.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        working[column] = working[column].map(escape_latex)
    return working


def escape_column_labels(df: pd.DataFrame) -> pd.DataFrame:
    escaped = df.copy()
    if isinstance(escaped.columns, pd.MultiIndex):
        escaped.columns = pd.MultiIndex.from_tuples(
            tuple(escape_latex(part) for part in column)
            for column in escaped.columns
        )
        if escaped.columns.names is not None:
            escaped.columns.names = [
                escape_latex(name) if name is not None else None
                for name in escaped.columns.names
            ]
    else:
        escaped.columns = [escape_latex(column) for column in escaped.columns]
        if escaped.columns.name is not None:
            escaped.columns.name = escape_latex(escaped.columns.name)
    return escaped


def scale_percentage_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
    *,
    factor: float = 100.0,
    decimals: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    scaled = df.copy()
    for column in columns:
        if column in scaled.columns:
            numeric = pd.to_numeric(scaled[column], errors="coerce") * factor
            if decimals is not None:
                numeric = numeric.round(decimals)
            scaled[column] = numeric
    return scaled


def round_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
    *,
    decimals: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rounded = df.copy()
    for column in columns:
        if column in rounded.columns:
            rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(
                decimals
            )
    return rounded


def build_triplet_directory(embedder_model: str, num_articles: int, num_books: int) -> str:
    parts = (
        sanitize_component(embedder_model),
        f"articles_{num_articles}",
        f"books_{num_books}",
    )
    return "_".join(parts)


def formatters_from_columns(columns: Iterable[Hashable]) -> dict[Hashable, Callable[[object], str]]:
    def float_formatter(value: object) -> str:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, numbers.Real):
            return f"{float(value):.2f}"
        return str(value)

    return {column: float_formatter for column in columns}


def format_integer_value(value: object) -> str:
    if pd.isna(value):  # type: ignore[arg-type]
        return ""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return str(value)
    if pd.isna(numeric):  # type: ignore[arg-type]
        return ""
    try:
        return f"{int(round(numeric))}"
    except (TypeError, ValueError):
        return str(value)


def format_percentage_value(value: object) -> str:
    if pd.isna(value):  # type: ignore[arg-type]
        return ""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return str(value)
    return f"{numeric:.2f}"


def truncate_text(text: str, max_length: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_length:
        return text
    ellipsis = "…"
    return text[: max_length - len(ellipsis)] + ellipsis


def chunked(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def configure_pyalex() -> None:
    if pyalex is None:  # pragma: no cover - import guard
        raise RuntimeError(
            "pyalex is required to resolve topic names. Install it or run the script "
            "in the environment where pyalex is available."
        ) from PYALEX_IMPORT_ERROR
    if not getattr(pyalex.config, "email", None):
        email = os.environ.get("PYALEX_EMAIL", DEFAULT_PYALEX_EMAIL)
        pyalex.config.email = email


def normalize_topic_id(topic_id: object | None) -> str | None:
    if topic_id is None:
        return None
    value = str(topic_id).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"nan", "none", ""}:
        return None
    return value


def extract_short_topic_id(openalex_id: str | None) -> str | None:
    if not openalex_id:
        return None
    return openalex_id.rsplit("/", 1)[-1]


def lookup_topic_display_name(topic_name_map: Mapping[str, str], topic_id: object) -> str:
    normalized = normalize_topic_id(topic_id)
    if not normalized:
        return ""
    return topic_name_map.get(normalized, "")


def get_topic_display_name_map(topic_ids: Iterable[object]) -> dict[str, str]:
    normalized: set[str] = set()
    for topic_id in topic_ids:
        normalized_id = normalize_topic_id(topic_id)
        if normalized_id:
            normalized.add(normalized_id)

    missing: list[str] = [topic_id for topic_id in normalized if topic_id not in TOPIC_DISPLAY_NAME_CACHE]

    if missing:
        configure_pyalex()
        for batch in chunked(missing, TOPIC_BATCH_SIZE):
            try:
                results: Iterable[Any] = (
                    pyalex.Topics()
                    .filter(id="|".join(batch))
                    .select(["id", "display_name"])
                    .get()
                )
            except Exception as exc:  # pragma: no cover - external dependency
                LOGGER.warning("Failed to fetch topic names for %s: %s", batch, exc)
                continue

            for record in results:
                if not isinstance(record, dict):
                    continue
                raw_id = record.get("id")  # type: ignore[arg-type]
                topic_id = extract_short_topic_id(raw_id)
                display_name = record.get("display_name")  # type: ignore[arg-type]
                if topic_id and display_name:
                    TOPIC_DISPLAY_NAME_CACHE[topic_id] = display_name

    return {topic_id: TOPIC_DISPLAY_NAME_CACHE.get(topic_id, "") for topic_id in normalized}


def _apply_group_hlines(latex: str, group_breaks: Sequence[object] | None) -> str:
    if not group_breaks:
        return latex

    lines = latex.splitlines()
    try:
        data_start = next(i for i, line in enumerate(lines) if line.strip() == "\\midrule") + 1
        data_end = next(i for i, line in enumerate(lines) if line.strip() == "\\bottomrule")
    except StopIteration:
        return latex

    body_lines = lines[data_start:data_end]
    if len(body_lines) != len(group_breaks):
        LOGGER.debug(
            "Skipping group hlines due to length mismatch: %s rows vs %s breaks",
            len(body_lines),
            len(group_breaks),
        )
        return latex

    normalized_breaks = ["" if value is None else str(value) for value in group_breaks]
    new_body_lines: list[str] = []
    for idx, line in enumerate(body_lines):
        new_body_lines.append(line)
        if idx < len(body_lines) - 1 and normalized_breaks[idx] != normalized_breaks[idx + 1]:
            new_body_lines.append("\\hline")

    lines[data_start:data_end] = new_body_lines
    return "\n".join(lines)


def write_latex(
    df: pd.DataFrame,
    output_file: Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    formatters: Mapping[Hashable, Callable[[object], str]] | None = None,
    overwrite: bool = False,
    group_breaks: Sequence[object] | None = None,
) -> None:
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    escaped_caption = escape_latex(caption) if caption is not None else None

    safe_df = escape_column_labels(escape_dataframe(df))

    effective_formatters = formatters
    if formatters:
        escaped_formatters: dict[Hashable, Callable[[object], str]] = {}
        for key, formatter in formatters.items():
            if isinstance(key, tuple):
                escaped_key = tuple(
                    escape_latex(part) if isinstance(part, str) else part
                    for part in key
                )
            elif isinstance(key, str):
                escaped_key = escape_latex(key)
            else:
                escaped_key = key
            escaped_formatters[escaped_key] = formatter
        effective_formatters = escaped_formatters

    latex = safe_df.to_latex(
        index=False,
        escape=False,
        na_rep="",
        caption=escaped_caption,
        label=label,
        formatters=effective_formatters,
        multicolumn=True,
        multicolumn_format="c",
    )
    latex = re.sub(r"\\\\([&%$#_{}~^])", r"\\\\\1", latex)
    latex = _apply_group_hlines(latex, group_breaks)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(latex)
    LOGGER.info("Wrote %s", output_file)


def write_triplet_gap_coverage_table(
    df: pd.DataFrame,
    output_file: Path,
    *,
    caption: str | None,
    label: str,
    coverage_columns: Sequence[str],
    overwrite: bool,
) -> None:
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    escaped_caption = escape_latex(caption) if caption is not None else None
    column_labels = [
        escape_latex(column)
        for column in coverage_columns
    ]

    if coverage_columns:
        tabular_spec = "cc|" + "c" * len(coverage_columns)
    else:
        tabular_spec = "cc"

    lines: list[str] = ["\\begin{table}", "\\centering"]
    lines.append(f"\\begin{{tabular}}{{{tabular_spec}}}")
    lines.append("\\toprule")

    header_line = "\\multicolumn{1}{c}{\\#} & \\multicolumn{1}{c}{\\#}"
    if coverage_columns:
        header_line += (
            f" & \\multicolumn{{{len(coverage_columns)}}}{{c}}{{Mean gap coverage}}"
        )
    lines.append(header_line + " " + "\\\\")

    second_header = "articles & books"
    for column_label in column_labels:
        second_header += f" & {column_label}"
    lines.append(second_header + " " + "\\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        values: list[str] = [
            format_integer_value(row.get("# articles")),
            format_integer_value(row.get("# books")),
        ]
        for column in coverage_columns:
            values.append(format_percentage_value(row.get(column)))
        lines.append(" & ".join(values) + " " + "\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if escaped_caption is not None:
        lines.append(f"\\caption{{{escaped_caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    output_file.write_text("\n".join(lines))
    LOGGER.info("Wrote %s", output_file)


def sort_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "denoiser",
        "drop_k",
        "gap_coverage_fraction_mean",
        "proposal_book_rank_fraction_mean",
        "run",
    ]
    available = [column for column in sort_columns if column in df.columns]
    if not available:
        return df
    return df.sort_values(available, na_position="last")


def select_best_summary_row(df: pd.DataFrame) -> pd.Series | None:
    if "gap_coverage_fraction_mean" not in df.columns:
        return None
    ranked = df.sort_values(
        ["gap_coverage_fraction_mean", "proposal_book_rank_fraction_mean"],
        ascending=[False, True],
        na_position="last",
    )
    if ranked.empty:
        return None
    best = ranked.iloc[0]
    if pd.isna(best.get("gap_coverage_fraction_mean")):
        return None
    return best


def select_baseline_summary_row(df: pd.DataFrame) -> pd.Series:
    required_columns = {"denoiser", "drop_k", "gap_coverage_fraction_mean"}
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required summary columns for baseline selection: "
            + ", ".join(missing)
        )

    working = df.copy()
    denoiser_series = working["denoiser"].astype(str).str.strip().str.lower()
    drop_k_numeric = pd.to_numeric(working["drop_k"], errors="coerce")
    baseline = working.loc[(denoiser_series == "none") & (drop_k_numeric == 0)]

    if baseline.empty:
        raise ValueError("No baseline row with denoiser=NONE and drop_k=0 was found")

    sort_columns = ["gap_coverage_fraction_mean", "proposal_book_rank_fraction_mean"]
    available_columns = [column for column in sort_columns if column in baseline.columns]
    ascending = [column != "gap_coverage_fraction_mean" for column in available_columns]

    ranked = baseline.sort_values(
        available_columns,
        ascending=ascending,
        na_position="last",
    )
    selected = ranked.iloc[0]
    if pd.isna(selected.get("gap_coverage_fraction_mean")):
        raise ValueError(
            "Baseline row has missing gap_coverage_fraction_mean value"
        )
    return selected


def load_best_run_details(
    run_relative_path: str,
    outputs_dir: Path,
) -> tuple[pd.DataFrame, Path] | tuple[None, None]:
    run_path = outputs_dir / Path(run_relative_path)
    csv_path = run_path / "gaps_analysis_results.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing gaps_analysis_results.csv for %s", run_relative_path)
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Unable to read %s: %s", csv_path, exc)
        return None, None
    return df, csv_path


def prepare_best_run_table(
    df: pd.DataFrame,
    *,
    max_rows: int,
    max_title_length: int,
    topic_name_length: int,
) -> pd.DataFrame:
    missing = [column for column in BEST_RUN_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns in gaps analysis data: " + ", ".join(missing)
        )

    working = df.copy()
    if "proposal_title" in working.columns:
        if max_title_length > 0:
            working["proposal_title"] = working["proposal_title"].map(
                lambda value: truncate_text(value, max_title_length)
            )
        else:
            working = working.drop(columns=["proposal_title"])

    if "topic_id" in working.columns:
        topic_ids = [
            normalize_topic_id(topic_id)
            for topic_id in working["topic_id"].dropna().unique().tolist()
        ]
        topic_ids = [topic_id for topic_id in topic_ids if topic_id]
        if topic_ids:
            topic_name_map = get_topic_display_name_map(topic_ids)
            working["topic_display_name"] = working["topic_id"].map(
                lambda tid: lookup_topic_display_name(topic_name_map, tid)
                or normalize_topic_id(tid)
                or ""
            )
            if topic_name_length >= 0:
                working["topic_display_name"] = working["topic_display_name"].map(
                    lambda value: truncate_text(str(value), topic_name_length)
                )
            else:
                working["topic_display_name"] = working["topic_display_name"].astype(str)
        working = working.drop(columns=["topic_id"], errors="ignore")

    if max_rows is not None and max_rows > 0:
        working = working.head(max_rows)

    integer_columns = [
        "gap_idx",
        "proposal_book_rank",
        "proposal_article_rank",
        "gap_size",
    ]
    for column in integer_columns:
        if column in working.columns:
            numeric = pd.to_numeric(working[column], errors="coerce")
            working[column] = numeric.round().astype("Int64")

    display_columns: list[str] = []
    if "topic_display_name" in working.columns:
        display_columns.append("topic_display_name")
    for column in BEST_RUN_REQUIRED_COLUMNS:
        if column == "topic_id":
            continue
        display_columns.append(column)
    optional_columns = [
        "gap_size",
        "gap_size_percentile",
    ]
    for column in optional_columns:
        if column in working.columns:
            display_columns.append(column)

    return working[display_columns]


def build_summary_caption(
    friendly_embedder: str,
    gap_idx: int,
    num_articles: int,
    num_books: int,
) -> str:
    return (
        f"Run summary for {friendly_embedder} (gap={gap_idx}, "
        f"articles={num_articles}, books={num_books})"
    )


def build_best_run_caption(
    run_id: str,
    gap_idx: int,
    denoiser: Any,
    drop_k: Any,
) -> str:
    return (
        f"Top gaps for run {run_id} (gap={gap_idx}, denoiser={denoiser}, "
        f"drop_k={drop_k})"
    )


def ordered_embedder_columns(columns: Sequence[str]) -> Sequence[str]:
    desired = [friendly_embedder_name(embedder) for embedder in EMBEDDER_DISPLAY_NAME_MAP]
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in desired:
        if candidate and candidate in columns and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for column in columns:
        if column not in {"# articles", "# books"} and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        summary_frames = load_summary_frames(args.run_summary_file)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to load run summary files: %s", exc)
        raise SystemExit(1) from exc

    for gap_idx, summary_df in summary_frames.items():
        if summary_df.empty:
            LOGGER.warning(
                "Run summary for gap %s is empty; skipping LaTeX generation",
                gap_idx,
            )
            continue

        missing_summary_columns = [
            column for column in SUMMARY_COLUMNS if column not in summary_df.columns
        ]
        if missing_summary_columns:
            LOGGER.warning(
                "Run summary for gap %s is missing expected columns: %s",
                gap_idx,
                ", ".join(missing_summary_columns),
            )

        grouped = summary_df.groupby(
            ["embedder_model", "llm_num_articles", "llm_num_books"], dropna=False
        )

        baseline_rows: list[dict[str, Any]] = []

        for (embedder_model, num_articles, num_books), group in grouped:
            if pd.isna(embedder_model) or pd.isna(num_articles) or pd.isna(num_books):
                LOGGER.warning(
                    "Skipping group with missing identifiers (gap %s): embedder_model=%s, articles=%s, books=%s",
                    gap_idx,
                    embedder_model,
                    num_articles,
                    num_books,
                )
                continue

            try:
                triplet_dir_name = build_triplet_directory(
                    str(embedder_model), int(num_articles), int(num_books)
                )
            except Exception as exc:
                LOGGER.warning(
                    "Unable to build triplet directory name for %s/%s/%s (gap %s): %s",
                    embedder_model,
                    num_articles,
                    num_books,
                    gap_idx,
                    exc,
                )
                continue

            gap_output_dir = args.output_root / triplet_dir_name / f"gap{gap_idx}"

            embedder_model_str = str(embedder_model)
            friendly_embedder = (
                friendly_embedder_name(embedder_model_str) or embedder_model_str
            )

            summary_subset = group.loc[
                :, [column for column in SUMMARY_COLUMNS if column in group.columns]
            ]

            hidden_columns = {
                "embedder_model",
                "llm_num_articles",
                "llm_num_books",
                "row_count",
            }
            summary_subset = summary_subset.drop(
                columns=hidden_columns & set(summary_subset.columns),
                errors="ignore",
            )

            summary_sorted = sort_summary_frame(summary_subset)
            summary_hide_columns = {
                "run",
                "proposal_article_rank_fraction_median",
                "proposal_book_rank_fraction_std",
                "gap_coverage_fraction_median",
            }
            columns_to_drop = [
                column for column in summary_hide_columns if column in summary_sorted.columns
            ]
            summary_for_output = summary_sorted.drop(columns=columns_to_drop, errors="ignore")
            summary_scaled = scale_percentage_columns(
                summary_for_output,
                SUMMARY_PERCENTAGE_COLUMNS,
                decimals=2,
            )
            summary_display = summary_scaled.rename(columns=friendly_column_name)
            summary_percentage_columns = [
                friendly_column_name(column)
                for column in SUMMARY_PERCENTAGE_COLUMNS
                if column in summary_scaled.columns
            ]
            group_breaks = (
                summary_sorted["denoiser"].tolist()
                if "denoiser" in summary_sorted.columns
                else None
            )

            try:
                baseline_row = select_baseline_summary_row(group)
            except (KeyError, ValueError) as exc:
                raise RuntimeError(
                    "Unable to build baseline summary for "
                    f"gap={gap_idx}, embedder_model={friendly_embedder}, articles={num_articles}, "
                    f"books={num_books}: {exc}"
                ) from exc

            baseline_rows.append(
                {
                    "embedder_model": friendly_embedder,
                    "llm_num_articles": int(num_articles),
                    "llm_num_books": int(num_books),
                    "gap_coverage_fraction_mean": baseline_row.get(
                        "gap_coverage_fraction_mean"
                    ),
                }
            )

            caption = build_summary_caption(
                friendly_embedder,
                gap_idx,
                int(num_articles),
                int(num_books),
            )
            label_embedder_component = sanitize_component(friendly_embedder).lower()
            label = (
                f"tab:summary_gap{gap_idx}_{label_embedder_component}_"
                f"a{int(num_articles)}_b{int(num_books)}"
            )

            write_latex(
                summary_display,
                gap_output_dir / "run_gap_summary.tex",
                caption=caption,
                label=label,
                formatters=formatters_from_columns(summary_percentage_columns),
                overwrite=args.overwrite,
                group_breaks=group_breaks,
            )

            best_row = select_best_summary_row(group)
            if best_row is None:
                LOGGER.warning(
                    "No best run identified for gap %s, %s (articles=%s, books=%s)",
                    gap_idx,
                    friendly_embedder,
                    num_articles,
                    num_books,
                )
                continue

            best_run_id = str(best_row.get("run"))
            best_run_df, csv_path = load_best_run_details(best_run_id, args.outputs_dir)
            if best_run_df is None or csv_path is None:
                continue

            try:
                best_table = prepare_best_run_table(
                    best_run_df,
                    max_rows=args.max_gap_rows,
                    max_title_length=args.max_title_length,
                    topic_name_length=args.topic_name_length,
                )
            except KeyError as exc:
                LOGGER.warning("%s; skipping best run table for %s", exc, best_run_id)
                continue

            best_table = best_table.drop(
                columns=["proposal_book_rank", "proposal_article_rank"],
                errors="ignore",
            )

            caption = build_best_run_caption(
                best_run_id,
                gap_idx,
                best_row.get("denoiser"),
                best_row.get("drop_k"),
            )
            label = (
                f"tab:best_run_gap{gap_idx}_{sanitize_component(best_run_id).lower()}"
            )

            best_table_scaled = scale_percentage_columns(
                best_table,
                BEST_RUN_PERCENTAGE_COLUMNS,
                decimals=2,
            )
            for column in (
                "proposal_book_rank_fraction",
                "proposal_article_rank_fraction",
            ):
                if column in best_table_scaled.columns:
                    best_table_scaled[column] = 100.0 - pd.to_numeric(
                        best_table_scaled[column], errors="coerce"
                    )
                    best_table_scaled[column] = best_table_scaled[column].where(
                        (best_table_scaled[column] >= 0)
                        & (best_table_scaled[column] <= 100),
                        float("nan"),
                    )
            best_table_scaled = round_numeric_columns(
                best_table_scaled,
                BEST_RUN_PERCENTAGE_COLUMNS,
                decimals=2,
            )
            formatter_columns = {
                "proposal_book_rank_fraction",
                "proposal_article_rank_fraction",
                "gap_coverage_fraction",
                "gap_size_percentile",
            } & set(best_table_scaled.columns)
            best_table_display = best_table_scaled.rename(columns=friendly_column_name)
            formatter_columns_display = [
                friendly_column_name(column)
                for column in formatter_columns
            ]
            formatters = formatters_from_columns(formatter_columns_display)

            write_latex(
                best_table_display,
                gap_output_dir / "best_run_gaps_analysis.tex",
                caption=caption,
                label=label,
                formatters=formatters,
                overwrite=args.overwrite,
            )

        if baseline_rows:
            best_summary_df = pd.DataFrame(baseline_rows)
            best_summary_df = scale_percentage_columns(
                best_summary_df,
                TRIPLET_PERCENTAGE_COLUMNS,
                decimals=2,
            )

            for column in ["llm_num_articles", "llm_num_books"]:
                if column in best_summary_df.columns:
                    best_summary_df[column] = (
                        pd.to_numeric(best_summary_df[column], errors="coerce")
                        .round()
                        .astype("Int64")
                    )

            best_summary_df = best_summary_df.rename(
                columns={
                    "llm_num_articles": "# articles",
                    "llm_num_books": "# books",
                }
            )

            pivot_df = (
                best_summary_df.pivot_table(
                    index=["# articles", "# books"],
                    columns="embedder_model",
                    values="gap_coverage_fraction_mean",
                    aggfunc="first",
                )
                .reset_index()
                .sort_values(["# articles", "# books"])
                .reset_index(drop=True)
            )

            pivot_df.columns.name = None

            embedder_columns = [
                column
                for column in pivot_df.columns
                if column not in {"# articles", "# books"}
            ]

            rename_map = {
                column: friendly_embedder_name(column)
                for column in embedder_columns
            }
            pivot_df = pivot_df.rename(columns=rename_map)

            ordered_columns = ordered_embedder_columns(list(pivot_df.columns))
            column_order = ["# articles", "# books"] + [
                column for column in ordered_columns if column not in {"# articles", "# books"}
            ]
            pivot_df = pivot_df[column_order]

            for column in ["# articles", "# books"]:
                if column in pivot_df.columns:
                    pivot_df[column] = (
                        pd.to_numeric(pivot_df[column], errors="coerce")
                        .round()
                        .astype("Int64")
                    )

            coverage_columns = [
                column
                for column in pivot_df.columns
                if column not in {"# articles", "# books"}
            ]

            friendly_embedder_list = [column for column in coverage_columns if column]
            unique_embedder_names = list(dict.fromkeys(friendly_embedder_list))
            if not unique_embedder_names:
                friendly_embedder_text = "the available embedder models"
            elif len(unique_embedder_names) == 1:
                friendly_embedder_text = unique_embedder_names[0]
            elif len(unique_embedder_names) == 2:
                friendly_embedder_text = " and ".join(unique_embedder_names)
            else:
                friendly_embedder_text = (
                    ", ".join(unique_embedder_names[:-1])
                    + f", and {unique_embedder_names[-1]}"
                )

            gap_root_dir = args.output_root / f"gap{gap_idx}"
            write_triplet_gap_coverage_table(
                pivot_df,
                gap_root_dir / "triplet_gap_coverage.tex",
                caption=(
                    "Gap coverage fraction averaged over all topics for various settings of the number of example articles and books given to the LLM, comparing "
                    f"{friendly_embedder_text}."
                ),
                label=f"tab:triplet_gap_coverage_gap{gap_idx}",
                coverage_columns=coverage_columns,
                overwrite=args.overwrite,
            )
        else:
            LOGGER.warning(
                "No baseline rows collected for gap %s; skipping aggregate best triplet table",
                gap_idx,
            )


if __name__ == "__main__":
    main()
