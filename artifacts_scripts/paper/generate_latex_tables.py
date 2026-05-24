from __future__ import annotations

import argparse
import logging
import numbers
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence, Literal

import pandas as pd

from artifacts_scripts.common import (
    EMBEDDER_DISPLAY_NAME_MAP,
    EMBEDDER_PATH_TO_NAME_MAP,
    COLUMN_DISPLAY_NAME_MAP_PLAIN,
    add_per_topic_gap_idx,
    filter_to_gap_idx,
    friendly_column_name,
    friendly_embedder_name,
    friendly_denoiser_name,
)

import pyalex

LOGGER = logging.getLogger(__name__)

# Set to True to use longtable environment, False to use table* environment
USE_LONGTABLE = False

# Set to True to write only the inner tabular/longtable environment.
# When True, non-longtable outputs omit the outer table/table* wrapper (and caption/label).
TABULAR_ONLY = False

# When USE_LONGTABLE is False and TABULAR_ONLY is False, break the table into
# multiple table* floats of at most this many rows each.  Set to None to emit a
# single table* regardless of length.
CHUNK_ROWS: int | None = 35

DEFAULT_PYALEX_EMAIL = "gchurchill@cs.stonybrook.edu"
TOPIC_BATCH_SIZE = 25
TOPIC_DISPLAY_NAME_CACHE: dict[str, str] = {}

EXCLUDED_DENOISERS: set[str] = {"W", "WC_CCLIP_1"}#{"AUG_LPP", "CCLIP_01", "CCLIP_1", "CCA", "CCP", "JAD_CCP", "JAD_HALF_L", "ORTHO_MANIFOLD"}

# Whitelist for (num_articles, num_books) combinations to generate tables for
# Set to None to generate all combinations
TRIPLET_WHITELIST: set[tuple[int, int]] | None = {(10, 5)}

DENOISER_DISPLAY_NAME_MAP: dict[str, str] = {
    "LIP": r"LIP",
}


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
    "proposal_gap_size_percentile_mean",
)

SUMMARY_FLOAT_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction_mean",
    "proposal_article_rank_fraction_mean",
    "proposal_article_rank_fraction_median",
    "proposal_book_rank_fraction_std",
    "gap_coverage_fraction_mean",
    "gap_coverage_fraction_median",
    "proposal_gap_size_percentile_mean",
)

SUMMARY_PERCENTAGE_COLUMNS = SUMMARY_FLOAT_COLUMNS

BEST_RUN_PERCENTAGE_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction",
    "proposal_article_rank_fraction",
    "gap_coverage_fraction",
    "proposal_gap_size_percentile",
    "target_gap_size_percentile_mean",
)

SUMMARY_RANK_FRACTION_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction_mean",
    "proposal_article_rank_fraction_mean",
    "proposal_article_rank_fraction_median",
)

BEST_RUN_RANK_FRACTION_COLUMNS: Iterable[str] = (
    "proposal_book_rank_fraction",
    "proposal_article_rank_fraction",
)

TRIPLET_PERCENTILE_COLUMNS: Iterable[str] = (
    "proposal_gap_size_percentile_mean",
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
    "proposal_gap_size_percentile",
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
        default=40,
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


def escape_latex_preserve_math(text: object | None) -> str:
    """Escape LaTeX special characters while preserving math mode ($...$).
    
    This function escapes text like escape_latex(), but protects content within
    $...$ delimiters from escaping, allowing for inline math expressions.
    
    Args:
        text: Text to escape, which may contain math mode expressions.
        
    Returns:
        Text with LaTeX special characters escaped except those in $...$ blocks.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    value = str(text)
    
    # Extract math mode blocks and replace them temporarily with unique markers
    # that won't be affected by LaTeX escaping (using non-alphanumeric safe markers)
    math_blocks: list[str] = []
    math_pattern = re.compile(r"\$([^$]*)\$")
    
    def replace_math(match: re.Match[str]) -> str:
        math_blocks.append(match.group(0))
        # Use hex encoding to create a marker that won't be escaped
        return f"\x00MATHBLOCK{len(math_blocks) - 1}\x00"
    
    value = math_pattern.sub(replace_math, value)
    
    # Escape LaTeX special characters in non-math text
    value = LATEX_SIMPLE_ESCAPE_PATTERN.sub(r"\\\1", value)
    for char, replacement in LATEX_SPECIAL_ESCAPE_MAP.items():
        value = re.sub(rf"(?<!\\){re.escape(char)}", replacement, value)
    
    # Restore math blocks
    for i, math_block in enumerate(math_blocks):
        value = value.replace(f"\x00MATHBLOCK{i}\x00", math_block)
    
    return value


def escape_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    text_columns = working.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        working[column] = working[column].map(escape_latex)
    return working


def escape_column_label(text: object) -> str:
    """Escape a single column label, preserving known LaTeX commands like \\makecell."""
    if text is None:
        return ""
    text_str = str(text)
    if text_str.startswith("\\makecell"):
        return text_str
    return escape_latex(text_str)


def escape_column_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Escape column labels, but preserve LaTeX commands like \\makecell."""

    escaped = df.copy()
    if isinstance(escaped.columns, pd.MultiIndex):
        escaped.columns = pd.MultiIndex.from_tuples(
            tuple(escape_column_label(part) for part in column)
            for column in escaped.columns
        )
        if escaped.columns.names is not None:
            escaped.columns.names = [
                escape_column_label(name) if name is not None else None
                for name in escaped.columns.names
            ]
    else:
        escaped.columns = [escape_column_label(column) for column in escaped.columns]
        if escaped.columns.name is not None:
            escaped.columns.name = escape_column_label(escaped.columns.name)
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


def invert_fraction_columns(
    df: pd.DataFrame,
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    inverted = df.copy()
    for column in columns:
        if column in inverted.columns:
            numeric = pd.to_numeric(inverted[column], errors="coerce")
            inverted_values = 1.0 - numeric
            valid_mask = numeric.isna() | ((numeric >= 0.0) & (numeric <= 1.0))
            inverted[column] = inverted_values.where(valid_mask, float("nan"))
    return inverted


def build_triplet_directory(embedder_model: str, num_articles: int, num_books: int) -> str:
    # Use the ENUM embedder name (e.g. GIST_SMALL, GEMMA_300M from
    # EMBEDDER_PATH_TO_NAME_MAP) so the output directory matches the path
    # the paper's \input{...} statements reference
    # (GIST_SMALL_articles_10_books_5). Using the raw model path produced
    # dirs like avsolatorio_GIST_small_Embedding_v0_articles_10_books_5
    # which don't match the paper.
    enum_name = EMBEDDER_PATH_TO_NAME_MAP.get(embedder_model, embedder_model)
    parts = (
        sanitize_component(enum_name),
        f"articles_{num_articles}",
        f"books_{num_books}",
    )
    return "_".join(parts)


def formatters_from_columns(
    columns: Iterable[str], *, decimals: int = 2
) -> dict[str, Callable[[object], str]]:
    def float_formatter(value: object) -> str:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, numbers.Real):
            return f"{float(value):.{decimals}f}"
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


def display_denoiser_name(value: object) -> str:
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return text
    mapped = DENOISER_DISPLAY_NAME_MAP.get(text.upper())
    if mapped is not None:
        return mapped
    return text


def _bolden_value(value: object) -> object:
    if value is None:
        return value
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return value
    except TypeError:
        pass
    text = str(value)
    if not text:
        return value
    if text.startswith("\\textbf{") and text.endswith("}"):
        return text
    return f"\\textbf{{{text}}}"


def format_monospace_value(value: object) -> str:
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except TypeError:
        pass
    text = str(value) if value is not None else ""
    if not text:
        return ""
    return f"\\texttt{{{text}}}"


def bolden_rows(df: pd.DataFrame, indices: Iterable[Hashable]) -> pd.DataFrame:
    bold_indices = [index for index in indices if index in df.index]
    if not bold_indices:
        return df
    working = df.copy()
    for column in working.columns:
        working[column] = working[column].astype(object)
    positions = [pos for pos in working.index.get_indexer(bold_indices) if pos >= 0]
    if not positions:
        return working
    for pos in positions:
        working.iloc[pos, :] = working.iloc[pos, :].map(_bolden_value)
    return working


def bolden_cells(
    df: pd.DataFrame,
    indices: Iterable[Hashable],
    columns: Iterable[Hashable],
) -> pd.DataFrame:
    """Bold specific cells at the intersection of given row indices and columns."""
    bold_indices = [index for index in indices if index in df.index]
    if not bold_indices:
        return df
    
    bold_columns = [column for column in columns if column in df.columns]
    if not bold_columns:
        return df
    
    working = df.copy()
    for column in bold_columns:
        working[column] = working[column].astype(object)
        # Bold only the cells in this column at the specified indices
        for index in bold_indices:
            working.at[index, column] = _bolden_value(working.at[index, column])  # type: ignore[index, assignment]
    return working


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


def _fix_longtable_caption_structure(latex: str) -> str:
    """Fix longtable caption placement.
    
    Pandas places caption+label before \\endfirsthead and caption[] after \\endfirsthead,
    but the correct structure should have caption+label as the first line inside \\endfirsthead
    (after \\begin{longtable}) and caption[] should be in \\endhead.
    
    Correct structure:
        \\begin{longtable}{...}
        \\caption{Full text} \\label{...} \\\\
        \\toprule
        Headers \\\\
        \\midrule
        \\endfirsthead
        \\caption[]{Continuation text} \\\\
        \\toprule
        Headers \\\\
        \\midrule
        \\endhead
    """
    lines = latex.splitlines()
    
    # Check if this is a longtable
    longtable_start = None
    for i, line in enumerate(lines):
        if "\\begin{longtable" in line:
            longtable_start = i
            break
    
    if longtable_start is None:
        return latex
    
    # Find the first caption with label (should be right after \\begin{longtable})
    first_caption_idx = None
    first_caption_lines = []
    
    for i in range(longtable_start + 1, len(lines)):
        if lines[i].strip().startswith("\\caption{"):
            # Check if this has a label (first caption)
            if "\\label{" in lines[i]:
                first_caption_idx = i
                first_caption_lines.append(lines[i])
                break
            elif i + 1 < len(lines) and "\\label{" in lines[i + 1]:
                first_caption_idx = i
                first_caption_lines.append(lines[i])
                first_caption_lines.append(lines[i + 1])
                break
        # Stop searching at endfirsthead
        if lines[i].strip() == "\\endfirsthead":
            break
    
    if not first_caption_lines:
        return latex  # No caption with label found
    
    # Ensure first_caption_idx is not None
    if first_caption_idx is None:
        return latex
    
    # Caption is already in the right place if it's immediately after \\begin{longtable}
    if first_caption_idx == longtable_start + 1:
        return latex
    
    # Move the caption+label to be first line after \\begin{longtable}
    result_lines = []
    skip_until = first_caption_idx + len(first_caption_lines)
    
    for i, line in enumerate(lines):
        if i == longtable_start:
            result_lines.append(line)
            # Insert caption lines right after \\begin{longtable}
            result_lines.extend(first_caption_lines)
        elif first_caption_idx <= i < skip_until:
            # Skip the original caption lines
            continue
        else:
            result_lines.append(line)
    
    return "\n".join(result_lines)


def _apply_group_hlines(latex: str, group_breaks: Sequence[object] | None) -> str:
    if not group_breaks:
        return latex

    lines = latex.splitlines()
    is_longtable = any("\\begin{longtable" in line for line in lines)

    data_start: int | None = None
    data_end: int | None = None

    if is_longtable:
        try:
            last_endlastfoot = max(
                i for i, line in enumerate(lines) if line.strip() == "\\endlastfoot"
            )
            data_start = last_endlastfoot + 1
        except ValueError:
            endhead_positions = [
                i for i, line in enumerate(lines) if line.strip() == "\\endhead"
            ]
            if not endhead_positions:
                return latex
            data_start = max(endhead_positions) + 1
            while (
                data_start < len(lines)
                and lines[data_start].strip() in {"", "\\midrule", "\\toprule"}
            ) or (
                data_start < len(lines)
                and lines[data_start].strip().startswith("\\multicolumn")
            ):
                data_start += 1

        try:
            data_end = next(
                i for i in range(len(lines) - 1, -1, -1)
                if lines[i].strip().startswith("\\end{longtable}")
            )
        except StopIteration:
            data_end = len(lines)
    else:
        try:
            data_start = next(
                i for i, line in enumerate(lines) if line.strip() == "\\midrule"
            ) + 1
            data_end = next(
                i for i, line in enumerate(lines) if line.strip() == "\\bottomrule"
            )
        except StopIteration:
            return latex

    if data_start is None:
        return latex
    if data_end is None:
        data_end = len(lines)

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
    formatters: Mapping[str, Callable[[object], str]] | None = None,
    overwrite: bool = False,
    group_breaks: Sequence[object] | None = None,
    bold_rows: Sequence[Hashable] | None = None,
    bold_columns: Sequence[Hashable] | None = None,
    bold_cell_map: Mapping[str, Sequence[Hashable]] | None = None,
    table_environment: Literal["table", "longtable"] = "table",
    tabular_only: bool = TABULAR_ONLY,
    comment: str | None = None,
    chunk_rows: int | None = CHUNK_ROWS,
) -> None:
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    if table_environment not in {"table", "longtable"}:
        raise ValueError(f"Unsupported table environment: {table_environment}")

    # Override table_environment based on USE_LONGTABLE constant
    if table_environment == "longtable" and not USE_LONGTABLE:
        table_environment = "table"

    escaped_caption = escape_latex(caption) if caption is not None else None
    # Unescape math expressions like $k$ or $\mathrm{...}$ (escape_latex adds backslashes).
    if escaped_caption:
        # Match $...$ with any content including backslashes, braces, etc.
        escaped_caption = re.sub(r"\\\$([^$]+?)\\\$", r"$\1$", escaped_caption)
        # Unescape curly braces within math mode
        escaped_caption = re.sub(r"\$([^$]*)\\\{([^$]*)\\\}([^$]*)\$", r"$\1{\2}\3$", escaped_caption)

    safe_df = escape_column_labels(escape_dataframe(df))

    if bold_rows:
        if bold_columns:
            # Bold only specific columns in the specified rows
            safe_df = bolden_cells(safe_df, bold_rows, bold_columns)
        else:
            # Bold entire rows
            safe_df = bolden_rows(safe_df, bold_rows)

    if bold_cell_map:
        escaped_map: dict[Hashable, Sequence[Hashable]] = {}
        for key, indices in bold_cell_map.items():
            if isinstance(key, tuple):
                escaped_key: Hashable = tuple(
                    escape_column_label(part) if isinstance(part, str) else part
                    for part in key
                )
            elif isinstance(key, str):
                escaped_key = escape_column_label(key)
            else:
                escaped_key = key
            escaped_map[escaped_key] = indices

        for column, indices in escaped_map.items():
            if not indices:
                continue
            safe_df = bolden_cells(safe_df, indices, [column])

    effective_formatters = formatters
    if formatters:
        escaped_formatters: dict[Hashable, Callable[[object], str]] = {}
        for key, formatter in formatters.items():
            if isinstance(key, tuple):
                escaped_key = tuple(
                    escape_column_label(part) if isinstance(part, str) else part
                    for part in key
                )
            elif isinstance(key, str):
                escaped_key = escape_column_label(key)
            else:
                escaped_key = key
            escaped_formatters[escaped_key] = formatter
        effective_formatters = escaped_formatters

    if table_environment == "longtable":
        latex = safe_df.to_latex(
            index=False,
            escape=False,
            na_rep="",
            caption=escaped_caption,
            label=label,
            formatters=effective_formatters,
            multicolumn=True,
            multicolumn_format="c",
            longtable=True,
        )
        latex = _fix_longtable_caption_structure(latex)
        latex = _apply_group_hlines(latex, group_breaks)
        latex = re.sub(r"\\\\([&%$#_{}~^])", r"\\\\\\1", latex)
        if comment:
            latex = f"% {comment}\n" + latex
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
        LOGGER.info("Wrote %s", output_file)
        return

    # Default: wrap in standard table environment
    if chunk_rows is not None and not tabular_only and len(safe_df) > chunk_rows:
        # Split into multiple table* floats so each fits on a page.
        table_env = "table*" if not USE_LONGTABLE else "table"
        table_parts: list[str] = []

        for chunk_start in range(0, len(safe_df), chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, len(safe_df))
            chunk_df = safe_df.iloc[chunk_start:chunk_end]

            chunk_latex = chunk_df.to_latex(
                index=False,
                escape=False,
                na_rep="",
                caption=None,
                label=None,
                formatters=effective_formatters,
                multicolumn=True,
                multicolumn_format="c",
            )
            chunk_latex = re.sub(r"\\\\([&%$#_{}~^])", r"\\\\\\1", chunk_latex)

            if group_breaks:
                chunk_breaks = list(group_breaks)[chunk_start:chunk_end]
                chunk_latex = _apply_group_hlines(chunk_latex, chunk_breaks)

            chunk_lines = chunk_latex.splitlines()

            result_lines = [f"\\begin{{{table_env}}}[t]"]
            if chunk_start == 0:
                if escaped_caption:
                    result_lines.append(f"\\caption{{{escaped_caption}}}")
                if label:
                    result_lines.append(f"\\label{{{label}}}")
            else:
                result_lines.append("\\ContinuedFloat")
                if escaped_caption:
                    result_lines.append(f"\\caption[]{{{escaped_caption} (continued)}}")
            result_lines.append("\\centering")
            result_lines.extend(chunk_lines)
            result_lines.append(f"\\end{{{table_env}}}")
            table_parts.append("\n".join(result_lines))

        latex = "\n\n".join(table_parts)
    else:
        latex = safe_df.to_latex(
            index=False,
            escape=False,
            na_rep="",
            caption=None,
            label=None,
            formatters=effective_formatters,
            multicolumn=True,
            multicolumn_format="c",
        )
        latex = re.sub(r"\\\\([&%$#_{}~^])", r"\\\\\\1", latex)
        latex = _apply_group_hlines(latex, group_breaks)

        lines = latex.splitlines()

        if not tabular_only:
            # Use table* instead of table when USE_LONGTABLE is False
            table_env = "table*" if not USE_LONGTABLE else "table"
            result_lines = [f"\\begin{{{table_env}}}", "\\centering"]
            result_lines.extend(lines)

            if escaped_caption:
                result_lines.append(f"\\caption{{{escaped_caption}}}")
            if label:
                result_lines.append(f"\\label{{{label}}}")

            result_lines.append(f"\\end{{{table_env}}}")
            latex = "\n".join(result_lines)
        else:
            # Only the inner tabular environment.
            latex = "\n".join(lines)

    if comment:
        latex = f"% {comment}\n" + latex
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(latex)
    LOGGER.info("Wrote %s", output_file)


# def write_triplet_gap_coverage_table(
#     df: pd.DataFrame,
#     output_file: Path,
#     *,
#     # caption: str | None,
#     # label: str,
#     coverage_columns: Sequence[str],
#     overwrite: bool,
# ) -> None:
#     if output_file.exists() and not overwrite:
#         LOGGER.info("Skipping existing file %s", output_file)
#         return
#
#     output_file.parent.mkdir(parents=True, exist_ok=True)
#
#     # escaped_caption = escape_latex(caption) if caption is not None else None
#     column_labels = [
#         escape_latex(column)
#         for column in coverage_columns
#     ]
#
#     if coverage_columns:
#         tabular_spec = "cc|" + "c" * len(coverage_columns)
#     else:
#         tabular_spec = "cc"
#
#     # lines: list[str] = ["\\begin{table}", "\\centering"]
#     lines: list[str] = []
#     lines.append(f"\\begin{{tabular}}{{{tabular_spec}}}")
#     lines.append("\\toprule")
#
#     header_line = "\\makecell{\\#\\\\articles} & \\makecell{\\#\\\\books}"
#     if coverage_columns:
#         header_line += (
#             f" & \\multicolumn{{{len(coverage_columns)}}}{{c}}{{\\makecell{{Mean gap\\\\coverage}}}}"
#         )
#     lines.append(header_line + " " + "\\\\")
#
#     if coverage_columns:
#         second_header = " & "  # Empty cells under # articles and # books
#         for column_label in column_labels:
#             second_header += f" & {column_label}"
#         lines.append(second_header + " " + "\\\\")
#     lines.append("\\midrule")
#
#     for _, row in df.iterrows():
#         values: list[str] = [
#             format_integer_value(row.get("# articles")),
#             format_integer_value(row.get("# books")),
#         ]
#         for column in coverage_columns:
#             values.append(format_percentage_value(row.get(column)))
#         lines.append(" & ".join(values) + " " + "\\\\")
#
#     lines.append("\\bottomrule")
#     lines.append("\\end{tabular}")
#     # if escaped_caption is not None:
#     #     lines.append(f"\\caption{{{escaped_caption}}}")
#     # lines.append(f"\\label{{{label}}}")
#     # lines.append("\\end{table}")
#
#     output_file.write_text("\n".join(lines))
#     LOGGER.info("Wrote %s", output_file)


def write_ablated_triplet_table(
    df: pd.DataFrame,
    output_file: Path,
    *,
    column_groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    overwrite: bool,
) -> None:
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Calculate tabular spec
    # cc | c... | c...
    spec_parts = ["cc|"]
    
    for i, (_, cols) in enumerate(column_groups):
        if cols:
            # Add divider after each group except the last
            if i < len(column_groups) - 1:
                spec_parts.append("c" * len(cols) + "|")
            else:
                spec_parts.append("c" * len(cols))
            
    tabular_spec = "".join(spec_parts)

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{tabular_spec}}}")
    lines.append("\\toprule")

    # Header Row 1: References used label spanning first 2 cols, then Group Headers (metrics)
    header_row_1 = [r"\multicolumn{2}{c|}{\makecell{References\\used}}"]
    
    for i, (label, cols) in enumerate(column_groups):
        if cols:
            # Add divider after each group except the last
            col_spec = "c|" if i < len(column_groups) - 1 else "c"
            # Don't wrap label in \makecell if it already contains LaTeX commands
            if label.startswith("\\makecell"):
                header_row_1.append(f"\\multicolumn{{{len(cols)}}}{{{col_spec}}}{{{label}}}")
            else:
                header_row_1.append(f"\\multicolumn{{{len(cols)}}}{{{col_spec}}}{{\\makecell{{{label}}}}}")
            
    lines.append(" & ".join(header_row_1) + r" \\")
    
    # Header Row 2: art, bks, Embedder names
    header_row_2 = ["art", "bks"]
    for _, cols in column_groups:
        for _, display_label in cols:
            header_row_2.append(escape_latex(display_label))
            
    lines.append(" & ".join(header_row_2) + r" \\")
    lines.append("\\midrule")

    # Apply whitelist and ordering for (# articles, # books)
    allowed_pairs: list[tuple[int, int]] = [
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 5),
        (5, 0),
        (5, 5),
        (10, 5),
    ]
    allowed_set = set(allowed_pairs)
    order_map = {pair: idx for idx, pair in enumerate(allowed_pairs)}

    working = df.copy()
    # Ensure numeric for filtering and ordering
    a_all = pd.to_numeric(working["# articles"], errors="coerce")
    b_all = pd.to_numeric(working["# books"], errors="coerce")
    pairs_all = list(zip(a_all.fillna(-1).astype(int), b_all.fillna(-1).astype(int)))
    mask = [pair in allowed_set for pair in pairs_all]
    working = working.loc[mask].copy()

    # Determine display order according to whitelist sequence
    a_work = pd.to_numeric(working["# articles"], errors="coerce")
    b_work = pd.to_numeric(working["# books"], errors="coerce")
    pairs_work = list(zip(a_work.fillna(-1).astype(int), b_work.fillna(-1).astype(int)))
    working["__order"] = [order_map[pair] for pair in pairs_work]
    working = working.sort_values("__order").drop(columns="__order")

    # Data Rows
    for _, row in working.iterrows():
        values: list[str] = [
            format_integer_value(row.get("# articles")),
            format_integer_value(row.get("# books")),
        ]
        
        for _, cols in column_groups:
            for col_key, _ in cols:
                values.append(format_percentage_value(row.get(col_key)))
                
        lines.append(" & ".join(values) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    output_file.write_text("\n".join(lines))
    LOGGER.info("Wrote %s", output_file)


def sort_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    sort_columns = [
        "__denoiser_priority",
        "denoiser",
        "drop_k",
        "gap_coverage_fraction_mean",
        "proposal_book_rank_fraction_mean",
        "proposal_article_rank_fraction_mean",
        "run",
    ]

    if "denoiser" in working.columns:
        priority_order = ["NONE"]
        priority_map = {name: index for index, name in enumerate(priority_order)}
        default_priority = len(priority_order)
        working["__denoiser_priority"] = (
            working["denoiser"]
            .astype(str)
            .map(lambda value: priority_map.get(value, default_priority))
        )

    available = [column for column in sort_columns if column in working.columns]
    if not available:
        return working

    descending_columns = {
        "gap_coverage_fraction_mean",
        "proposal_book_rank_fraction_mean",
        "proposal_article_rank_fraction_mean",
    }
    ascending = [column not in descending_columns for column in available]
    sorted_df = working.sort_values(available, ascending=ascending, na_position="last")
    return sorted_df.drop(columns="__denoiser_priority", errors="ignore")


def select_best_summary_row(df: pd.DataFrame) -> pd.Series | None:
    # Primary selection metric changed to proposal_gap_size_percentile_mean
    if "proposal_gap_size_percentile_mean" not in df.columns:
        return None
    sort_columns = [
        "proposal_gap_size_percentile_mean",  # higher percentile (closer to 1) preferred
        "gap_coverage_fraction_mean",         # tie-breaker 1
        "proposal_book_rank_fraction_mean",   # tie-breaker 2 (already inverted earlier)
        "proposal_article_rank_fraction_mean",# tie-breaker 3 (already inverted earlier)
    ]
    available_columns = [column for column in sort_columns if column in df.columns]
    if not available_columns:
        return None
    ranked = df.sort_values(
        available_columns,
        ascending=[False] * len(available_columns),
        na_position="last",
    )
    if ranked.empty:
        return None
    best = ranked.iloc[0]
    if pd.isna(best.get("gap_coverage_fraction_mean")):
        return None
    return best


def select_baseline_summary_row(df: pd.DataFrame, embedder_model: str) -> pd.Series:
    required_columns = {
        "denoiser",
        "drop_k",
        "proposal_gap_size_percentile_mean",
    }
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required summary columns for baseline selection: "
            + ", ".join(missing)
        )

    denoiser_target, drop_k_target = get_baseline_criteria(embedder_model)

    working = df.copy()
    denoiser_series = working["denoiser"].astype(str).str.strip()
    denoiser_target_stripped = denoiser_target.strip() if isinstance(denoiser_target, str) else denoiser_target
    mask = denoiser_series == denoiser_target_stripped
    if drop_k_target is not None:
        drop_k_numeric = pd.to_numeric(working["drop_k"], errors="coerce")
        mask &= drop_k_numeric == drop_k_target
    baseline = working.loc[mask]

    if baseline.empty:
        if drop_k_target is None:
            raise ValueError(
                f"No baseline row with denoiser={denoiser_target} was found"
            )
        raise ValueError(
            f"No baseline row with denoiser={denoiser_target} and drop_k={drop_k_target} was found"
        )

    sort_columns = [
        "proposal_gap_size_percentile_mean",
        "gap_coverage_fraction_mean",
        "proposal_book_rank_fraction_mean",
        "proposal_article_rank_fraction_mean",
    ]
    available_columns = [column for column in sort_columns if column in baseline.columns]
    if not available_columns:
        raise ValueError("Baseline group is missing sorting columns")
    ascending = [False] * len(available_columns)

    ranked = baseline.sort_values(
        available_columns,
        ascending=ascending,
        na_position="last",
    )
    selected = ranked.iloc[0]
    if pd.isna(selected.get("proposal_gap_size_percentile_mean")):
        raise ValueError(
            "Baseline row has missing proposal_gap_size_percentile_mean value"
        )
    return selected


def select_summary_row_by_criteria(df: pd.DataFrame, denoiser: str, drop_k: int) -> pd.Series:
    required_columns = {"denoiser", "drop_k", "gap_coverage_fraction_mean"}
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required summary columns for selection: "
            + ", ".join(missing)
        )

    working = df.copy()
    denoiser_series = working["denoiser"].astype(str).str.strip()
    drop_k_numeric = pd.to_numeric(working["drop_k"], errors="coerce")
    denoiser_target = denoiser.strip() if isinstance(denoiser, str) else denoiser
    selected = working.loc[(denoiser_series == denoiser_target) & (drop_k_numeric == drop_k)]

    if selected.empty:
        raise ValueError(f"No row with denoiser={denoiser} and drop_k={drop_k} was found")

    sort_columns = [
        "gap_coverage_fraction_mean",
        "proposal_book_rank_fraction_mean",
        "proposal_article_rank_fraction_mean",
    ]
    available_columns = [column for column in sort_columns if column in selected.columns]
    if not available_columns:
        raise ValueError("Selected group is missing sorting columns")
    ascending = [False] * len(available_columns)

    ranked = selected.sort_values(
        available_columns,
        ascending=ascending,
        na_position="last",
    )
    selected_row = ranked.iloc[0]
    if pd.isna(selected_row.get("gap_coverage_fraction_mean")):
        raise ValueError(
            "Selected row has missing gap_coverage_fraction_mean value"
        )
    return selected_row


def get_baseline_criteria(embedder_model: str) -> tuple[str, int | None]:
    # The (denoiser, drop_k) tuple that defines the "headline" config per
    # embedder — used to select the row that populates the §5.2 ablation
    # table. Updated 2026-05 from (CCLIP, 10) / (LIP, 384) to plain LPP
    # after the gemma4-quantitative-revision sweep showed plain LPP matches
    # or beats CCLIP on every metric and aligns the paper with the He &
    # Niyogi (2003) citation.
    #
    # Accept either the raw model path (used by callers that read directly
    # from the run summary CSV's `embedder_model` column) or the enum name
    # (legacy callers).
    enum_name = EMBEDDER_PATH_TO_NAME_MAP.get(embedder_model, embedder_model)
    return {"GIST_SMALL": ("LPP", 10),
            "GEMMA_300M": ("LPP", 394)}[enum_name]


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
    gap_idx: int,
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
    
    # Add per_topic_gap_idx if not already present
    if "per_topic_gap_idx" not in working.columns:
        working = add_per_topic_gap_idx(working)
    
    # Filter to only the specified gap index
    working = filter_to_gap_idx(working, gap_idx)
    
    if working.empty:
        LOGGER.warning(
            "No rows found after filtering to gap_idx=%s", gap_idx
        )
    
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

    working = invert_fraction_columns(working, BEST_RUN_RANK_FRACTION_COLUMNS)

    integer_columns = [
        "gap_idx",
        "proposal_book_rank",
        "proposal_article_rank",
        "num_articles",
        "num_books",
    ]
    for column in integer_columns:
        if column in working.columns:
            numeric = pd.to_numeric(working[column], errors="coerce")
            working[column] = numeric.round().astype("Int64")

    if (
        "target_gap_size_percentile_mean" not in working.columns
        and "gap_size_percentile" in working.columns
    ):
        working["target_gap_size_percentile_mean"] = working["gap_size_percentile"]

    desired_order = [
        "topic_display_name",
        "num_books",
        "num_articles",
        "gap_coverage_fraction",
        "proposal_gap_size_percentile",
        "target_gap_size_percentile_mean",
    ]

    for column in desired_order:
        if column not in working.columns:
            working[column] = pd.NA

    return working[desired_order].copy()


def build_summary_caption(
    friendly_embedder: str,
    gap_idx: int,
    num_articles: int,
    num_books: int,
) -> str:
    return (
        # f"Run summary for {friendly_embedder} (gap={gap_idx}, "
        f"Denoiser results for {friendly_embedder} ("
        f"articles={num_articles}, books={num_books})"
    )


def build_best_run_caption(
    friendly_embedder: str,
    denoiser: Any,
    drop_k: Any,
) -> str:
    # Use denoiser mapping with parameters (e.g., CCLIP(alpha=0.5))
    denoiser_display = friendly_denoiser_name(str(denoiser), latex=True)
    # Explicitly use plain $k$ without makecell wrapper in caption.
    return f"Best {friendly_embedder} run (Denoiser={denoiser_display}, $k$={drop_k})"


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


def generate_best_denoiser_summary_table(
    df: pd.DataFrame,
    output_file: Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
    tabular_only: bool = TABULAR_ONLY,
) -> None:
    """Generate a summary table with one row per denoiser showing best metrics.
    
    Columns: Denoiser, Best Gap Coverage %, k for coverage, Best Proposal Gap %ile, k for %ile
    """
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    if df.empty:
        LOGGER.warning("Input dataframe is empty; skipping best denoiser summary")
        return

    required_columns = {"denoiser", "drop_k", "gap_coverage_fraction_mean", "proposal_gap_size_percentile_mean"}
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        LOGGER.warning("Missing required columns for best denoiser summary: %s", missing)
        return

    # Group by denoiser and find best rows for each metric
    summary_rows = []
    
    for denoiser_name, group in df.groupby("denoiser", dropna=False):
        denoiser_str = friendly_denoiser_name(str(denoiser_name) if denoiser_name is not None else None, latex=True)
        
        # Find row with best gap coverage
        coverage_numeric = pd.to_numeric(group["gap_coverage_fraction_mean"], errors="coerce")
        if coverage_numeric.notna().any():
            best_coverage_idx = coverage_numeric.idxmax()
            best_coverage_row = group.loc[best_coverage_idx]
            best_coverage = best_coverage_row["gap_coverage_fraction_mean"]
            best_coverage_k = best_coverage_row["drop_k"]
        else:
            # All values are NaN, use first row
            best_coverage = float("nan")
            best_coverage_k = float("nan")
        
        # Find row with best proposal gap percentile
        percentile_numeric = pd.to_numeric(group["proposal_gap_size_percentile_mean"], errors="coerce")
        if percentile_numeric.notna().any():
            best_percentile_idx = percentile_numeric.idxmax()
            best_percentile_row = group.loc[best_percentile_idx]
            best_percentile = best_percentile_row["proposal_gap_size_percentile_mean"]
            best_percentile_k = best_percentile_row["drop_k"]
        else:
            # All values are NaN, use first row
            best_percentile = float("nan")
            best_percentile_k = float("nan")
        
        summary_rows.append({
            "Denoiser": denoiser_str,
            "Best Gap Coverage %": best_coverage,
            "k (coverage)": best_coverage_k,
            "Best Proposal Gap %ile": best_percentile,
            "k (%ile)": best_percentile_k,
        })
    
    if not summary_rows:
        LOGGER.warning("No summary rows generated; skipping output")
        return
    
    summary_df = pd.DataFrame(summary_rows)

    # Sort with NONE denoiser at the top
    priority_order = ["NONE"]
    priority_map = {name: index for index, name in enumerate(priority_order)}
    default_priority = len(priority_order)
    summary_df["__denoiser_priority"] = (
        summary_df["Denoiser"]
        .astype(str)
        .map(lambda value: priority_map.get(value, default_priority))
    )
    summary_df = summary_df.sort_values(
        ["__denoiser_priority", "Denoiser"],
        ascending=True
    ).drop(columns="__denoiser_priority")

    # Scale percentage columns
    percentage_columns = ["Best Gap Coverage %", "Best Proposal Gap %ile"]
    summary_df = scale_percentage_columns(
        summary_df,
        percentage_columns,
        decimals=2,
    )

    # Find the best (maximum) value in each Best* column to bold
    best_coverage_values = pd.to_numeric(summary_df["Best Gap Coverage %"], errors="coerce")
    best_percentile_values = pd.to_numeric(summary_df["Best Proposal Gap %ile"], errors="coerce")

    if best_coverage_values.notna().any():
        max_coverage = best_coverage_values.max()
        coverage_best_indices = set(summary_df.index[best_coverage_values == max_coverage].tolist())
    else:
        coverage_best_indices = set()

    if best_percentile_values.notna().any():
        max_percentile = best_percentile_values.max()
        percentile_best_indices = set(summary_df.index[best_percentile_values == max_percentile].tolist())
    else:
        percentile_best_indices = set()

    # Write custom LaTeX with vertical groupings
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    gap_coverage_label = COLUMN_DISPLAY_NAME_MAP_PLAIN["gap_coverage_fraction_mean"]
    proposal_gap_label = COLUMN_DISPLAY_NAME_MAP_PLAIN[
        "proposal_gap_size_percentile_mean"
    ]

    # Tabular spec: l | rr | rr (Denoiser | cvg% k | %ile k)
    lines_tex: list[str] = []
    if not tabular_only:
        lines_tex.extend([
            r"\begin{table*}",
            r"\centering",
        ])
    lines_tex.extend([
        r"\begin{tabular}{l|rr|rr}",
        r"\toprule",
        f" & \\multicolumn{{2}}{{c|}}{{{gap_coverage_label}}} & \\multicolumn{{2}}{{c}}{{{proposal_gap_label}}} \\\\",
        r"Denoiser & $k$ & \% & $k$ & \% \\",
        r"\midrule",
    ])

    for idx, row in summary_df.iterrows():
        denoiser_escaped = escape_latex_preserve_math(row["Denoiser"])
        
        coverage_val = format_percentage_value(row["Best Gap Coverage %"])
        if idx in coverage_best_indices and coverage_val:
            coverage_val = f"\\textbf{{{coverage_val}}}"
        
        coverage_k = format_integer_value(row["k (coverage)"])
        
        percentile_val = format_percentage_value(row["Best Proposal Gap %ile"])
        if idx in percentile_best_indices and percentile_val:
            percentile_val = f"\\textbf{{{percentile_val}}}"
        
        percentile_k = format_integer_value(row["k (%ile)"])
        
        lines_tex.append(f"{denoiser_escaped} & {coverage_k} & {coverage_val} & {percentile_k} & {percentile_val} " + r"\\")

    lines_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    if not tabular_only:
        if caption:
            escaped_caption = escape_latex(caption)
            lines_tex.append(f"\\caption{{{escaped_caption}}}")
        if label:
            lines_tex.append(f"\\label{{{label}}}")
        lines_tex.append(r"\end{table*}")

    output_file.write_text("\n".join(lines_tex))
    LOGGER.info("Wrote %s", output_file)


def generate_combined_best_denoiser_table(
    embedder_dfs: dict[str, pd.DataFrame],
    output_file: Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
    tabular_only: bool = TABULAR_ONLY,
    comment: str | None = None,
) -> None:
    """Generate a combined table pairing embedders with nested column group labels.

    Args:
        embedder_dfs: Dict mapping embedder names to their summary dataframes
        output_file: Path to write the LaTeX table
        caption: Optional table caption
        label: Optional LaTeX label
        overwrite: Whether to overwrite existing files
        tabular_only: Whether to output only the tabular environment
        comment: Optional provenance comment to prepend (each line is %-prefixed)
    """
    if output_file.exists() and not overwrite:
        LOGGER.info("Skipping existing file %s", output_file)
        return

    if not embedder_dfs:
        LOGGER.warning("No embedder dataframes provided; skipping combined table")
        return

    # Process each embedder's data
    embedder_summaries: dict[str, pd.DataFrame] = {}
    
    for embedder_name, df in embedder_dfs.items():
        if df.empty:
            continue
            
        required_columns = {"denoiser", "drop_k", "gap_coverage_fraction_mean", "proposal_gap_size_percentile_mean"}
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            LOGGER.warning("Missing required columns for %s: %s", embedder_name, missing)
            continue

        # Group by denoiser and find best rows for each metric
        summary_rows = []
        
        for denoiser_name, group in df.groupby("denoiser", dropna=False):
            denoiser_str = friendly_denoiser_name(str(denoiser_name) if denoiser_name is not None else None, latex=True)
            
            # Find row with best gap coverage
            coverage_numeric = pd.to_numeric(group["gap_coverage_fraction_mean"], errors="coerce")
            if coverage_numeric.notna().any():
                best_coverage_idx = coverage_numeric.idxmax()
                best_coverage_row = group.loc[best_coverage_idx]
                best_coverage = best_coverage_row["gap_coverage_fraction_mean"]
                best_coverage_k = best_coverage_row["drop_k"]
            else:
                best_coverage = float("nan")
                best_coverage_k = float("nan")
            
            # Find row with best proposal gap percentile
            percentile_numeric = pd.to_numeric(group["proposal_gap_size_percentile_mean"], errors="coerce")
            if percentile_numeric.notna().any():
                best_percentile_idx = percentile_numeric.idxmax()
                best_percentile_row = group.loc[best_percentile_idx]
                best_percentile = best_percentile_row["proposal_gap_size_percentile_mean"]
                best_percentile_k = best_percentile_row["drop_k"]
            else:
                best_percentile = float("nan")
                best_percentile_k = float("nan")
            
            summary_rows.append({
                "Denoiser": denoiser_str,
                f"{embedder_name}_coverage_%": best_coverage,
                f"{embedder_name}_coverage_k": best_coverage_k,
                f"{embedder_name}_percentile_%": best_percentile,
                f"{embedder_name}_percentile_k": best_percentile_k,
            })
        
        if summary_rows:
            embedder_summaries[embedder_name] = pd.DataFrame(summary_rows)

    if not embedder_summaries:
        LOGGER.warning("No valid summaries generated; skipping output")
        return

    # Merge all embedder summaries on Denoiser
    combined_df = None
    for embedder_name, summary_df in embedder_summaries.items():
        if combined_df is None:
            combined_df = summary_df
        else:
            combined_df = combined_df.merge(summary_df, on="Denoiser", how="outer")

    if combined_df is None or combined_df.empty:
        LOGGER.warning("Combined dataframe is empty; skipping output")
        return

    # Sort with priority for denoiser names
    priority_order = ["NONE"]
    priority_map = {name: index for index, name in enumerate(priority_order)}
    default_priority = len(priority_order)
    combined_df["__denoiser_priority"] = (
        combined_df["Denoiser"]
        .astype(str)
        .map(lambda value: priority_map.get(value, default_priority))
    )
    combined_df = combined_df.sort_values(
        ["__denoiser_priority", "Denoiser"],
        ascending=True
    ).drop(columns="__denoiser_priority")

    # Scale percentage columns
    percentage_columns = [col for col in combined_df.columns if col.endswith("_%")]
    for col in percentage_columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce") * 100.0
        combined_df[col] = combined_df[col].round(2)

    # Find best values for each metric across all embedders
    coverage_cols = [col for col in combined_df.columns if "_coverage_%" in col]
    percentile_cols = [col for col in combined_df.columns if "_percentile_%" in col]
    
    coverage_best_cells: set[tuple[int, str]] = set()
    percentile_best_cells: set[tuple[int, str]] = set()
    
    if coverage_cols:
        for col in coverage_cols:
            values = pd.to_numeric(combined_df[col], errors="coerce")
            if values.notna().any():
                max_val = values.max()
                for idx in combined_df.index[values == max_val]:
                    coverage_best_cells.add((idx, col))
    
    if percentile_cols:
        for col in percentile_cols:
            values = pd.to_numeric(combined_df[col], errors="coerce")
            if values.notna().any():
                max_val = values.max()
                for idx in combined_df.index[values == max_val]:
                    percentile_best_cells.add((idx, col))

    # Build LaTeX table with nested headers
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    gap_coverage_label = COLUMN_DISPLAY_NAME_MAP_PLAIN["gap_coverage_fraction_mean"]
    proposal_gap_label = COLUMN_DISPLAY_NAME_MAP_PLAIN["proposal_gap_size_percentile_mean"]
    
    embedder_names = list(embedder_summaries.keys())
    ordered_embedders = ordered_embedder_columns(embedder_names)
    
    # Tabular spec: l | (rr)* | (rr)* for each embedder in each metric group
    spec_parts = ["l"]
    for _ in range(len(ordered_embedders)):
        spec_parts.append("|rr")
    for _ in range(len(ordered_embedders)):
        spec_parts.append("|rr")
    tabular_spec = "".join(spec_parts)
    
    lines_tex: list[str] = []
    if not tabular_only:
        lines_tex.extend([
            r"\begin{table*}",
            r"\centering",
        ])
    
    lines_tex.extend([
        f"\\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
    ])
    
    # Header Row 1: Metric group labels
    header1_parts = [""]
    # Gap coverage group
    header1_parts.append(f"\\multicolumn{{{len(ordered_embedders) * 2}}}{{c|}}{{{gap_coverage_label}}}")
    # Proposal gap percentile group
    header1_parts.append(f"\\multicolumn{{{len(ordered_embedders) * 2}}}{{c}}{{{proposal_gap_label}}}")
    lines_tex.append(" & ".join(header1_parts) + r" \\")
    
    # Header Row 2: Embedder labels within each metric group
    header2_parts = [""]
    # Coverage embedders - always add divider to maintain separation from percentile group
    for i, embedder in enumerate(ordered_embedders):
        header2_parts.append(f"\\multicolumn{{2}}{{c|}}{{{escape_latex(embedder)}}}")
    # Percentile embedders
    for i, embedder in enumerate(ordered_embedders):
        divider = "|" if i < len(ordered_embedders) - 1 else ""
        header2_parts.append(f"\\multicolumn{{2}}{{c{divider}}}{{{escape_latex(embedder)}}}")
    lines_tex.append(" & ".join(header2_parts) + r" \\")
    
    # Header Row 3: k and % labels
    header3_parts = ["Denoiser"]
    for _ in range(len(ordered_embedders) * 2):  # Coverage and percentile
        header3_parts.extend([r"$k$", r"\%"])
    header3_line = " & ".join(header3_parts) + r" \\"
    lines_tex.append(header3_line)
    lines_tex.append(r"\midrule")
    
    # Data rows
    for idx, row in combined_df.iterrows():
        denoiser_escaped = escape_latex_preserve_math(row["Denoiser"])
        row_parts = [denoiser_escaped]
        
        # Coverage values
        for embedder in ordered_embedders:
            coverage_col = f"{embedder}_coverage_%"
            k_col = f"{embedder}_coverage_k"
            
            coverage_val = format_percentage_value(row.get(coverage_col))
            if (idx, coverage_col) in coverage_best_cells and coverage_val:
                coverage_val = f"\\textbf{{{coverage_val}}}"
            
            k_val = format_integer_value(row.get(k_col))
            row_parts.extend([k_val, coverage_val])
        
        # Percentile values
        for embedder in ordered_embedders:
            percentile_col = f"{embedder}_percentile_%"
            k_col = f"{embedder}_percentile_k"
            
            percentile_val = format_percentage_value(row.get(percentile_col))
            if (idx, percentile_col) in percentile_best_cells and percentile_val:
                percentile_val = f"\\textbf{{{percentile_val}}}"
            
            k_val = format_integer_value(row.get(k_col))
            row_parts.extend([k_val, percentile_val])
        
        lines_tex.append(" & ".join(row_parts) + r" \\")
    
    lines_tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])
    
    if not tabular_only:
        if caption:
            escaped_caption = escape_latex(caption)
            lines_tex.append(f"\\caption{{{escaped_caption}}}")
        if label:
            lines_tex.append(f"\\label{{{label}}}")
        lines_tex.append(r"\end{table*}")

    body = "\n".join(lines_tex)
    if comment:
        commented = "\n".join(f"% {line}" for line in comment.splitlines())
        body = commented + "\n" + body
    output_file.write_text(body + "\n")
    LOGGER.info("Wrote %s", output_file)



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

        if EXCLUDED_DENOISERS and "denoiser" in summary_df.columns:
            before_count = len(summary_df)
            summary_df = summary_df[
                ~summary_df["denoiser"].astype(str).isin(EXCLUDED_DENOISERS)
            ]
            filtered_count = before_count - len(summary_df)
            if filtered_count > 0:
                LOGGER.info(
                    "Filtered %s rows for excluded denoisers %s in gap %s",
                    filtered_count,
                    ", ".join(sorted(EXCLUDED_DENOISERS)),
                    gap_idx,
                )
            if summary_df.empty:
                LOGGER.warning(
                    "All rows excluded after filtering denoisers for gap %s; skipping",
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
        
        # Track embedder data for combined tables by (num_articles, num_books)
        triplet_embedder_data: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}

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
            
            # Check if this triplet passes the whitelist for individual table generation
            triplet_key = (int(num_articles), int(num_books))
            should_generate_individual_tables = (
                TRIPLET_WHITELIST is None or triplet_key in TRIPLET_WHITELIST
            )

            embedder_model_str = str(embedder_model)
            friendly_embedder = (
                friendly_embedder_name(embedder_model_str) or embedder_model_str
            )
            
            # Only create output directory if we're generating individual tables
            gap_output_dir: Path | None = None
            if should_generate_individual_tables:
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
                    should_generate_individual_tables = False
                else:
                    gap_output_dir = args.output_root / triplet_dir_name / f"gap{gap_idx}"

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

            summary_subset = invert_fraction_columns(
                summary_subset,
                SUMMARY_RANK_FRACTION_COLUMNS,
            )

            summary_sorted = sort_summary_frame(summary_subset)
            summary_hide_columns = {
                "run",
                "proposal_article_rank_fraction_median",
                "proposal_article_rank_fraction_mean",
                "proposal_book_rank_fraction_std",
                "proposal_book_rank_fraction_mean",
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
            denoiser_column_name = friendly_column_name("denoiser")
            if denoiser_column_name in summary_display.columns:
                summary_display[denoiser_column_name] = summary_display[
                    denoiser_column_name
                ].map(display_denoiser_name)
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

            highlight_source_columns = [
                "gap_coverage_fraction_mean",
                "proposal_book_rank_fraction_mean",
                "proposal_article_rank_fraction_mean",
                "proposal_gap_size_percentile_mean",
            ]
            highlight_indices_by_column: dict[str, list[Hashable]] = {}
            if "denoiser" in summary_sorted.columns:
                for _, denoiser_group in summary_sorted.groupby(
                    "denoiser", dropna=False
                ):
                    for column in highlight_source_columns:
                        if column not in denoiser_group.columns:
                            continue
                        numeric_values = pd.to_numeric(
                            denoiser_group[column], errors="coerce"
                        )
                        if not numeric_values.notna().any():
                            continue
                        max_value = numeric_values.max()
                        if pd.isna(max_value):
                            continue
                        winners = denoiser_group.loc[
                            numeric_values == max_value
                        ].index.tolist()
                        if not winners:
                            continue
                        target_list = highlight_indices_by_column.setdefault(column, [])
                        for winner in winners:
                            if winner not in target_list:
                                target_list.append(winner)

            bold_cell_map: dict[str, Sequence[Hashable]] = {}
            for column, indices in highlight_indices_by_column.items():
                if not indices:
                    continue
                friendly_name = friendly_column_name(column)
                if friendly_name not in summary_display.columns:
                    continue
                bold_cell_map[friendly_name] = list(indices)

            try:
                baseline_row = select_baseline_summary_row(summary_subset, embedder_model_str)
            except (KeyError, ValueError) as exc:
                LOGGER.warning(
                    "Unable to find baseline for gap=%s, embedder_model=%s, articles=%s, books=%s: %s. Using NaN.",
                    gap_idx, friendly_embedder, num_articles, num_books, exc
                )
                baseline_row = pd.Series({
                    "gap_coverage_fraction_mean": float("nan"),
                    "proposal_gap_size_percentile_mean": float("nan"),
                    "denoiser": None,
                    "drop_k": None,
                })

            baseline_rows.append(
                {
                    "embedder_model": friendly_embedder,
                    "llm_num_articles": int(num_articles),
                    "llm_num_books": int(num_books),
                    "gap_coverage_fraction_mean": baseline_row.get(
                        "gap_coverage_fraction_mean"
                    ),
                    "proposal_gap_size_percentile_mean": baseline_row.get(
                        "proposal_gap_size_percentile_mean"
                    ),
                    "denoiser": display_denoiser_name(baseline_row.get("denoiser")),
                    "drop_k": baseline_row.get("drop_k"),
                }
            )

            # Only generate individual tables if this triplet passes the whitelist
            if should_generate_individual_tables and gap_output_dir is not None:
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

                # Create formatters for all numeric columns in the display
                summary_formatters = formatters_from_columns(summary_percentage_columns)

                write_latex(
                    summary_display,
                    gap_output_dir / "run_gap_summary.tex",
                    caption=caption,
                    label=label,
                    formatters=summary_formatters,
                    overwrite=args.overwrite,
                    table_environment="longtable",
                    group_breaks=group_breaks,
                    bold_cell_map=bold_cell_map or None,
                    tabular_only=TABULAR_ONLY,
                )

                # Generate best denoiser summary table
                best_denoiser_caption = (
                    f"Best denoiser configurations for {friendly_embedder} "
                    f"(articles={int(num_articles)}, books={int(num_books)})"
                )
                best_denoiser_label = (
                    f"tab:best_denoisers_gap{gap_idx}_{label_embedder_component}_"
                    f"a{int(num_articles)}_b{int(num_books)}"
                )
                generate_best_denoiser_summary_table(
                    summary_sorted,
                    gap_output_dir / "best_denoisers_summary.tex",
                    caption=best_denoiser_caption,
                    label=best_denoiser_label,
                    overwrite=args.overwrite,
                    tabular_only=TABULAR_ONLY,
                )
            
            # Store data for combined table generation (regardless of whitelist)
            triplet_key = (int(num_articles), int(num_books))
            if triplet_key not in triplet_embedder_data:
                triplet_embedder_data[triplet_key] = {}
            triplet_embedder_data[triplet_key][friendly_embedder] = summary_sorted.copy()

            # Only process best run details if generating individual tables
            if not should_generate_individual_tables or gap_output_dir is None:
                continue
                
            best_row = select_best_summary_row(summary_subset)
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
                    gap_idx=gap_idx,
                    max_rows=args.max_gap_rows,
                    max_title_length=args.max_title_length,
                    topic_name_length=args.topic_name_length,
                )
            except KeyError as exc:
                LOGGER.warning("%s; skipping best run table for %s", exc, best_run_id)
                continue

            best_table = best_table.drop(
                columns=[
                    "proposal_book_rank",
                    "proposal_article_rank",
                    "gap_size",
                    "gap_size_percentile",
                ],
                errors="ignore",
            )

            caption = build_best_run_caption(
                friendly_embedder,
                best_row.get("denoiser"),
                best_row.get("drop_k"),
            )
            # Use stable label based on embedder/articles/books instead of run timestamp
            label = (
                f"tab:best_run_gap{gap_idx}_{sanitize_component(embedder_model_str).lower()}"
                f"_articles_{num_articles}_books_{num_books}"
            )

            best_table_scaled = scale_percentage_columns(
                best_table,
                BEST_RUN_PERCENTAGE_COLUMNS,
                decimals=1,
            )
            best_table_scaled = round_numeric_columns(
                best_table_scaled,
                BEST_RUN_PERCENTAGE_COLUMNS,
                decimals=1,
            )
            formatter_columns = {
                "proposal_book_rank_fraction",
                "proposal_article_rank_fraction",
                "gap_coverage_fraction",
                "proposal_gap_size_percentile",
                "target_gap_size_percentile_mean",
            } & set(best_table_scaled.columns)
            best_table_display = best_table_scaled.rename(columns=friendly_column_name)
            formatter_columns_display = [
                friendly_column_name(column)
                for column in formatter_columns
            ]
            formatters = formatters_from_columns(
                formatter_columns_display,
                decimals=1,
            )
            topic_column_label = friendly_column_name("topic_display_name")
            if (
                isinstance(topic_column_label, str)
                and topic_column_label in best_table_display.columns
            ):
                formatters = dict(formatters)
                formatters[topic_column_label] = format_monospace_value

            write_latex(
                best_table_display,
                gap_output_dir / "best_run_gaps_analysis.tex",
                caption=caption,
                label=label,
                formatters=formatters,
                overwrite=args.overwrite,
                table_environment="longtable",
                tabular_only=TABULAR_ONLY,
                comment=best_run_id,
            )

        # Generate combined best denoiser tables for each (num_articles, num_books) pair
        for triplet_key, embedder_data in triplet_embedder_data.items():
            num_articles, num_books = triplet_key
            
            # Check if this triplet passes the whitelist for combined table generation
            should_generate_combined_table = (
                TRIPLET_WHITELIST is None or triplet_key in TRIPLET_WHITELIST
            )
            if not should_generate_combined_table:
                continue
            
            # Only generate if we have data for multiple embedders
            if len(embedder_data) < 2:
                continue
            
            combined_caption = (
                f"Best denoiser configurations across embedders "
                f"(articles={num_articles}, books={num_books})"
            )
            combined_label = (
                f"tab:combined_denoisers_gap{gap_idx}_"
                f"a{num_articles}_b{num_books}"
            )
            
            combined_output_dir = args.output_root / f"gap{gap_idx}" / f"articles_{num_articles}_books_{num_books}"
            combined_output_dir.mkdir(parents=True, exist_ok=True)

            combined_comment = (
                f"Generated by artifacts_scripts/generate_latex_tables.py\n"
                f"Source: {args.run_summary_file}\n"
                f"Gap {gap_idx}, articles={num_articles}, books={num_books}, "
                f"embedders={', '.join(embedder_data.keys())}"
            )

            # main.tex wraps this \input in its own \begin{table*}...\end{table*},
            # so emit bare tabular regardless of the global TABULAR_ONLY (which
            # stays False so write_latex can chunk long per-topic tables).
            generate_combined_best_denoiser_table(
                embedder_data,
                combined_output_dir / "combined_best_denoisers.tex",
                caption=combined_caption,
                label=combined_label,
                overwrite=args.overwrite,
                tabular_only=True,
                comment=combined_comment,
            )

        if baseline_rows:
            best_summary_df = pd.DataFrame(baseline_rows)
            best_summary_df = scale_percentage_columns(
                best_summary_df,
                TRIPLET_PERCENTILE_COLUMNS,
                decimals=2,
            )

            for column in ["llm_num_articles", "llm_num_books"]:
                if column in best_summary_df.columns:
                    best_summary_df[column] = (
                        pd.to_numeric(best_summary_df[column], errors="coerce")
                        .round()
                        .astype("Int64")
                    )

            embedder_settings: dict[str, tuple[Any, Any]] = {}
            for row in baseline_rows:
                embedder = row["embedder_model"]
                denoiser = row["denoiser"]
                drop_k = row["drop_k"]
                embedder_settings[embedder] = (denoiser, drop_k)

            for column in ["# articles", "# books"]:
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

            # Pivot for both values
            pivot_df = best_summary_df.pivot_table(
                index=["# articles", "# books"],
                columns="embedder_model",
                values=["proposal_gap_size_percentile_mean", "gap_coverage_fraction_mean"],
                aggfunc="first"
            )
            # pivot_df is indexed by (# articles, # books). Columns are MultiIndex (metric, embedder).
            
            # Sort index
            pivot_df = pivot_df.sort_index()
            
            # Determine embedder order
            raw_embedders = pivot_df.columns.get_level_values(1).unique().tolist()
            embedder_map = {e: friendly_embedder_name(e) for e in raw_embedders}
            friendly_embedders = list(embedder_map.values())
            ordered_friendly = ordered_embedder_columns(friendly_embedders)
            friendly_to_raw = {v: k for k, v in embedder_map.items()}
            ordered_raw_embedders = [friendly_to_raw[f] for f in ordered_friendly if f in friendly_to_raw]
            
            # Flatten DF
            flat_df = pivot_df.index.to_frame().reset_index(drop=True)
            
            column_groups = []
            
            for metric_col, metric_label in [
                ("gap_coverage_fraction_mean", COLUMN_DISPLAY_NAME_MAP_PLAIN.get("gap_coverage_fraction_mean", "Target gap\\\\coverage percentage")),
                ("proposal_gap_size_percentile_mean", COLUMN_DISPLAY_NAME_MAP_PLAIN.get("proposal_gap_size_percentile_mean", "Proposal gap\\\\percentile")),
            ]:
                group_cols = []
                for embedder in ordered_raw_embedders:
                    if (metric_col, embedder) in pivot_df.columns:
                        col_name = f"{metric_col}_{embedder}"
                        flat_df[col_name] = pivot_df[(metric_col, embedder)].values
                        display_name = friendly_embedder_name(embedder)
                        group_cols.append((col_name, display_name))
                
                if group_cols:
                    column_groups.append((metric_label, group_cols))
            
            for col in ["# articles", "# books"]:
                flat_df[col] = pd.to_numeric(flat_df[col], errors="coerce").round().astype("Int64")

            gap_root_dir = args.output_root / f"gap{gap_idx}"
            write_ablated_triplet_table(
                flat_df,
                gap_root_dir / "ablated_triplet_comparison.tex",
                column_groups=column_groups,
                overwrite=args.overwrite,
            )

            # We only need to calculate embedder descriptions once for logging/info if needed,
            # but the original code used it for the caption which is now commented out.
            # So we can skip the description generation block or keep it if we want to use it later.
            # For now, I'll omit the description generation since it was only used for the caption.

        else:
            LOGGER.warning(
                "No baseline rows collected for gap %s; skipping aggregate best triplet table",
                gap_idx,
            )


if __name__ == "__main__":
    main()
