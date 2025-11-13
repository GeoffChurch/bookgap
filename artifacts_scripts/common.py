from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)

PROPOSAL_BOOK_RANK_FRACTION_COLUMN = "proposal_book_rank_fraction"
PROPOSAL_ARTICLE_RANK_FRACTION_COLUMN = "proposal_article_rank_fraction"
GAP_COVERAGE_FRACTION_COLUMN = "gap_coverage_fraction"
GAP_ZSCORE_COLUMN = "gap_zscore"

EMBEDDER_DISPLAY_NAME_MAP = {
    "GIST_SMALL": "GIST",
    "GEMMA_300M": "Gemma",
}


def friendly_embedder_name(embedder_model: Optional[str]) -> str:
    if embedder_model is None:
        return ""
    return EMBEDDER_DISPLAY_NAME_MAP.get(embedder_model, embedder_model)


COLUMN_DISPLAY_NAME_MAP = {
    "denoiser": "Denoiser",
    "drop_k": r"\makecell{$k$}",
    "proposal_book_rank_fraction_mean": r"\makecell{Book\\rank \%}",
    "proposal_article_rank_fraction_mean": r"\makecell{Article\\rank \%}",
    "proposal_article_rank_fraction_median": r"\makecell{Article rank \%\\(median)}",
    "proposal_book_rank_fraction_std": r"\makecell{Book rank \%\\(std)}",
    "gap_coverage_fraction_mean": r"\makecell{Gap\\coverage \%}",
    "gap_coverage_fraction_median": r"\makecell{Gap coverage \%\\(median)}",
    "proposal_gap_size_percentile_mean": r"\makecell{Proposal\\Gap \%-ile}",
    "row_count": "Rows",
    "proposal_book_rank": r"\makecell{Book\\rank}",
    "proposal_book_rank_fraction": r"\makecell{Book\\rank \%}",
    "proposal_article_rank": r"\makecell{Article\\rank}",
    "proposal_article_rank_fraction": r"\makecell{Article\\rank \%}",
    "gap_coverage_fraction": r"\makecell{Gap\\coverage \%}",
    "gap_idx": "Gap",
    "gap_size": r"\makecell{Gap\\size}",
    "gap_size_percentile": r"\makecell{Gap size\\percentile}",
    "topic_display_name": "Topic",
    "num_articles": r"\makecell{\# articles}",
    "num_books": r"\makecell{\# books}",
}


def friendly_column_name(column: str) -> str:
    return COLUMN_DISPLAY_NAME_MAP.get(column, column)

@dataclass
class RunMetadata:
    denoiser: Optional[str]
    drop_k: Optional[int]
    embedder_model: Optional[str]
    num_articles: Optional[int]
    num_books: Optional[int]


def find_run_directories(outputs_dir: Path) -> Iterable[Path]:
    """Yield candidate run directories containing a Hydra config."""
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory {outputs_dir} does not exist")

    config_paths = sorted(outputs_dir.rglob("config.yaml"))
    for config_path in config_paths:
        if config_path.parent.name != ".hydra":
            continue
        run_dir = config_path.parent.parent
        if run_dir.is_dir():
            yield run_dir


def load_run_metadata(run_dir: Path) -> Optional[RunMetadata]:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        LOGGER.warning("Missing config.yaml in %s", run_dir)
        return None

    config = OmegaConf.load(config_path)

    denoiser = OmegaConf.select(config, "cloud.denoiser", default=None)
    drop_k = OmegaConf.select(config, "cloud.drop_k", default=None)
    embedder_model = OmegaConf.select(config, "cloud.embedder_model", default=None)
    num_articles = OmegaConf.select(config, "llm.num_articles", default=None)
    num_books = OmegaConf.select(config, "llm.num_books", default=None)

    return RunMetadata(
        denoiser=str(denoiser) if denoiser is not None else None,
        drop_k=int(drop_k) if drop_k is not None else None,
        embedder_model=str(embedder_model) if embedder_model is not None else None,
        num_articles=int(num_articles) if num_articles is not None else None,
        num_books=int(num_books) if num_books is not None else None,
    )


def load_results_dataframe(run_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = run_dir / "gaps_analysis_results.csv"
    if not csv_path.exists():
        LOGGER.warning("Missing gaps_analysis_results.csv in %s", run_dir)
        return None

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - unexpected IO errors
        LOGGER.warning("Failed to read %s: %s", csv_path, exc)
        return None


def valid_rank_fraction_mask(df: pd.DataFrame) -> pd.Series:
    if PROPOSAL_BOOK_RANK_FRACTION_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{PROPOSAL_BOOK_RANK_FRACTION_COLUMN}' missing from dataframe"
        )
    column = df[PROPOSAL_BOOK_RANK_FRACTION_COLUMN]
    return column.notna() & (column != -1)


def relative_run_path(run_dir: Path, outputs_dir: Path) -> Path:
    try:
        return run_dir.relative_to(outputs_dir)
    except ValueError:
        return run_dir


def add_per_topic_gap_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'per_topic_gap_idx' column to the dataframe.
    
    This column represents the gap index within each topic, computed as the
    cumulative count of rows within each topic_id group. Gaps are assumed to
    be ordered by decreasing gap_size within each topic.
    
    If a 'gap_size' column is present, validates that gap sizes are non-increasing
    within each topic.
    
    Args:
        df: DataFrame containing at least a 'topic_id' column
        
    Returns:
        A copy of the input DataFrame with the 'per_topic_gap_idx' column added
        
    Raises:
        ValueError: If 'topic_id' column is missing from the dataframe or if
                   gap_size increases within any topic
    """
    if "topic_id" not in df.columns:
        raise ValueError("Column 'topic_id' missing from dataframe")
    
    # Validate gap_size is non-increasing within each topic
    if "gap_size" in df.columns:
        for topic_id, topic_df in df.groupby("topic_id", sort=False):
            gap_sizes = topic_df["gap_size"].tolist()
            previous_size = None
            for size in gap_sizes:
                if previous_size is not None and pd.notna(previous_size) and pd.notna(size):
                    if float(size) > float(previous_size):
                        raise ValueError(
                            f"Gap size increased within topic {topic_id}: "
                            f"{previous_size} -> {size}"
                        )
                previous_size = size
    
    result = df.copy()
    result["per_topic_gap_idx"] = result.groupby("topic_id", sort=False).cumcount()
    return result


def filter_to_gap_idx(df: pd.DataFrame, gap_idx: int) -> pd.DataFrame:
    """
    Filter dataframe to only include rows matching the specified gap index.
    
    Args:
        df: DataFrame containing a 'per_topic_gap_idx' column
        gap_idx: The gap index to filter to (0 = first/largest gap per topic)
        
    Returns:
        A filtered copy of the DataFrame containing only rows where
        per_topic_gap_idx equals gap_idx
        
    Raises:
        ValueError: If 'per_topic_gap_idx' column is missing from the dataframe
    """
    if "per_topic_gap_idx" not in df.columns:
        raise ValueError(
            "Column 'per_topic_gap_idx' missing from dataframe. "
            "Did you forget to call add_per_topic_gap_idx()?"
        )
    
    return df[df["per_topic_gap_idx"] == gap_idx].copy()
