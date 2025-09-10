import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from bookgap.config_manager import AnalysisConfiguration
from bookgap.results_manager import GapResults


def test_gapresults_csv_roundtrip_preserves_schema_and_rows():
    # Build analysis configuration
    from bookgap.config_manager import GapConfiguration, LLMConfiguration
    from bookgap.types import CellType
    from bookgap.llm.llm_implementation import Implementation
    
    gap_config = GapConfiguration(
        denoiser="JAD_EQUAL",
        drop_k=10,
        radius_fraction=0.40,
        cell_type=CellType.KBALL
    )
    
    llm_config = LLMConfiguration(
        implementation=Implementation.STRUCTURED,
        provider="openai",
        model="gpt-4",
        temperature=0.2,
        max_tokens=2000,
        chapters_min=10,
        chapters_max=15,
        requested_num_prompt_articles=30,
        requested_num_prompt_books=7
    )
    
    config = AnalysisConfiguration(
        gap_config=gap_config,
        llm_config=llm_config
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        csv_path = config_dir / "gap_analysis.csv"
        
        results = GapResults(config)
        results.csv_path = csv_path  # Override the default path

        # Add two realistic rows
        results.add_row(
            topic_name="AI Planning",
            topic_id="T1",
            gap_idx=0,
            gap_size=12,
            gap_size_percentile=0.70,
            gap_zscore=1.25,
            shapiro_log10_pval=-1.50,
            num_articles=123,
            num_books=45,
            actual_num_prompt_articles=28,
            actual_num_prompt_books=6,
        )
        results.add_row(
            topic_name="Machine Learning",
            topic_id="T2",
            gap_idx=1,
            gap_size=25,
            gap_size_percentile=0.90,
            gap_zscore=2.10,
            shapiro_log10_pval=-3.20,
            num_articles=256,
            num_books=60,
            actual_num_prompt_articles=30,
            actual_num_prompt_books=7,
        )

        df_written = results.get_dataframe()

        # Sanity: expected columns present
        expected_cols = {
            "denoiser",
            "drop_k",
            "radius_fraction",
            "cell_type",
            "gap_idx",
            "topic_id",
            "topic_name",
            "num_articles",
            "num_books",
            "gap_size",
            "gap_size_percentile",
            "gap_zscore",
            "shapiro_log10_pval",
            "requested_num_prompt_articles",
            "requested_num_prompt_books",
            "actual_num_prompt_articles",
            "actual_num_prompt_books",
        }
        assert expected_cols.issubset(set(df_written.columns))
        assert (df_written["cell_type"] == config.gap_config.cell_type.name.lower()).all()

        # CSV path and existence
        assert csv_path.exists()

        # Load back via API and compare frames (ignore dtypes)
        loaded = GapResults.load_from_csv(csv_path, config)
        df_loaded = loaded.get_dataframe()

        # Ensure same shape and rows (order-insensitive, dtype-insensitive)
        pdt.assert_frame_equal(
            df_written.sort_values(list(df_written.columns)).reset_index(drop=True),
            df_loaded.sort_values(list(df_loaded.columns)).reset_index(drop=True),
            check_dtype=False,
            check_like=True,
        )

        # Extra: verify requested/actual columns retained correctly
        assert (df_loaded["requested_num_prompt_articles"] == config.llm_config.requested_num_prompt_articles).all()
        assert (df_loaded["requested_num_prompt_books"] == config.llm_config.requested_num_prompt_books).all()
        assert set(df_loaded["actual_num_prompt_books"]) == {6, 7}
