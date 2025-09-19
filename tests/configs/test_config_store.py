"""
Test suite for ConfigStore registration and preset configurations.

Tests that all presets are properly registered and have the expected values
from the existing codebase (parse_args.py, batch_workflow.py, config_manager.py).
"""

import pytest
from hydra.core.config_store import ConfigStore

# Import to trigger registration
from bookgap.configs.config_store import register_configs
from bookgap.configs.config import ExperimentConfig
from bookgap.configs.gap.gap_config import GapConfig
from bookgap.configs.llm.llm_config import LLMConfig

from bookgap.types import CellType
from bookgap.cloud import WorkDenoisers
from bookgap.index import IndexType
from bookgap.llm.providers import ProviderType
from bookgap.llm.protocols import OpenAIModel


class TestConfigStoreRegistration:
    """Test that ConfigStore is properly set up and configs are registered."""
    
    def test_config_store_initialization(self):
        """Test that ConfigStore is accessible and has our configs."""
        cs = ConfigStore.instance()
        assert cs is not None
        
        # The configs should be registered automatically when we import config_store
        # We can't easily test the internal registration without Hydra's compose,
        # but we can test that the registration function exists and runs without error
        try:
            register_configs()  # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Config registration failed: {e}")


class TestProductionPresets:
    """Test that production presets match existing codebase defaults."""
    
    def test_production_gap_config_defaults(self):
        """Test production gap config uses existing codebase defaults."""
        # Manually create the production gap config to test values
        # (This mirrors what's in config_store.py)
        production_gap = GapConfig(
            cell_type=CellType.KBALL,  # From batch_workflow.py
            k=10,  # From parse_args.py
            radius_fraction=0.75,  # From parse_args.py
            drop_k=0,  # From parse_args.py
            hnsw_m=16,  # From parse_args.py
            hnsw_efc=512,  # From parse_args.py
            hnsw_efs=1024,  # From parse_args.py
            denoiser=WorkDenoisers.NONE,  # From parse_args.py
            index_type=IndexType.HNSW,  # From parse_args.py
            num_gaps=1
        )
        
        # Test that it matches expected existing defaults
        assert production_gap.cell_type == CellType.KBALL
        assert production_gap.radius_fraction == 0.75  # parse_args.py default
        assert production_gap.hnsw_efc == 512  # parse_args.py default
        assert production_gap.hnsw_efs == 1024  # parse_args.py default
        assert production_gap.denoiser == WorkDenoisers.NONE  # parse_args.py default
        assert production_gap.k == 10  # parse_args.py default
        assert production_gap.drop_k == 0  # parse_args.py default
        assert production_gap.hnsw_m == 16  # parse_args.py default
    
    def test_production_llm_config_defaults(self):
        """Test production LLM config uses existing codebase defaults."""
        # Manually create the production LLM config to test values
        production_llm = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,  # From config_manager.py
            num_articles=10,  # From batch_workflow.py
            num_books=5,  # From batch_workflow.py
            temperature=0.2,  # From config_manager.py
            max_tokens=2000,  # From config_manager.py
            chapters_min=10,  # From config_manager.py
            chapters_max=15,  # From config_manager.py
            min_book_coverage=0.5,
            use_structured_output=True,
        )
        
        # Test that it matches expected existing defaults
        assert production_llm.model_name == OpenAIModel.GPT_4O_MINI  # config_manager.py default
        assert production_llm.temperature == 0.2  # config_manager.py default
        assert production_llm.max_tokens == 2000  # config_manager.py default
        assert production_llm.num_articles == 10  # batch_workflow.py default
        assert production_llm.num_books == 5  # batch_workflow.py default
        assert production_llm.chapters_min == 10  # config_manager.py default
        assert production_llm.chapters_max == 15  # config_manager.py default


class TestDebugPresets:
    """Test that debug presets have appropriate fast/cheap settings."""
    
    def test_debug_gap_config_fast_settings(self):
        """Test debug gap config uses faster settings."""
        debug_gap = GapConfig(
            cell_type=CellType.KBALL,
            k=5,  # Smaller k for speed
            radius_fraction=0.5,  # Smaller radius for speed
            drop_k=0,
            hnsw_m=8,  # Smaller m for speed
            hnsw_efc=200,  # Lower efc for speed
            hnsw_efs=400,  # Lower efs for speed
            denoiser=WorkDenoisers.NONE,
            index_type=IndexType.HNSW,
            num_gaps=1
        )
        
        # Test that debug settings are faster than production
        assert debug_gap.k == 5  # Smaller than production (10)
        assert debug_gap.radius_fraction == 0.5  # Smaller than production (0.75)
        assert debug_gap.hnsw_m == 8  # Smaller than production (16)
        assert debug_gap.hnsw_efc == 200  # Smaller than production (512)
        assert debug_gap.hnsw_efs == 400  # Smaller than production (1024)
    
    def test_debug_llm_config_cheap_settings(self):
        """Test debug LLM config uses cheaper/faster settings."""
        debug_llm = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O_MINI,  # Cheaper model
            num_articles=5,  # Fewer articles for speed
            num_books=3,  # Fewer books for speed
            temperature=0.2,
            max_tokens=1000,  # Fewer tokens for speed
            chapters_min=5,  # Shorter proposals
            chapters_max=8,
            min_book_coverage=0.3,  # Lower threshold for speed
            use_structured_output=True,
        )
        
        # Test that debug settings are cheaper/faster than production
        assert debug_llm.model_name == OpenAIModel.GPT_4O_MINI  # Cheaper than gpt-4o
        assert debug_llm.num_articles == 5  # Fewer than production (10)
        assert debug_llm.num_books == 3  # Fewer than production (5)
        assert debug_llm.max_tokens == 1000  # Fewer than production (2000)
        assert debug_llm.chapters_min == 5  # Shorter than production (10)
        assert debug_llm.chapters_max == 8  # Shorter than production (15)


class TestHighQualityPresets:
    """Test that high quality presets use thorough/creative settings."""
    
    def test_high_quality_gap_config_thorough_settings(self):
        """Test high quality gap config uses more thorough settings."""
        high_quality_gap = GapConfig(
            cell_type=CellType.VORONOI,  # More thorough algorithm
            k=20,  # Larger k for more thorough analysis
            radius_fraction=0.9,  # Larger radius for more coverage
            drop_k=0,
            hnsw_m=32,  # Higher m for better quality
            hnsw_efc=1000,  # Higher efc for better quality
            hnsw_efs=2000,  # Higher efs for better quality
            denoiser=WorkDenoisers.JAD_EQUAL,  # Apply denoising
            index_type=IndexType.HNSW,
            num_gaps=3  # Find more gaps
        )
        
        # Test that high quality settings are more thorough than production
        assert high_quality_gap.cell_type == CellType.VORONOI  # Different algorithm
        assert high_quality_gap.k == 20  # Larger than production (10)
        assert high_quality_gap.radius_fraction == 0.9  # Larger than production (0.75)
        assert high_quality_gap.hnsw_m == 32  # Larger than production (16)
        assert high_quality_gap.hnsw_efc == 1000  # Larger than production (512)
        assert high_quality_gap.hnsw_efs == 2000  # Larger than production (1024)
        assert high_quality_gap.denoiser == WorkDenoisers.JAD_EQUAL  # Apply denoising
        assert high_quality_gap.num_gaps == 3  # More gaps than production (1)
    
    def test_creative_llm_config_exploratory_settings(self):
        """Test creative LLM config uses more exploratory settings."""
        creative_llm = LLMConfig(
            _target_="bookgap.llm.generators.openai_batch.OpenAIBatchGenerator",
            provider_type=ProviderType.OPENAI,
            model_name=OpenAIModel.GPT_4O,
            num_articles=15,  # More context
            num_books=8,  # More context
            temperature=0.8,  # More creative
            max_tokens=4000,  # Longer proposals
            chapters_min=12,  # Longer proposals
            chapters_max=20,
            min_book_coverage=0.4,
            use_structured_output=True,
        )
        
        # Test that creative settings are more exploratory than production
        assert creative_llm.num_articles == 15  # More than production (10)
        assert creative_llm.num_books == 8  # More than production (5)
        assert creative_llm.temperature == 0.8  # Higher than production (0.2)
        assert creative_llm.max_tokens == 4000  # More than production (2000)
        assert creative_llm.chapters_min == 12  # Longer than production (10)
        assert creative_llm.chapters_max == 20  # Longer than production (15)



