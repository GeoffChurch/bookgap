"""
Test suite for ConfigStore registration and preset configurations.

Tests that all presets are properly registered and have the expected values
from the existing codebase (parse_args.py, batch_workflow.py, config_manager.py).
"""

import pytest
from hydra.core.config_store import ConfigStore

# Import to trigger registration
from bookgap.configs.config_store import register_configs


class TestConfigStoreRegistration:
    """Test that ConfigStore is properly set up and configs are registered."""
    
    def test_config_store_initialization(self):
        """Test that ConfigStore is accessible and has our configs."""
        cs = ConfigStore.instance()
        assert cs is not None
        
        try:
            register_configs()  # Should not raise an exception
        except Exception as e:
            pytest.fail(f"Config registration failed: {e}")

        # Check that some known groups/names are registered via public API
        gap_nodes = set(cs.list("gap"))
        llm_nodes = set(cs.list("llm"))

        assert any(name.startswith("production") for name in gap_nodes)
        assert any(name.startswith("production") for name in llm_nodes)
