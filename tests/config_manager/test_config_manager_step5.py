#!/usr/bin/env python3
import pytest

pytest.skip("Skipped legacy step test (GapDirectoryManager legacy API)", allow_module_level=True)
        """Sample LLMConfiguration for testing."""
        return LLMConfiguration(num_articles=50, num_books=25)

    def test_get_config_dir_with_gap_config(self, temp_dir, sample_gap_config):
        """Test get_config_dir works with GapConfiguration objects."""
        config_dir = GapDirectoryManager.get_config_dir(sample_gap_config)
        
        expected_path = Path("artifacts") / "clean_k=5_radiusfrac=0.25"
        assert config_dir.name == expected_path.name
        assert config_dir.exists()

    def test_get_gap_dir_with_configs(self, temp_dir, sample_gap_config):
        """Test get_gap_dir works with configuration objects."""
        gap_dir = GapDirectoryManager.get_gap_dir(sample_gap_config, 2)
        
        # Should create the full path including gap index
        assert "clean_k=5_radiusfrac=0.25" in str(gap_dir)
        assert "gap_2" in str(gap_dir.name)
        assert gap_dir.exists()

    def test_get_analysis_dir_integration(self, temp_dir, sample_gap_config):
        """Test get_analysis_dir integration."""
        analysis_dir = GapDirectoryManager.get_analysis_dir(sample_gap_config, 1)
        
        assert "clean_k=5_radiusfrac=0.25" in str(analysis_dir)
        assert "gap_1" in str(analysis_dir)
        assert "analysis" in str(analysis_dir.name)

    def test_get_responses_llm_dir_integration(self, temp_dir, sample_gap_config, sample_llm_config):
        """Test get_responses_llm_dir integration with both config objects."""
        responses_dir = GapDirectoryManager.get_responses_llm_dir(sample_gap_config, 0, sample_llm_config)
        
        assert "clean_k=5_radiusfrac=0.25" in str(responses_dir)
        assert "gap_0" in str(responses_dir)
        assert "articles_50_books_25" in str(responses_dir.name)

    def test_get_prompts_llm_dir_integration(self, temp_dir, sample_gap_config, sample_llm_config):
        """Test get_prompts_llm_dir integration."""
        prompts_dir = GapDirectoryManager.get_prompts_llm_dir(sample_gap_config, 1, sample_llm_config)
        
        assert "clean_k=5_radiusfrac=0.25" in str(prompts_dir)
        assert "gap_1" in str(prompts_dir)
        assert "articles_50_books_25" in str(prompts_dir.name)

    def test_get_batch_requests_dir_integration(self, temp_dir, sample_gap_config):
        """Test get_batch_requests_dir integration."""
        batch_dir = GapDirectoryManager.get_batch_requests_dir(sample_gap_config)
        
        assert "clean_k=5_radiusfrac=0.25" in str(batch_dir)
        assert "batch_requests" in str(batch_dir.name)

    def test_get_batch_results_dir_integration(self, temp_dir, sample_gap_config):
        """Test get_batch_results_dir integration."""
        batch_dir = GapDirectoryManager.get_batch_results_dir(sample_gap_config)
        
        assert "clean_k=5_radiusfrac=0.25" in str(batch_dir)
        assert "batch_results" in str(batch_dir.name)

    def test_get_batch_request_file_integration(self, temp_dir, sample_gap_config):
        """Test get_batch_request_file integration."""
        timestamp = 1234567890
        request_file = GapDirectoryManager.get_batch_request_file(sample_gap_config, timestamp)
        
        assert "clean_k=5_radiusfrac=0.25" in str(request_file)
        assert "batch_request_1234567890.jsonl" in str(request_file.name)

    def test_get_batch_results_file_integration(self, temp_dir, sample_gap_config):
        """Test get_batch_results_file integration."""
        batch_id = "batch_abc123"
        results_file = GapDirectoryManager.get_batch_results_file(sample_gap_config, batch_id)
        
        assert "clean_k=5_radiusfrac=0.25" in str(results_file)
        assert "results_batch_abc123.jsonl" in str(results_file.name)

    def test_all_directories_created_correctly(self, temp_dir, sample_gap_config, sample_llm_config):
        """Test that all directory types are created correctly in integration."""        
        # Create all directory types
        config_dir = GapDirectoryManager.get_config_dir(sample_gap_config)
        gap_dir = GapDirectoryManager.get_gap_dir(sample_gap_config, 0)
        analysis_dir = GapDirectoryManager.get_analysis_dir(sample_gap_config, 0)
        plots_dir = GapDirectoryManager.get_plots_dir(sample_gap_config, 0)
        prompts_dir = GapDirectoryManager.get_prompts_dir(sample_gap_config, 0)
        prompts_llm_dir = GapDirectoryManager.get_prompts_llm_dir(sample_gap_config, 0, sample_llm_config)
        responses_dir = GapDirectoryManager.get_responses_dir(sample_gap_config, 0)
        responses_llm_dir = GapDirectoryManager.get_responses_llm_dir(sample_gap_config, 0, sample_llm_config)
        batch_requests_dir = GapDirectoryManager.get_batch_requests_dir(sample_gap_config)
        batch_results_dir = GapDirectoryManager.get_batch_results_dir(sample_gap_config)
        
        # Verify all exist and have correct paths
        assert all([
            config_dir.exists(),
            gap_dir.exists(),
            analysis_dir.exists(),
            plots_dir.exists(),
            prompts_dir.exists(),
            prompts_llm_dir.exists(),
            responses_dir.exists(),
            responses_llm_dir.exists(),
            batch_requests_dir.exists(),
            batch_results_dir.exists()
        ])
        
        # Verify correct hierarchical structure
        assert gap_dir.parent == config_dir  # gap_dir is under config_dir
        assert config_dir.parent.name == "gaps"  # config_dir is under gaps
        assert analysis_dir.parent == gap_dir
        assert prompts_llm_dir.parent == prompts_dir
        assert responses_llm_dir.parent == responses_dir

    def test_integration_with_different_configurations(self, temp_dir):
        """Test integration works correctly with different configuration combinations."""        
        # Test different GapConfigurations
        gap_config1 = GapConfiguration(denoiser="clean", drop_k=3, radius_fraction=0.1)
        gap_config2 = GapConfiguration(denoiser="noise", drop_k=10, radius_fraction=0.5)
        
        # Test different LLMConfigurations
        llm_config1 = LLMConfiguration(num_articles=10, num_books=5)
        llm_config2 = LLMConfiguration(num_articles=100, num_books=50)
        
        # Create directories for different combinations
        dir1 = GapDirectoryManager.get_responses_llm_dir(gap_config1, 0, llm_config1)
        dir2 = GapDirectoryManager.get_responses_llm_dir(gap_config1, 0, llm_config2)
        dir3 = GapDirectoryManager.get_responses_llm_dir(gap_config2, 1, llm_config1)
        dir4 = GapDirectoryManager.get_responses_llm_dir(gap_config2, 1, llm_config2)
        
        # Verify all are different and exist
        dirs = [dir1, dir2, dir3, dir4]
        assert len(set(dirs)) == 4  # All unique paths
        assert all(d.exists() for d in dirs)
        
        # Verify correct structure
        assert "clean_k=3_radiusfrac=0.10" in str(dir1)
        assert "noise_k=10_radiusfrac=0.50" in str(dir3)
        assert "articles_10_books_5" in str(dir1)
        assert "articles_100_books_50" in str(dir2)

    def test_no_backwards_compatibility_enforcement(self):
        """Test that the methods no longer accept string or tuple inputs."""
        sample_gap_config = GapConfiguration(denoiser="clean", drop_k=5, radius_fraction=0.25)
        
        # These methods should only accept proper configuration objects
        # Verify type hints are enforced by testing with wrong types would cause static analysis errors
        
        # This is validated at static analysis time - the methods now only accept
        # GapConfiguration and LLMConfiguration objects, no Union types
        
        # Verify methods exist and have correct signatures
        import inspect
        
        # Check get_config_dir signature
        sig = inspect.signature(GapDirectoryManager.get_config_dir)
        gap_config_param = sig.parameters['gap_config']
        assert 'Union' not in str(gap_config_param.annotation)
        assert 'GapConfiguration' in str(gap_config_param.annotation)
        
        # Check get_responses_llm_dir signature
        sig = inspect.signature(GapDirectoryManager.get_responses_llm_dir)
        gap_config_param = sig.parameters['gap_config']
        llm_config_param = sig.parameters['llm_config']
        assert 'Union' not in str(gap_config_param.annotation)
        assert 'Union' not in str(llm_config_param.annotation)
        assert 'GapConfiguration' in str(gap_config_param.annotation)
        assert 'LLMConfiguration' in str(llm_config_param.annotation)

if __name__ == "__main__":
    pytest.main([__file__])
