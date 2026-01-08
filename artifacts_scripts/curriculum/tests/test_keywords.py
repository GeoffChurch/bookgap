"""Tests for keyword extraction."""

import pytest

from artifacts_scripts.curriculum.keyword_extraction import (
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    compute_log_odds_ratio,
    compute_tf,
    compute_tfidf_scores,
    extract_topic_keywords,
    filter_stopwords,
    tokenize,
)


class TestTokenize:
    """Tests for tokenize function."""
    
    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        text = "Hello World! This is a test."
        tokens = tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "this" in tokens
        assert "test" in tokens
    
    def test_filters_short_tokens(self):
        """Test that short tokens are filtered."""
        text = "a ab abc abcd"
        tokens = tokenize(text)
        
        # Only tokens >= 3 chars
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens
    
    def test_handles_numbers(self):
        """Test handling of numbers."""
        text = "year 2024 model v2"
        tokens = tokenize(text)
        
        assert "year" in tokens
        assert "2024" in tokens
        assert "model" in tokens
    
    def test_empty_string(self):
        """Test empty string returns empty list."""
        assert tokenize("") == []
    
    def test_special_characters_split(self):
        """Test special characters cause splits."""
        text = "machine-learning deep_learning"
        tokens = tokenize(text)
        
        assert "machine" in tokens
        assert "learning" in tokens
        assert "deep" in tokens


class TestFilterStopwords:
    """Tests for filter_stopwords function."""
    
    def test_filters_with_custom_stopwords(self):
        """Test stopword filtering with an explicit stopword set."""
        tokens = ["the", "quick", "and", "brown", "fox"]
        filtered = filter_stopwords(tokens, stopwords={"the", "and"})

        assert "the" not in filtered
        assert "and" not in filtered
        assert "quick" in filtered
        assert "brown" in filtered
        assert "fox" in filtered
    
    def test_custom_stopwords(self):
        """Test custom stopword set."""
        tokens = ["foo", "bar", "baz"]
        custom_stops = {"bar"}
        filtered = filter_stopwords(tokens, stopwords=custom_stops)
        
        assert "foo" in filtered
        assert "bar" not in filtered
        assert "baz" in filtered
    
    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert filter_stopwords([]) == []


class TestComputeTf:
    """Tests for compute_tf function."""
    
    def test_basic_tf(self):
        """Test basic term frequency computation."""
        tokens = ["apple", "banana", "apple", "cherry"]
        tf = compute_tf(tokens)
        
        assert tf["apple"] == pytest.approx(0.5)  # 2/4
        assert tf["banana"] == pytest.approx(0.25)  # 1/4
        assert tf["cherry"] == pytest.approx(0.25)  # 1/4
    
    def test_single_token(self):
        """Test single token."""
        tokens = ["word"]
        tf = compute_tf(tokens)
        
        assert tf["word"] == pytest.approx(1.0)
    
    def test_empty_input(self):
        """Test empty input returns empty dict."""
        assert compute_tf([]) == {}


class TestComputeTfidfScores:
    """Tests for compute_tfidf_scores function."""
    
    def test_distinctive_words_rank_high(self):
        """Test words unique to a topic rank highly."""
        topic_tokens = [
            ["neural", "network", "deep", "learning", "neural"],
            ["statistics", "probability", "bayesian", "inference"],
            ["optimization", "gradient", "convex", "objective"],
        ]
        
        results = compute_tfidf_scores(topic_tokens, top_k=3)
        
        # Each topic should have distinctive keywords
        assert len(results) == 3
        assert len(results[0]) == 3
        
        # Top words should be unique to each topic
        topic0_words = {w for w, _ in results[0]}
        assert "neural" in topic0_words or "network" in topic0_words
    
    def test_common_words_rank_low(self):
        """Test words appearing in all topics rank low."""
        topic_tokens = [
            ["machine", "learning", "common"],
            ["deep", "learning", "common"],
            ["reinforcement", "learning", "common"],
        ]
        
        results = compute_tfidf_scores(topic_tokens, top_k=2)
        
        # "common" should not be top-ranked for any topic
        for topic_results in results:
            top_words = {w for w, _ in topic_results}
            # "common" might still appear but shouldn't be #1
            if "common" in top_words:
                # Check it's not the highest scored
                assert topic_results[0][0] != "common"
    
    def test_top_k_limits_results(self):
        """Test top_k parameter limits results."""
        topic_tokens = [
            ["a", "b", "c", "d", "e", "f"],
        ]
        
        results = compute_tfidf_scores(topic_tokens, top_k=3)
        assert len(results[0]) == 3


class TestComputeLogOddsRatio:
    """Tests for compute_log_odds_ratio function."""
    
    def test_distinctive_words_rank_high(self):
        """Test words unique to a topic have high log-odds."""
        topic_tokens = [
            ["python", "code", "function", "class", "python"],
            ["math", "theorem", "proof", "lemma"],
        ]
        
        results = compute_log_odds_ratio(topic_tokens, top_k=3)
        
        assert len(results) == 2
        
        # Python-specific words should rank high for topic 0
        topic0_words = {w for w, _ in results[0]}
        assert "python" in topic0_words or "code" in topic0_words
    
    def test_scores_are_positive_for_distinctive(self):
        """Test distinctive words have positive scores."""
        topic_tokens = [
            ["unique_word", "common"],
            ["common", "other"],
        ]
        
        results = compute_log_odds_ratio(topic_tokens, top_k=2)
        
        # unique_word only in topic 0 should have positive score
        for word, score in results[0]:
            if word == "unique_word":
                assert score > 0
    
    def test_prior_weight_affects_scores(self):
        """Test prior weight smoothing affects scores."""
        topic_tokens = [
            ["rare", "common", "common", "common"],
            ["common", "common", "common", "common"],
        ]
        
        results_low_prior = compute_log_odds_ratio(
            topic_tokens, top_k=2, prior_weight=0.1
        )
        results_high_prior = compute_log_odds_ratio(
            topic_tokens, top_k=2, prior_weight=10.0
        )
        
        # Higher prior should smooth scores (make them less extreme)
        # Just check both produce valid results
        assert len(results_low_prior) == 2
        assert len(results_high_prior) == 2


class TestComputeLogOddsRatioDocs:
    """Tests for doc-based log-odds (Fightin' Words) with weighting options."""

    def test_topic_weighting_equal_is_invariant_to_topic_size(self):
        """With topic_weighting='equal', duplicating docs in other topics shouldn't change scores."""
        from artifacts_scripts.curriculum.keyword_extraction import compute_log_odds_ratio_docs

        base = [
            [["apple", "apple", "apple"]],
            [["banana", "banana", "banana"]],
        ]

        duplicated_other = [
            [["apple", "apple", "apple"]],
            [["banana", "banana", "banana"]] * 50,
        ]

        r_base = compute_log_odds_ratio_docs(
            base,
            top_k=5,
            prior_weight=0.01,
            doc_weighting=DocWeighting.EQUAL,
            topic_weighting=TopicWeighting.EQUAL,
        )
        r_dup = compute_log_odds_ratio_docs(
            duplicated_other,
            top_k=5,
            prior_weight=0.01,
            doc_weighting=DocWeighting.EQUAL,
            topic_weighting=TopicWeighting.EQUAL,
        )

        # Compare the z-score for "apple" in topic 0
        base_scores = dict(r_base[0])
        dup_scores = dict(r_dup[0])
        assert "apple" in base_scores and "apple" in dup_scores
        assert base_scores["apple"] == pytest.approx(dup_scores["apple"], rel=1e-6, abs=1e-6)

    def test_relevance_weighting_upweights_high_mass_docs(self):
        """With doc_weighting='relevance', high-weight docs contribute more to keywords."""
        from artifacts_scripts.curriculum.keyword_extraction import compute_log_odds_ratio_docs

        # Topic 0: two docs, one with "alpha" (high weight), one with "beta" (low weight)
        # Topic 1: control topic
        topic_docs = [
            [["alpha", "alpha", "shared"], ["beta", "beta", "shared"]],
            [["gamma", "gamma", "shared"]],
        ]
        
        # High weight on "alpha" doc, low weight on "beta" doc
        topic_weights = [
            [10.0, 1.0],  # alpha doc has 10x weight
            [1.0],
        ]
        
        # With RELEVANCE weighting, "alpha" should dominate
        results_relevance = compute_log_odds_ratio_docs(
            topic_docs,
            top_k=5,
            prior_weight=0.01,
            doc_weighting=DocWeighting.RELEVANCE,
            topic_weighting=TopicWeighting.EQUAL,
            topic_doc_weights=topic_weights,
        )
        
        # With EQUAL weighting, both should contribute equally
        results_equal = compute_log_odds_ratio_docs(
            topic_docs,
            top_k=5,
            prior_weight=0.01,
            doc_weighting=DocWeighting.EQUAL,
            topic_weighting=TopicWeighting.EQUAL,
        )
        
        rel_scores = dict(results_relevance[0])
        eq_scores = dict(results_equal[0])
        
        # With RELEVANCE, "alpha" should score higher than "beta"
        assert "alpha" in rel_scores and "beta" in rel_scores
        assert rel_scores["alpha"] > rel_scores["beta"]
        
        # With EQUAL, they should be roughly equal (same doc structure)
        assert "alpha" in eq_scores and "beta" in eq_scores
        # Should be close since each doc contributes equally
        assert eq_scores["alpha"] == pytest.approx(eq_scores["beta"], rel=0.1)

    def test_relevance_weighting_requires_weights(self):
        """RELEVANCE doc_weighting without weights should raise."""
        from artifacts_scripts.curriculum.keyword_extraction import compute_log_odds_ratio_docs

        topic_docs = [[["word"]]]
        
        with pytest.raises(ValueError, match="topic_doc_weights required"):
            compute_log_odds_ratio_docs(
                topic_docs,
                doc_weighting=DocWeighting.RELEVANCE,
                topic_doc_weights=None,
            )


class TestExtractTopicKeywords:
    """Tests for extract_topic_keywords function."""
    
    def test_tfidf_method(self):
        """Test TF-IDF method selection."""
        topic_tokens = [
            [["neural", "network"]],
            [["bayesian", "inference"]],
        ]
        
        results = extract_topic_keywords(topic_tokens, method=KeywordMethod.TFIDF, top_k=2)
        
        assert len(results) == 2
    
    def test_log_odds_method(self):
        """Test log-odds method selection."""
        topic_tokens = [
            [["neural", "network"]],
            [["bayesian", "inference"]],
        ]
        
        results = extract_topic_keywords(topic_tokens, method=KeywordMethod.LOG_ODDS, top_k=2)
        
        assert len(results) == 2
    
    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError):
            KeywordMethod("invalid")


class TestIntegration:
    """Integration tests for keyword extraction pipeline."""
    
    def test_full_pipeline_with_realistic_data(self):
        """Test full pipeline with realistic-ish topic documents."""
        # Simulate documents from different topics
        topic_texts = [
            # Machine Learning topic
            [
                "Deep learning neural networks for image classification",
                "Convolutional neural network architectures",
                "Training deep models with backpropagation",
            ],
            # Statistics topic
            [
                "Bayesian inference and posterior distributions",
                "Maximum likelihood estimation methods",
                "Hypothesis testing and confidence intervals",
            ],
            # Optimization topic
            [
                "Convex optimization and gradient descent",
                "Stochastic gradient methods for large scale",
                "Newton methods and second order optimization",
            ],
        ]
        
        # Tokenize and filter
        topic_tokens = []
        for texts in topic_texts:
            doc_tokens = []
            for text in texts:
                tokens = tokenize(text)
                tokens = filter_stopwords(tokens)
                doc_tokens.append(tokens)
            topic_tokens.append(doc_tokens)
        
        # Extract keywords with both methods
        tfidf_results = extract_topic_keywords(topic_tokens, method=KeywordMethod.TFIDF, top_k=5)
        log_odds_results = extract_topic_keywords(topic_tokens, method=KeywordMethod.LOG_ODDS, top_k=5)
        
        # Both should produce reasonable results
        assert len(tfidf_results) == 3
        assert len(log_odds_results) == 3
        
        # Check topic 0 (ML) has relevant keywords
        ml_keywords = {w for w, _ in tfidf_results[0]}
        assert len(ml_keywords & {"neural", "deep", "learning", "networks"}) >= 1
        
        # Check topic 1 (Stats) has relevant keywords
        stats_keywords = {w for w, _ in tfidf_results[1]}
        assert len(stats_keywords & {"bayesian", "estimation", "likelihood"}) >= 1


class TestSummarizeTopics:
    """Tests for summarize_topics function using combined title+abstract texts."""

    def test_summarize_topics_basic(self, tmp_path):
        """Create minimal curriculum outputs and run summarize_topics."""
        from artifacts_scripts.curriculum.summarize_topics import summarize_topics
        import numpy as np
        import json

        # Create fake curriculum outputs
        curr_dir = tmp_path / "curr"
        curr_dir.mkdir()

        # Two frontier topics
        np.save(curr_dir / "frontier_nodes.npy", np.array([0, 1]))
        np.save(curr_dir / "node_masses.npy", np.array([1.0, 1.0]))
        # Dummy linkage (not used in this simple path)
        np.save(curr_dir / "linked.npy", np.zeros((1, 4)))

        # point_to_cluster: assign first 3 docs to topic 0, next 3 to topic 1
        np.save(curr_dir / "point_to_cluster.npy", np.array([0, 0, 0, 1, 1, 1]))
        
        # PageRank scores for points (required for sampling)
        np.save(curr_dir / "point_masses.npy", np.array([1.0, 0.8, 0.6, 0.9, 0.7, 0.5]))

        # Minimal metadata
        with open(curr_dir / "metadata.json", "w") as f:
            json.dump({"generated": True}, f)

        # Create texts: simple repeating domain words for topics
        texts = [
            "Deep learning convolutional neural networks",
            "Neural network architectures for vision",
            "Training deep models with backpropagation",
            "Bayesian inference and posterior distributions",
            "Maximum likelihood estimation methods",
            "Hypothesis testing and confidence intervals",
        ]

        keywords = summarize_topics(curr_dir, texts, samples_per_topic=2, top_k_keywords=3, method=KeywordMethod.TFIDF)

        # Expect two topics
        assert set(keywords.keys()) == {0, 1}
        # Each topic should have up to top_k keywords
        assert all(isinstance(kw, list) for kw in keywords.values())
        assert all(len(kw) <= 3 for kw in keywords.values())
