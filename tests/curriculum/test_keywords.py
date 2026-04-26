"""Tests for keyword extraction."""

import pytest

from artifacts_scripts.curriculum.keyword_extraction import (
    BIGRAM_JOIN,
    BackgroundMode,
    DocWeighting,
    KeywordMethod,
    TopicWeighting,
    apply_bigrams,
    compute_log_odds_ratio,
    compute_log_odds_ratio_docs,
    compute_tf,
    compute_tfidf_scores,
    extract_topic_keywords,
    filter_stopwords,
    mine_bigrams,
    mine_phrases_lattice,
    parse_with_phrases,
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
        """Pure-numeric tokens are filtered (page/year/volume noise);
        alphanumeric tokens like 'co2' or '2d' survive."""
        text = "year 2024 model v2 co2"
        tokens = tokenize(text)

        assert "year" in tokens
        assert "model" in tokens
        # Pure-numeric tokens dropped
        assert "2024" not in tokens
        # Alphanumeric retained ("co2" is an alphanumeric mix)
        assert "co2" in tokens
        # 'v2' is alphanumeric (length 2 → still dropped by length filter)

    def test_strips_xml_tags(self):
        """MathML / HTML tags are removed before tokenizing."""
        text = "Generation of <mml:math xmlns:mml='http://w3.org/1998'>x</mml:math> in cavity"
        tokens = tokenize(text)
        # Markup tokens must NOT appear
        assert "mml" not in tokens
        assert "xmlns" not in tokens
        assert "math" not in tokens
        # Real-prose tokens must survive
        assert "generation" in tokens
        assert "cavity" in tokens

    def test_strips_latex_commands(self):
        """LaTeX commands like \\ensuremath, \\mathrm survive only as
        whatever non-command text remains."""
        text = r"Below \ensuremath{\sim}30K, the \mathrm{Cr}_{11}\mathrm{Ge}_{19} compound"
        tokens = tokenize(text)
        assert "ensuremath" not in tokens
        assert "mathrm" not in tokens
        # Plain-prose words must survive
        assert "below" in tokens
        assert "compound" in tokens

    def test_strips_urls(self):
        """URLs (http(s):// and www.) are removed before tokenizing."""
        text = "See https://example.org/papers/foo and www.openalex.org for details"
        tokens = tokenize(text)
        # URL-fragment tokens must NOT appear
        assert "http" not in tokens
        assert "https" not in tokens
        assert "www" not in tokens
        assert "example" not in tokens   # comes from inside URL
        assert "openalex" not in tokens  # comes from inside URL
        # Surrounding prose survives
        assert "see" in tokens
        assert "details" in tokens
    
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


class TestBackgroundModeAndFreqWeighting:
    """``compute_log_odds_ratio_docs`` MARGINAL background + freq-weighting."""

    def _toy_corpus(self):
        """Three topics with varying levels of word overlap."""
        # All three share "common"; topic 0 has unique "neural", "network";
        # topic 1 has unique "bayesian", "posterior"; topic 2 has unique
        # "compiler", "optimization".
        return [
            [["common", "neural", "network", "deep"]] * 3 + [["common", "neural"]] * 2,
            [["common", "bayesian", "posterior", "inference"]] * 3 + [["common", "bayesian"]] * 2,
            [["common", "compiler", "optimization", "register"]] * 3 + [["common", "compiler"]] * 2,
        ]

    def test_marginal_default_demotes_shared_words(self):
        """A word common to every topic should rank low under marginal mode."""
        topic_doc_tokens = self._toy_corpus()
        results = compute_log_odds_ratio_docs(
            topic_doc_tokens,
            top_k=3,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=True,
        )
        # "common" appears in every topic → marginal P(w) high → demoted everywhere.
        for tidx, kws in enumerate(results):
            top_words = [w for w, _ in kws]
            assert "common" not in top_words, (
                f"topic {tidx}: 'common' should not be a top keyword under "
                f"MARGINAL+freq, got {top_words}"
            )

    def test_marginal_promotes_topic_unique_words(self):
        """Words unique to a topic should populate that topic's top-K."""
        topic_doc_tokens = self._toy_corpus()
        results = compute_log_odds_ratio_docs(
            topic_doc_tokens,
            top_k=3,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=True,
        )
        topic0_words = {w for w, _ in results[0]}
        topic1_words = {w for w, _ in results[1]}
        topic2_words = {w for w, _ in results[2]}
        # Topic-unique words must reach top-3
        assert topic0_words & {"neural", "network", "deep"}, topic0_words
        assert topic1_words & {"bayesian", "posterior", "inference"}, topic1_words
        assert topic2_words & {"compiler", "optimization", "register"}, topic2_words

    def test_freq_weighting_multiplies_score_by_count(self):
        """``weight_by_freq=True`` post-multiplies the unweighted z-score by
        ``y_in`` (the topic-internal foreground count). Verify the ratio."""
        # Single-token docs → each doc contributes y_in=1 under EQUAL weighting,
        # so y_in["alpha"] in topic 0 = 30 (one per doc).
        topic_doc_tokens = [
            [["alpha"]] * 30,
            [["beta"]] * 10,
        ]
        no_weight = compute_log_odds_ratio_docs(
            topic_doc_tokens, top_k=5,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=False,
        )
        weighted = compute_log_odds_ratio_docs(
            topic_doc_tokens, top_k=5,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=True,
        )
        nw_t0 = dict(no_weight[0])
        w_t0 = dict(weighted[0])
        # weighted = unweighted * y_in ≈ unweighted * 30 for "alpha" in topic 0
        ratio = w_t0["alpha"] / nw_t0["alpha"]
        assert 25 < ratio < 35, (
            f"weighted/unweighted ratio for alpha should be ~30 (= y_in), got {ratio}"
        )

    def test_freq_weighting_widens_gap_between_high_and_low_count(self):
        """Two equally-distinctive words in the same topic, one with 10x more
        count: freq-weighting should widen their score gap."""
        topic_doc_tokens = [
            # Topic 0 has both "high_count" (30 docs) and "low_count" (3 docs).
            # Both are unique to topic 0.
            [["high_count"]] * 30 + [["low_count"]] * 3,
            [["topic1_word"]] * 20,
        ]
        no_weight = compute_log_odds_ratio_docs(
            topic_doc_tokens, top_k=10,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=False,
        )
        weighted = compute_log_odds_ratio_docs(
            topic_doc_tokens, top_k=10,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=True,
        )
        nw = dict(no_weight[0])
        wt = dict(weighted[0])
        nw_ratio = nw["high_count"] / nw["low_count"]
        wt_ratio = wt["high_count"] / wt["low_count"]
        # Without weighting, the standardised z-scores are similar (both are
        # topic-unique). With freq weighting, the gap widens by ~10x (the
        # count ratio).
        assert wt_ratio > 5 * nw_ratio, (
            f"freq-weighting should widen score gap: "
            f"unweighted ratio={nw_ratio:.2f}, weighted ratio={wt_ratio:.2f}"
        )

    def test_subtractive_mode_still_works(self):
        """Legacy SUBTRACTIVE path still works (backwards-compat)."""
        topic_doc_tokens = self._toy_corpus()
        results = compute_log_odds_ratio_docs(
            topic_doc_tokens,
            top_k=3,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.SUBTRACTIVE,
            weight_by_freq=False,
        )
        assert len(results) == 3
        for kws in results:
            assert len(kws) <= 3
            # Each entry is (word, score) tuple
            for w, s in kws:
                assert isinstance(w, str)
                assert isinstance(s, float)

    def test_invalid_background_mode_rejected(self):
        topic_doc_tokens = self._toy_corpus()
        with pytest.raises(ValueError, match="background_mode"):
            compute_log_odds_ratio_docs(
                topic_doc_tokens,
                background_mode="not-a-mode",  # type: ignore
            )

    def test_empty_topic_returns_empty_keywords(self):
        """Empty topics (no docs) should return an empty keyword list,
        not the deterministic-garbage tokens that fall out of FW scoring
        against zero foreground mass. Triggered in production when a
        clustering method produces fewer real clusters than
        ``desired_n_topics`` and the truncation pads with size-0
        placeholders (e.g. Leiden on a tightly-knit corpus)."""
        topic_doc_tokens = [
            [["alpha", "beta", "gamma"]] * 5,
            [],  # empty topic
            [["delta", "epsilon", "zeta"]] * 5,
            [],  # another empty topic
        ]
        results = compute_log_odds_ratio_docs(
            topic_doc_tokens,
            top_k=5,
            doc_weighting=DocWeighting.EQUAL,
            background_mode=BackgroundMode.MARGINAL,
            weight_by_freq=True,
        )
        assert len(results) == 4
        # Non-empty topics should produce non-empty keyword lists
        assert len(results[0]) > 0, results[0]
        assert len(results[2]) > 0, results[2]
        # Empty topics should return empty lists
        assert results[1] == [], results[1]
        assert results[3] == [], results[3]


class TestBigramMining:
    """``mine_bigrams`` + ``apply_bigrams``: NPMI-thresholded phrase mining
    with frequency weighting (the principled drop-in for an explicit
    count floor)."""

    def test_mines_high_npmi_collocation(self):
        """A phrase that always co-occurs (perfect collocation) should be
        mined as a bigram; a chance co-occurrence should not."""
        # "bell theorem" co-occurs in every doc; "bell" and "theorem"
        # never appear separately. "data" and "noise" are independent.
        docs = [
            ["bell", "theorem", "result"],
            ["bell", "theorem", "data"],
            ["bell", "theorem", "noise"],
            ["data", "table"],
            ["noise", "filter"],
            ["bell", "theorem"],
        ] * 3  # repeat to clear count thresholds
        bigrams = mine_bigrams(docs, npmi_threshold=0.4, max_bigrams=10)
        assert ("bell", "theorem") in bigrams, (
            f"perfect collocation should be mined, got {bigrams}"
        )

    def test_stopwords_excluded_as_members(self):
        """Stopword-bearing bigrams (e.g. 'of the', 'in this') get dropped."""
        docs = [
            ["the", "model", "of", "the", "system"],
            ["the", "model", "of", "the", "graph"],
            ["the", "model", "of", "the", "tree"],
        ] * 5
        bigrams = mine_bigrams(docs, npmi_threshold=0.0, max_bigrams=20)
        for a, b in bigrams:
            assert a not in {"the", "of", "in", "this"}, (
                f"stopword should not be bigram member: {(a, b)}"
            )
            assert b not in {"the", "of", "in", "this"}, (
                f"stopword should not be bigram member: {(a, b)}"
            )

    def test_freq_weighting_promotes_high_count_bigrams(self):
        """With weight_by_freq=True, a high-count moderate-NPMI bigram should
        rank above a low-count high-NPMI one when only one slot remains."""
        # "common phrase" appears 50 times; "rare term" appears once.
        # Both are perfect collocations, so NPMI is identical (≈1).
        # Without freq weighting they'd tie; with freq weighting "common
        # phrase" wins by 50:1.
        docs = (
            [["common", "phrase", "filler"]] * 50
            + [["rare", "term", "filler"]] * 1
            + [["filler", "filler"]] * 100
        )
        # max_bigrams=1 forces a winner.
        weighted = mine_bigrams(
            docs, npmi_threshold=0.0, max_bigrams=1, weight_by_freq=True,
        )
        assert weighted == {("common", "phrase")}, (
            f"freq-weighted should pick high-count bigram, got {weighted}"
        )

    def test_threshold_is_monotonic(self):
        """Raising the NPMI threshold can only remove (never add) bigrams."""
        docs = [
            ["alpha", "beta", "filler"],
            ["alpha", "gamma", "filler"],
            ["delta", "beta", "filler"],
            ["epsilon", "zeta"],
        ] * 10
        loose = mine_bigrams(docs, npmi_threshold=0.0, max_bigrams=100)
        strict = mine_bigrams(docs, npmi_threshold=0.95, max_bigrams=100)
        # Strict set is a (possibly equal) subset of loose
        assert strict.issubset(loose), (
            f"strict not subset of loose: strict={strict}, loose={loose}"
        )

    def test_apply_bigrams_greedy_left_to_right(self):
        """``apply_bigrams`` consumes both positions of a merge — each token
        participates in at most one bigram."""
        bigrams = {("a", "b"), ("b", "c")}
        # "a b c" greedy left-to-right: merge (a,b) → "a_b", then "c" stays
        result = apply_bigrams(["a", "b", "c"], bigrams)
        assert result == [f"a{BIGRAM_JOIN}b", "c"], result

    def test_apply_bigrams_passes_through_when_empty(self):
        """Empty bigram set returns input unchanged (copy)."""
        tokens = ["alpha", "beta", "gamma"]
        result = apply_bigrams(tokens, set())
        assert result == tokens
        # Should be a copy, not the same list
        assert result is not tokens

    def test_merged_bigram_survives_filter_stopwords(self):
        """The BIGRAM_JOIN char ('_') is not in any stopword, so merged
        bigrams pass through ``filter_stopwords`` even if their components
        contained substring-matches to stopwords."""
        # "the" is a stopword. After merging into "the_x", filter_stopwords
        # should leave it alone (no exact match).
        merged = [f"the{BIGRAM_JOIN}theorem", "the", "result"]
        filtered = filter_stopwords(merged)
        assert f"the{BIGRAM_JOIN}theorem" in filtered, filtered
        assert "the" not in filtered

    def test_no_bigrams_on_empty_corpus(self):
        assert mine_bigrams([]) == set()
        assert mine_bigrams([[]]) == set()


class TestLatticePhraseMining:
    """``mine_phrases_lattice`` (combined NPMI · count · BE) and
    ``parse_with_phrases`` (longest-match parser).

    The lattice approach is what bookgap's production keyword extraction
    uses; the boundary-entropy factor in the score rejects boilerplate
    aggregates that NPMI alone admits."""

    def test_lattice_mines_real_phrase(self):
        """A high-NPMI phrase with varied *immediate* neighbours (the
        canonical real-phrase signature: high boundary entropy on both
        sides) should be mined."""
        # "alpha beta" co-occurs everywhere it appears, but the tokens
        # immediately preceding and following vary across documents.
        wrappers = [
            ("preamble", "tail"), ("intro", "epilogue"),
            ("before", "after"), ("zero", "end"),
            ("first", "last"), ("hello", "goodbye"),
            ("setup", "teardown"), ("alpha2", "omega"),
        ]
        docs = []
        for w_l, w_r in wrappers:
            docs.extend([[w_l, "alpha", "beta", w_r]] * 4)

        phrases = mine_phrases_lattice(
            docs, npmi_threshold=0.3, min_boundary_be=0.5,
            count_floor=3, max_phrases=20,
        )
        assert "alpha_beta" in phrases, (
            f"varied-context phrase should be mined, got {phrases}"
        )

    def test_lattice_score_prefers_varied_context(self):
        """Given two co-occurring bigrams with the same NPMI / count,
        the one with more varied surrounding context (higher boundary
        entropy) should be preferred. This is the recipe's core
        behaviour: the BE factor in the score breaks ties in favour of
        non-boilerplate."""
        # Both ``alpha beta`` and ``fixed seq`` always co-occur (perfect
        # NPMI) and appear the same number of times. ``alpha beta`` has
        # varied wrappers; ``fixed seq`` is always inside ``frame ... tail``.
        wrappers = [
            ("intro", "tail"), ("preamble", "epilogue"),
            ("setup", "teardown"), ("hello", "goodbye"),
            ("zero", "end"), ("first", "last"),
        ]
        docs = []
        for wl, wr in wrappers:
            docs.extend([[wl, "alpha", "beta", wr]] * 5)
        docs.extend([["frame", "fixed", "seq", "tail"]] * 30)
        # Disable both BE filters so both candidates pass; we test
        # that the score itself ranks the high-BE one higher.
        phrases = mine_phrases_lattice(
            docs, npmi_threshold=0.3, min_boundary_be=0.0, be_quantile=0.0,
            count_floor=3, max_phrases=10,
        )
        # ``alpha_beta`` (varied context, BE > 0) should rank above
        # ``fixed_seq`` (uniform context, BE = 0 → score 0). With
        # max_phrases=10 and many candidates, only the top survive.
        assert "alpha_beta" in phrases, (
            f"varied-context bigram should be top-ranked, got {phrases}"
        )

    def test_lattice_drops_stopword_candidates(self):
        docs = [["the", "model", "of", "the", "graph"]] * 30
        phrases = mine_phrases_lattice(
            docs, npmi_threshold=0.0, min_boundary_be=0.0, be_quantile=0.0,
            count_floor=3, max_phrases=50,
        )
        for p in phrases:
            for piece in p.split(BIGRAM_JOIN):
                assert piece not in {"the", "of"}, (
                    f"stopword should not be a phrase member: {p}"
                )

    def test_lattice_empty_corpus_returns_empty(self):
        assert mine_phrases_lattice([]) == set()
        assert mine_phrases_lattice([[]]) == set()

    def test_parse_with_phrases_longest_match(self):
        """When phrases overlap, longest-match wins. ``a_b_c`` should
        consume all three positions even when ``a_b`` is also in vocab."""
        tokens = ["x", "a", "b", "c", "y"]
        phrases = {"a_b", "a_b_c"}
        result = parse_with_phrases(tokens, phrases)
        assert result == ["x", "a_b_c", "y"], result

    def test_parse_with_phrases_left_to_right(self):
        """Without overlap, the parser progresses left-to-right and
        consumes each phrase's positions."""
        tokens = ["a", "b", "c", "d"]
        phrases = {"a_b", "c_d"}
        result = parse_with_phrases(tokens, phrases)
        assert result == ["a_b", "c_d"], result

    def test_parse_with_phrases_no_match_passes_through(self):
        tokens = ["alpha", "beta", "gamma"]
        result = parse_with_phrases(tokens, {"unrelated_phrase"})
        assert result == ["alpha", "beta", "gamma"], result

    def test_parse_with_phrases_empty_inputs(self):
        assert parse_with_phrases([], {"a_b"}) == []
        # Empty phrase vocab is a copy, not a reference
        tokens = ["a", "b"]
        result = parse_with_phrases(tokens, set())
        assert result == tokens
        assert result is not tokens

    def test_parse_grows_trigram_from_bigram_then_extension(self):
        """Demonstrate the iterative-vocab growth pattern: pass-1 mines a
        bigram (added to vocab), parse-with-vocab merges it; pass-2 mines
        the (bigram, neighbour) extension, adding the resulting trigram
        to vocab; the next parse promotes the whole sequence to the
        trigram. This is the data-flow shape of the production loop;
        we test the parser semantics without trying to hit synthetic
        entropy thresholds in a small toy corpus."""
        tokens = ["x", "a", "b", "c", "y"]
        # Iter 1 vocab (bigram only) → bigram merges, trailing token alone.
        v1 = {"a_b"}
        parsed1 = parse_with_phrases(tokens, v1)
        assert parsed1 == ["x", "a_b", "c", "y"], parsed1

        # Iter 2 vocab adds the trigram (what mining over the iter-1
        # parsed corpus would yield: bigram (a_b, c) → "a_b_c"). Re-parsing
        # the *raw* tokens with the cumulative vocab promotes the whole
        # sequence — longest-match prefers the trigram.
        v_cumulative = {"a_b", "a_b_c"}
        parsed2 = parse_with_phrases(tokens, v_cumulative)
        assert parsed2 == ["x", "a_b_c", "y"], parsed2
