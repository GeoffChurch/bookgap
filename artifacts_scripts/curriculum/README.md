# Curriculum Building Module

Build a curriculum from a target topic by mining prerequisite relationships using personalized PageRank on the citation graph and organizing them into a hierarchical dendrogram structure.

## Overview

This module implements the curriculum building workflow described in `textbook_gen_visualization/proposal.md`:

1. **Build Topic Dendrogram** — Agglomerative clustering over embeddings
2. **Mine Prerequisites** — Personalized PageRank from target topic → lift to dendrogram → identify frontier → compute inter-topic dependencies

## Quick Start

The unified entry point is `main.py`, which orchestrates the full pipeline:

```bash
# Build curriculum + extract keywords + generate plot (all in one command)
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    desired_n_topics=5 \
    output_dir=results/lstm

# Skip plotting
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    desired_n_topics=5 \
    plot.enabled=false

# Skip keyword extraction
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    desired_n_topics=5 \
    keywords.enabled=false

# Full run with rank-before-clustering optimization (clusters only top-k relevant works)
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    desired_n_topics=10 \
    rank_before_clustering=true \
    ranking_top_k=100000

# Search the cache for papers by keyword (useful for finding OpenAlex IDs)
python -m artifacts_scripts.curriculum.search_cache "chain of thought" --limit 10
```

## Main Entry Point: `main.py`

The unified pipeline script that runs:

1. **Curriculum Building** — Citation graph + PPR + clustering + frontier selection
2. **Keyword Extraction** — TF-IDF topic summarization (optional)
3. **Visualization** — PCA projection with sequence arrows (optional)

### Configuration

All options are set via Hydra configuration:

```bash
# Basic usage
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    desired_n_topics=10

# Keyword extraction options
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    keywords.enabled=true \
    keywords.method=tfidf \      # or log_odds
    keywords.top_k=10 \
    keywords.samples_per_topic=100

# Plotting options
python -m artifacts_scripts.curriculum.main \
    "target_work_ids=['https://openalex.org/W2064675550']" \
    plot.enabled=true \
    plot.with_background=false \
    plot.figsize=[12,10] \
    plot.dpi=150
```

**Outputs** (saved to `output_dir`):
- `frontier_nodes.npy` — Frontier topic node indices
- `adj_frontier.npy` — Inter-topic dependency matrix
- `leaf_to_cluster.npy` — Leaf-to-cluster assignment
- `node_masses.npy` — PageRank masses for all nodes
- `metadata.json` — Run configuration and statistics
- `topic_work_ids.json` — Work IDs for each topic
- `topic_keywords.json` — Keywords for each topic (if enabled)
- `curriculum_plot.png` — Visualization (if enabled)

## Library Modules

The following modules provide reusable functions and are used by `main.py`:

### `build_curriculum.py`

Core curriculum building logic:
- Loads cloud and embeddings
- Builds citation graph and dendrogram
- Runs personalized PageRank from target works
- Identifies frontier topics
- Computes inter-topic dependency matrix

### `summarize_topics.py`

Generates keyword summaries for each curriculum topic using TF-IDF or log-odds keyword extraction.

**Key function:**
```python
from artifacts_scripts.curriculum.summarize_topics import summarize_topics

keywords = summarize_topics(
    curriculum_dir=Path("results/lstm"),
    # Pass pre-built text strings (title + abstract) instead of full work dicts
    texts=[f"{t}\n{a}" for t, a in zip(cloud.work_titles(), cloud.work_abstracts())],
    samples_per_topic=100,
    top_k_keywords=10,
    method="tfidf",
    seed=42,
)
```

### `plot_curriculum.py`

Visualizes the curriculum with PCA projection and linearization arrows.

**Key functions:**
```python
from artifacts_scripts.curriculum.plot_curriculum import (
    plot_curriculum,
)

# Basic plot with topic centroids
plot_curriculum(
    curriculum_dir=Path("results/lstm"),
    embeddings=embeddings,
    topic_keywords=keywords,
)

# With all works as background points
plot_curriculum(
    curriculum_dir=Path("results/lstm"),
    embeddings=embeddings,
    topic_keywords=keywords,
    background=True,
    max_points_per_topic=1000,
)
```

### `search_cache.py`

Search the local OpenAlex cache (~1.5M works) for papers by keyword in title or abstract.

```bash
# Basic search
python -m artifacts_scripts.curriculum.search_cache "attention mechanism"

# Limit results
python -m artifacts_scripts.curriculum.search_cache "transformer" --limit 5

# Case-sensitive search
python -m artifacts_scripts.curriculum.search_cache "BERT" --case-sensitive
```

**Output:** Displays matching works with OpenAlex IDs that can be used as `target_work_ids`.
  - Topic centroids as labeled points
  - Arrows showing linearization order (prerequisites first)
  - Topics numbered in learning order
  - Keyword labels (if `topic_keywords.json` exists)

## Configuration

```yaml
curriculum:
  target_work_ids:
    - "https://openalex.org/W..."
  desired_n_topics: 10       # Target number of curriculum topics
    method: "circuit_rank"     # or "page_rank"
    pagerank:
        alpha: 0.85              # PageRank damping factor
        tol: 1e-6                # Convergence tolerance
        max_iter: 100            # Maximum iterations
    circuitrank:
        similarity:
            density_correction_alpha: 0.0  # 0 disables density correction
  linkage_method: "ward"     # Clustering method (see below)
  rank_before_clustering: true  # Rank first, cluster only top-k works
  ranking_top_k: 100000      # Number of top works to cluster (when rank_before_clustering=true)
  output_dir: null           # Output directory (default: results/)
  
  # Keyword extraction options
  keywords:
    enabled: true
    method: "tfidf"          # tfidf or log_odds
    top_k: 10
    samples_per_topic: 100
    seed: 42
  
  # Plotting options  
  plot:
    enabled: true
    with_background: false
    figsize: [12, 10]
    dpi: 150
        # dimensionality reduction method: pca | lda | umap | lpp
        # - `lpp`: Locality Preserving Projection using the symmetrized citation graph
        #   (S = (A + A^T)/2), solves the GEP X^T L X a = lambda X^T D X a and projects
        #   onto the smallest eigenvectors; center=X is applied by default for visualization.
```

### Clustering Methods

| Method | Memory | Time | Best For |
|--------|--------|------|----------|
| `ward` | O(n²) | Fast | n < 100k works |
| `bisecting_kmeans` | O(n) | Slower | n > 100k works |

- **`ward`**: Scipy agglomerative clustering. High-quality balanced trees but requires ~8 bytes × n² memory for pairwise distances.
- **`bisecting_kmeans`**: Recursive top-down K-Means splitting. Scales to millions of points with O(n) memory.

### Rank-Before-Clustering Optimization

When `rank_before_clustering=true`, the pipeline:
1. Runs ranking (PageRank or CircuitRank) on the full citation graph
2. Selects top `ranking_top_k` works by relevance mass
3. Clusters only these relevant works

This is much faster than clustering the full dataset and produces more coherent topics since only prerequisite-relevant works are included.

## Python API

```python
from artifacts_scripts.curriculum.build_curriculum import run_curriculum_building
from artifacts_scripts.curriculum.dendrogram_ops import (
    compute_dendrogram_masses,
    find_mass_threshold_for_n_topics,
    get_frontier_nodes,
    gather_leaves,
    map_leaves_to_frontier_clusters,
)
from artifacts_scripts.curriculum.pagerank import (
    build_citation_adjacency_from_ids,
    compute_pagerank,
    compute_frontier_adjacency,
)
from artifacts_scripts.curriculum.summarize_topics import (
    sample_works_from_cluster,
    summarize_topics,
    load_curriculum_outputs,
)
from artifacts_scripts.curriculum.plot_curriculum import (
    plot_curriculum,
    plot_curriculum_with_background,
)
```
    summarize_topics,
)
from artifacts_scripts.curriculum.plot_curriculum import (
    plot_curriculum,
    plot_curriculum_with_background,
)
```

## Algorithm Summary

1. **Dendrogram Construction**
   - Apply Ward's agglomerative clustering to embeddings
   - Results in hierarchical topic structure

2. **Personalized PageRank**
   - Seed on target work IDs
   - Flow follows citation edges (citer → cited)
   - Mass accumulates on foundational/prerequisite works

3. **Mass Propagation**
   - Sum leaf masses up to internal nodes
   - Each node's mass = importance as prerequisite

4. **Frontier Selection**
   - Binary search for threshold τ yielding desired topic count
   - Frontier = nodes with mass > τ and children with mass ≤ τ

5. **Inter-topic Dependencies**
   - Batched PageRank from each frontier topic
   - Project masses onto other topics
   - Results in |T| × |T| asymmetric dependency matrix

6. **Linearization**
   - Score(t) = in-degree - out-degree
   - Sort by score (prerequisites first)

## Tests

```bash
# Run all tests
python -m pytest artifacts_scripts/curriculum/tests/ -v

# Run specific test file
python -m pytest artifacts_scripts/curriculum/tests/test_pagerank.py -v
```

## Complete Workflow Example

Here's a full example workflow for building a curriculum for the LSTM paper:

```bash
# Using the unified main.py (recommended)
# https://openalex.org/W2064675550 — "Long Short-Term Memory" (LSTM) paper
# https://openalex.org/W4391021820 — "Low-Rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition"
python -m artifacts_scripts.curriculum.main "source_work_ids=['https://openalex.org/W2064675550']" "target_work_ids=['https://openalex.org/W4391021820']" output_dir=lstm_to_lora_audio
python -m artifacts_scripts.curriculum.main "target_work_ids=['https://openalex.org/W2064675550', 'https://openalex.org/W4391021820']" output_dir=lstm_lora_audio
```

The outputs will be in `artifacts/curriculum/lstm_lora_audio/`:
- `metadata.json` — Run configuration and statistics
- `frontier_nodes.npy` — Frontier topic node indices
- `adj_frontier.npy` — Inter-topic dependency matrix
- `topic_work_ids.json` — Work IDs for each topic
- `topic_keywords.json` — Keywords for each topic
- `curriculum_plot.png` — Visualization of curriculum sequence

## See Also

- `PLAN.md` — Detailed implementation plan
- `textbook_gen_visualization/proposal.md` — Original algorithm description
- `textbook_gen_visualization/data_pipeline.py` — Reference implementation (synthetic data)
