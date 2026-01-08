"""Generate markdown summary for curriculum run."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from bookgap.cloud import Cloud
from bookgap.util import LOGGER
from .summarize_topics import load_curriculum_outputs

# Constants for summary generation
NUM_KEYWORDS_IN_SUMMARY = 10
NUM_WORKS_IN_SUMMARY = 3


def _compute_topic_mass_stats(
    point_masses: np.ndarray,
    point_to_cluster: np.ndarray,
    n_topics: int,
) -> Dict[int, Dict[str, float]]:
    """Compute entropy and Gini coefficient for each topic's work relevances.
    
    Returns:
        Dictionary mapping topic_idx to stats dict with keys:
        - total_mass: sum of work masses in topic
        - entropy: Shannon entropy of normalized masses
        - entropy_ratio: entropy / max possible entropy
        - gini: Gini coefficient of masses
    """
    topic_stats = {}
    
    for topic_idx in range(n_topics):
        local_indices = np.where(point_to_cluster == topic_idx)[0]
        
        if len(local_indices) == 0:
            topic_stats[topic_idx] = {
                "total_mass": 0.0,
                "entropy": 0.0,
                "entropy_ratio": 0.0,
                "gini": 0.0,
            }
            continue
            
        masses = point_masses[local_indices]
        total_mass = float(np.sum(masses))
        
        # Normalize to get distribution
        if total_mass > 0:
            probs = masses / total_mass
            
            # Entropy
            nonzero_probs = probs[probs > 0]
            entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
            max_entropy = np.log(len(masses)) if len(masses) > 1 else 1.0
            entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Gini coefficient
            sorted_masses = np.sort(masses)
            n = len(sorted_masses)
            cumsum = np.cumsum(sorted_masses)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_masses)) - (n + 1) * total_mass) / (n * total_mass) if total_mass > 0 else 0.0
        else:
            entropy = 0.0
            entropy_ratio = 0.0
            gini = 0.0
        
        topic_stats[topic_idx] = {
            "total_mass": total_mass,
            "entropy": float(entropy),
            "entropy_ratio": float(entropy_ratio),
            "gini": float(gini),
        }
    
    return topic_stats


def _format_mass_stats_section(metadata: Dict[str, Any]) -> str:
    """Format mass distribution statistics as markdown."""
    mass_stats = metadata.get("mass_distribution_stats", {})
    cluster_stats = metadata.get("cluster_size_stats", {})
    adj_stats = metadata.get("adjacency_stats", {})
    
    lines = []
    lines.append("## Diagnostic Statistics\n")
    
    # Mass distribution
    lines.append("### Mass Distribution\n")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    
    if mass_stats:
        lines.append(f"| Total Mass | {mass_stats.get('total_mass', 0):.4f} |")
        lines.append(f"| Entropy | {mass_stats.get('entropy', 0):.4f} ({100*mass_stats.get('entropy_ratio', 0):.1f}% of max) |")
        lines.append(f"| Effective # of Works | {mass_stats.get('effective_n', 0):.0f} |")
        lines.append(f"| Gini Coefficient | {mass_stats.get('gini', 0):.4f} |")
        lines.append(f"| Non-zero Masses | {mass_stats.get('n_nonzero', 0):,} |")
        lines.append(f"| Masses > 1e-4 | {mass_stats.get('n_above_1e4', 0):,} |")
        
        # Percentiles
        percentiles = mass_stats.get('percentiles', {})
        if percentiles:
            lines.append(f"| 99th Percentile | {percentiles.get('p99', 0):.2e} |")
            lines.append(f"| 99.9th Percentile | {percentiles.get('p99.9', 0):.2e} |")
    
    lines.append("")
    
    # Cluster sizes
    lines.append("### Cluster Sizes\n")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    
    if cluster_stats:
        lines.append(f"| Number of Clusters | {cluster_stats.get('n_clusters', 0)} |")
        lines.append(f"| Min Size | {cluster_stats.get('min_size', 0):,} |")
        lines.append(f"| Max Size | {cluster_stats.get('max_size', 0):,} |")
        lines.append(f"| Mean Size | {cluster_stats.get('mean_size', 0):.1f} |")
        lines.append(f"| Std Dev | {cluster_stats.get('std_size', 0):.1f} |")
        lines.append(f"| Singletons | {cluster_stats.get('n_singletons', 0)} |")
    
    lines.append("")
    
    # Inter-topic adjacency
    lines.append("### Inter-Topic Adjacency\n")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    
    if adj_stats:
        lines.append(f"| Off-diagonal Sum | {adj_stats.get('off_diag_sum', 0):.4f} |")
        lines.append(f"| Non-zero Off-diagonal | {adj_stats.get('n_nonzero_off_diag', 0)} |")
        lines.append(f"| All Zeros | {'Yes ⚠️' if adj_stats.get('is_all_zeros', False) else 'No'} |")
    
    lines.append("")
    
    return "\n".join(lines)


def _format_topic_overview_section(
    point_to_cluster: np.ndarray,
    point_masses: np.ndarray,
    frontier_nodes: np.ndarray,
    node_masses: np.ndarray,
    topic_keywords: Optional[Dict[int, List[Tuple[str, float]]]],
    titles: List[str],
    filtered_indices: Optional[np.ndarray],
    topic_stats: Dict[int, Dict[str, float]],
) -> str:
    """Format a brief topic overview table showing size, relevance stats, keywords, and top work."""
    n_topics = len(frontier_nodes)
    
    lines = []
    lines.append("### Topic Overview\n")
    lines.append("| Topic | Size | Relevance | Entropy | Gini | Top Keywords | Top Work |")
    lines.append("| :---: | ---: | ---: | ---: | ---: | :--- | :--- |")
    
    for topic_idx in range(n_topics):
        # Get topic size
        local_indices = np.where(point_to_cluster == topic_idx)[0]
        size = len(local_indices)
        
        # Get topic stats
        stats = topic_stats.get(topic_idx, {})
        total_mass = stats.get("total_mass", 0.0)
        entropy_ratio = stats.get("entropy_ratio", 0.0)
        gini = stats.get("gini", 0.0)
        
        # Get top 3 keywords
        if topic_keywords and topic_idx in topic_keywords:
            top_kws = [kw for kw, _ in topic_keywords[topic_idx][:3]]
            keywords_str = ", ".join(top_kws)
        else:
            keywords_str = "_none_"
        
        # Get top work title
        if size > 0:
            topic_local_masses = point_masses[local_indices]
            top_local_idx = local_indices[np.argmax(topic_local_masses)]
            
            if filtered_indices is not None:
                cloud_idx = filtered_indices[top_local_idx]
            else:
                cloud_idx = top_local_idx
            
            top_title = titles[cloud_idx]
            # Truncate title if too long
            if len(top_title) > 50:
                top_title = top_title[:47] + "..."
        else:
            top_title = "_none_"
        
        lines.append(f"| {topic_idx} | {size:,} | {total_mass:.4f} | {100*entropy_ratio:.1f}% | {gini:.4f} | {keywords_str} | {top_title} |")
    
    lines.append("")
    lines.append("---\n")
    
    return "\n".join(lines)


def generate_curriculum_summary(
    curriculum_dir: Path,
    cloud: Cloud,
    filtered_indices: Optional[np.ndarray],
    topic_keywords: Optional[Dict[int, List[Tuple[str, float]]]],
) -> None:
    """Generate a markdown summary of the curriculum.
    
    Args:
        curriculum_dir: Directory containing curriculum outputs.
        cloud: Cloud object for accessing work details.
        filtered_indices: Array of indices into cloud if filtered, else None.
        topic_keywords: Dictionary mapping topic index to list of (keyword, score) tuples.
    """
    LOGGER.info("Generating curriculum summary markdown...")
    
    outputs = load_curriculum_outputs(curriculum_dir)
    
    # Load necessary data
    point_to_cluster = outputs.point_to_cluster
    point_masses = outputs.point_masses
    node_masses = outputs.node_masses
    frontier_nodes = outputs.frontier_nodes
    
    if point_to_cluster is None or point_masses is None or node_masses is None or frontier_nodes is None:
        LOGGER.warning("Missing necessary outputs for summary generation.")
        return

    # Get linearization order from metadata
    linearization_order = outputs.metadata.get("linearization_order")
    if linearization_order is None:
        # Fallback to simple range if not found
        linearization_order = list(range(len(frontier_nodes)))
        
    # Load cloud data
    titles = cloud.work_titles()
    abstracts = cloud.work_abstracts()
    work_ids = cloud.work_ids()
    publication_dates = cloud.work_publication_dates()
    
    summary_path = curriculum_dir / "summary.md"
    
    with open(summary_path, "w") as f:
        f.write("# Curriculum Summary\n\n")
        
        # Write diagnostic statistics section at the top
        if outputs.metadata:
            f.write(_format_mass_stats_section(outputs.metadata))
        
        # Compute per-topic mass statistics
        n_topics = len(frontier_nodes)
        topic_stats = _compute_topic_mass_stats(point_masses, point_to_cluster, n_topics)
        
        # Write topic overview table
        f.write(_format_topic_overview_section(
            point_to_cluster=point_to_cluster,
            point_masses=point_masses,
            frontier_nodes=frontier_nodes,
            node_masses=node_masses,
            topic_keywords=topic_keywords,
            titles=titles,
            filtered_indices=filtered_indices,
            topic_stats=topic_stats,
        ))
        
        for rank, topic_idx in enumerate(linearization_order):
            frontier_node_idx = frontier_nodes[topic_idx]
            topic_mass = node_masses[frontier_node_idx]
            stats = topic_stats.get(topic_idx, {})
            
            # Find works for this topic to get count
            local_indices = np.where(point_to_cluster == topic_idx)[0]
            work_count = len(local_indices)
            
            f.write(f"## Topic {topic_idx}\n\n")
            f.write(f"**Total Relevance (Mass):** {topic_mass:.4f}\n\n")
            f.write(f"**Number of Works:** {work_count}\n\n")

            # If there are no works, skip the rest of this topic section
            if work_count == 0:
                f.write("_No works assigned to this topic._\n\n")
                f.write("---\n\n")
                continue
            
            # Per-topic mass distribution stats
            f.write("### Relevance Distribution\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| Entropy | {stats.get('entropy', 0):.4f} ({100*stats.get('entropy_ratio', 0):.1f}% of max) |\n")
            f.write(f"| Gini Coefficient | {stats.get('gini', 0):.4f} |\n")
            f.write("\n")
            
            # Keywords
            if topic_keywords and topic_idx in topic_keywords:
                f.write("### Top Keywords\n\n")
                f.write("| Keyword | Score |\n")
                f.write("| :--- | :--- |\n")
                for kw, score in topic_keywords[topic_idx][:NUM_KEYWORDS_IN_SUMMARY]:
                    f.write(f"| {kw} | {score:.4f} |\n")
                f.write("\n")
            
            # Top Works
            f.write("### Top Works\n\n")
                
            # Get masses and sort
            topic_local_masses = point_masses[local_indices]
            sorted_indices_local = np.argsort(topic_local_masses)[::-1]
            top_local_indices = local_indices[sorted_indices_local[:NUM_WORKS_IN_SUMMARY]]
            
            # Map to cloud indices
            if filtered_indices is not None:
                cloud_indices = filtered_indices[top_local_indices]
            else:
                cloud_indices = top_local_indices
                
            for i, cloud_idx in enumerate(cloud_indices):
                work_mass = point_masses[top_local_indices[i]]
                title = titles[cloud_idx]
                abstract = abstracts[cloud_idx]
                wid = work_ids[cloud_idx]
                pub_date = publication_dates[cloud_idx]
                
                # Clean abstract if None
                if abstract is None:
                    abstract = "_No abstract available._"
                else:
                    # Truncate abstract if too long? Maybe not for summary.
                    pass
                
                f.write(f"#### {i+1}. {title}\n\n")
                f.write(f"- **Relevance:** {work_mass:.6f}\n")
                f.write(f"- **Publication Date:** {pub_date}\n")
                f.write(f"- **Work ID:** [{wid}]({wid})\n\n")
                f.write(f"> {abstract}\n\n")
            
            f.write("---\n\n")
            
    LOGGER.info(f"Saved summary to {summary_path}")
