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
TOP_K_SHARES_PER_TOPIC = [1, 10, 100]
TOP_K_CENTER_FOR_SIM = 1000


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
        
        # Hubness shares within topic
        sorted_desc = np.sort(masses)[::-1]
        top_k_shares = {}
        for k in TOP_K_SHARES_PER_TOPIC:
            k_eff = min(int(k), len(sorted_desc))
            if k_eff <= 0 or total_mass <= 0:
                top_k_shares[f"top_{k}"] = 0.0
            else:
                top_k_shares[f"top_{k}"] = float(sorted_desc[:k_eff].sum() / total_mass)

        topic_stats[topic_idx] = {
            "total_mass": total_mass,
            "entropy": float(entropy),
            "entropy_ratio": float(entropy_ratio),
            "gini": float(gini),
            "top_k_shares": top_k_shares,
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
        top_k = mass_stats.get("top_k_shares", {})
        top_pct = mass_stats.get("top_pct_shares", {})
        if top_k:
            top_k_str = ", ".join([f"{k}: {v:.2%}" for k, v in top_k.items()])
            lines.append(f"| Top-k Mass Share | {top_k_str} |")
        if top_pct:
            top_pct_str = ", ".join([f"{k}: {v:.2%}" for k, v in top_pct.items()])
            lines.append(f"| Top-pct Mass Share | {top_pct_str} |")
    
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
    topic_similarities: Optional[Dict[int, Dict[str, float]]] = None,
    linearization_order: Optional[List[int]] = None,
) -> str:
    """Format a brief topic overview table showing size, relevance stats, keywords, and top work."""
    n_topics = len(frontier_nodes)
    
    # Use provided order or default to ID order
    if linearization_order is None:
        loop_iter = enumerate(range(n_topics))
    else:
        loop_iter = enumerate(linearization_order)

    lines = []
    lines.append("### Topic Overview\n")
    lines.append("| Rank | Size | Relevance | Entropy | Gini | Sim→Target | Sim→Source | Top Keywords | Top Work |")
    lines.append("| :---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
    
    for rank, topic_idx in loop_iter:
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

        sim_target = "-"
        sim_source = "-"
        if topic_similarities:
            sim_info = topic_similarities.get(topic_idx, {})
            if "sim_target" in sim_info:
                sim_target = f"{sim_info['sim_target']:.3f}"
            if "sim_source" in sim_info:
                sim_source = f"{sim_info['sim_source']:.3f}"
        
        lines.append(
            f"| {rank + 1} | {size:,} | {total_mass:.4f} | {100*entropy_ratio:.1f}% | {gini:.4f} | "
            f"{sim_target} | {sim_source} | {keywords_str} | {top_title} |"
        )
    
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
        
        # Compute per-topic centroid similarities to source/target embeddings
        topic_similarities: Optional[Dict[int, Dict[str, float]]] = None
        if outputs.metadata:
            target_ids = outputs.metadata.get("target_work_ids", []) or []
            source_ids = outputs.metadata.get("source_work_ids", []) or []
            id_to_idx = {wid: i for i, wid in enumerate(work_ids)}

            target_indices = [id_to_idx.get(wid) for wid in target_ids]
            target_indices = [i for i in target_indices if i is not None]
            source_indices = [id_to_idx.get(wid) for wid in source_ids]
            source_indices = [i for i in source_indices if i is not None]

            if target_indices or source_indices:
                embeddings = cloud.embeddings()
            else:
                embeddings = None

            target_vecs = embeddings[target_indices] if (embeddings is not None and target_indices) else None
            source_vecs = embeddings[source_indices] if (embeddings is not None and source_indices) else None

            if target_vecs is not None or source_vecs is not None:
                topic_similarities = {}
                for topic_idx in range(n_topics):
                    local_indices = np.where(point_to_cluster == topic_idx)[0]
                    if local_indices.size == 0:
                        continue

                    # Use top-N by mass to estimate centroid
                    local_masses = point_masses[local_indices]
                    order = np.argsort(local_masses)[::-1]
                    top_local = local_indices[order[:TOP_K_CENTER_FOR_SIM]]

                    if filtered_indices is not None:
                        cloud_indices = filtered_indices[top_local]
                    else:
                        cloud_indices = top_local

                    topic_vecs = embeddings[cloud_indices]
                    centroid = topic_vecs.mean(axis=0)
                    denom = np.linalg.norm(centroid) + 1e-12
                    centroid = centroid / denom

                    sim_info: Dict[str, float] = {}
                    if target_vecs is not None:
                        t_norm = target_vecs / (np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-12)
                        sim_info["sim_target"] = float(np.max(t_norm @ centroid))
                    if source_vecs is not None:
                        s_norm = source_vecs / (np.linalg.norm(source_vecs, axis=1, keepdims=True) + 1e-12)
                        sim_info["sim_source"] = float(np.max(s_norm @ centroid))
                    topic_similarities[topic_idx] = sim_info

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
            topic_similarities=topic_similarities,
            linearization_order=linearization_order,
        ))
        
        for rank, topic_idx in enumerate(linearization_order):
            frontier_node_idx = frontier_nodes[topic_idx]
            topic_mass = node_masses[frontier_node_idx]
            stats = topic_stats.get(topic_idx, {})
            
            # Find works for this topic to get count
            local_indices = np.where(point_to_cluster == topic_idx)[0]
            work_count = len(local_indices)
            
            f.write(f"## Topic {rank + 1}\n\n")
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
            top_k_shares = stats.get("top_k_shares", {})
            if top_k_shares:
                top_k_str = ", ".join([f"{k}: {v:.2%}" for k, v in top_k_shares.items()])
                f.write(f"| Top-k Mass Share | {top_k_str} |\n")
            f.write("\n")

            if topic_similarities and topic_idx in topic_similarities:
                sim_info = topic_similarities[topic_idx]
                f.write("### Seed Similarity\n\n")
                f.write("| Metric | Value |\n")
                f.write("| :--- | :--- |\n")
                if "sim_target" in sim_info:
                    f.write(f"| Centroid → Target (max cos) | {sim_info['sim_target']:.3f} |\n")
                if "sim_source" in sim_info:
                    f.write(f"| Centroid → Source (max cos) | {sim_info['sim_source']:.3f} |\n")
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
