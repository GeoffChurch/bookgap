"""Graph potential solver for curriculum building via circuit theory.

This module provides a generalized solver for computing potentials and currents
on directed graphs, treating citation networks as electrical circuits with diodes.

The solver supports:
- Fixed potentials (Dirichlet boundary conditions): "I want to reach this" (1V) or "I know this" (0V)
- Current injection/suction (Neumann conditions): "Emphasize this" (+I) or "Avoid this" (-I)
- Global leakage (virtual ground): Controlled by beta = log(alpha/(1-alpha))
- Batched computation: Solve for multiple boundary condition sets simultaneously

Physical Interpretation:
    Potential (phi): "Distance" or "Progress" metric (0 = Known, 1 = Goal)
    Current (I): "Traffic" or "Relevance" metric (highlights bottlenecks/bridges)
    Leakage: Controls decay from local (high leakage) to global (low leakage)

Use Cases:
    1. Curriculum Generation (Potential Source + Potential Sink):
       Set Goal=1V, Known=0V → Get prerequisite gradient
    
    2. Prerequisite Discovery (Potential Source + Leakage):
       Set Goal=1V, high leakage → Find immediate prerequisites only
    
    3. Steered Learning (Potential + Current mix):
       Set Goal=1V, Known=0V, Emphasize=+I, Avoid=-I → Biased path

References:
    See chat.md for the full theoretical development.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union
import time

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg._interface import LinearOperator
from tqdm import tqdm

from bookgap.util import LOGGER, TIMER


class _JacobiInversePreconditioner(LinearOperator):
    def __init__(self, inv_diag: NDArray[np.floating]):
        super().__init__(dtype=np.float64, shape=(inv_diag.shape[0], inv_diag.shape[0]))
        self._inv_diag = inv_diag

    def _matvec(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._inv_diag * x

    def _rmatvec(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._inv_diag * x


@TIMER.time(formatters=dict(
    adj=lambda a: f"<sparse {a.shape}, nnz={a.nnz}>",
    fixed_potentials=lambda f: f"<{len(f) if f else 0} nodes>",
    current_sources=lambda c: f"<shape={c.shape}>" if c is not None else "<none>",
))
def compute_graph_potential(
    adj: sp.csr_matrix,
    fixed_potentials: Optional[Mapping[int, Union[float, NDArray[np.floating]]]] = None,
    current_sources: Optional[NDArray[np.floating]] = None,
    leakage: float = 1e-6,
    beta: Optional[float] = None,
    normalize: bool = True,
    gmres_tol: float = 1e-6,
    gmres_maxiter: Optional[int] = None,
    gmres_preconditioner: Optional[str] = None,
    gmres_log_every: Optional[int] = None,
) -> NDArray[np.floating]:
    """Solve for potentials on a directed graph with boundary conditions.
    
    Solves the circuit equation:
        (D_out + G_leak * I - A) @ phi = current_sources
    
    subject to fixed potential constraints (Dirichlet boundary conditions).
    
    When normalize=True, uses column-normalized adjacency (each column sums to 1),
    which makes this equivalent to PageRank:
        - Set leakage = (1 - alpha) / alpha where alpha is the damping factor
        - For alpha=0.85: leakage ≈ 0.176
        - current_sources should be the teleport/seed vector
    
    The leakage parameter can be specified either directly or via beta:
        leakage = exp(-beta)
        beta = log(1/leakage)
    
    Higher beta → lower leakage → more global flow (superconductor)
    Lower beta → higher leakage → more local flow (resistor network)
    
    Args:
        adj: Sparse adjacency matrix (A[i,j] = 1 means i cites j).
             NOTE: This function internally transposes the matrix to ensure flow
             goes from Citer to Cited (downstream), mimicking PageRank.
             This means potentials represent "relevance" or "mass" flowing from
             source nodes to foundational works.
        fixed_potentials: Dictionary mapping node indices to fixed voltages.
                         Example: {goal_idx: 1.0, known_idx: 0.0}
        current_sources: Optional current injection vector(s), shape (n,) or (n, n_batches).
                        Positive = injection, Negative = suction.
                        If None, defaults to zero (pure potential problem).
        leakage: Global ground conductance (G_leak). Ensures matrix invertibility.
                Small leakage (~1e-6) gives lossless flow. Higher leakage gives
                exponential decay (PageRank-like).
        beta: Alternative parameterization: leakage = exp(-beta).
             If provided, overrides the leakage parameter.
             beta=6 → leakage≈0.002 (copper wire)
             beta=10 → leakage≈0.00005 (superconductor)
        normalize: If True, use column-normalized adjacency matrix (PageRank mode).
                  Each column is divided by its sum, making the matrix column-stochastic.
                  This is equivalent to PageRank with appropriate leakage setting.
        gmres_tol: Convergence tolerance for GMRES solver.
        gmres_maxiter: Maximum GMRES iterations (None = auto).
        gmres_preconditioner: Optional preconditioner. Supported: "jacobi".
            Note: for SciPy GMRES, this is the inverse-preconditioner M ≈ A^{-1}.
        gmres_log_every: If provided, log GMRES progress every N inner iterations.
        
    Returns:
        Potential values at all nodes, shape (n,) or (n, n_batches) matching current_sources.
    """
    # Transpose adjacency to ensure flow goes Citer -> Cited (downstream)
    # Original circuit physics (hitting probability) flows Cited -> Citer.
    # By transposing, we make potential flow downstream like PageRank mass.
    adj = sp.csr_matrix(adj.T)
    n = adj.shape[0]
    
    # Handle beta parameterization
    if beta is not None:
        leakage = np.exp(-beta)
        LOGGER.debug(f"Using beta={beta:.2f} → leakage={leakage:.2e}")
    
    # Handle batching
    fixed_potentials = dict(fixed_potentials) if fixed_potentials is not None else {}

    inferred_batches: Optional[int] = None
    if current_sources is None and fixed_potentials:
        # Infer n_batches from any vector-valued fixed potential.
        for v in fixed_potentials.values():
            if isinstance(v, np.ndarray):
                if v.ndim != 1:
                    raise ValueError(
                        "fixed_potentials ndarray values must be 1D vectors of length n_batches"
                    )
                inferred_batches = int(v.shape[0])
                break

    auto_squeeze = False
    if current_sources is None:
        n_batches = inferred_batches if inferred_batches is not None else 1
        current_sources = np.zeros((n, n_batches), dtype=np.float64)
        auto_squeeze = (n_batches == 1 and inferred_batches is None)

    # Help type-checkers: current_sources is now always an ndarray
    assert current_sources is not None

    squeeze_output = False
    if current_sources.ndim == 1:
        current_sources = current_sources.reshape(-1, 1)
        squeeze_output = True
    elif current_sources.ndim != 2:
        raise ValueError("current_sources must be 1D or 2D array")

    if auto_squeeze and current_sources.shape[1] == 1:
        squeeze_output = True

    n_batches = int(current_sources.shape[1])
    
    # Build Laplacian: L = (D_out + G_leak * I - A)
    # D_out[i,i] = sum of outgoing edges from i
    # This represents: outflow - inflow + leakage
    out_degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    if normalize:
        # Column-normalize adjacency matrix (PageRank mode)
        # Each column is divided by its sum, making matrix column-stochastic.
        # This is equivalent to row-normalizing the transposed matrix.
        # For nodes with no outgoing edges (dangling nodes), we leave them as-is
        # (they contribute 0 to the column sums).
        col_sums = np.asarray(adj.sum(axis=0)).flatten()
        # Avoid division by zero for dangling nodes
        col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)
        inv_col_sums = 1.0 / col_sums_safe
        # Create diagonal scaling matrix
        D_inv = sp.diags(inv_col_sums, format='csr')
        adj = sp.csr_matrix(adj @ D_inv)
        # In normalized mode, each node effectively has out-degree 1 (or 0 for dangling)
        # The Laplacian becomes: (I + G_leak * I - A_normalized) = ((1 + G_leak) * I - A_norm)
        L = (sp.diags(np.ones(n) * (1.0 + leakage), format='csr') - adj).tocsr()
        LOGGER.debug(f"Using column-normalized adjacency (PageRank mode)")
    else:
        D_out = sp.diags(out_degrees + leakage, format='csr')
        L = (D_out - adj).tocsr()
    
    # Handle fixed potentials via boundary conditions
    # For each fixed node, replace its equation with: phi[i] = V_i
    fixed_nodes = list(fixed_potentials.keys())
    free_nodes = [i for i in range(n) if i not in fixed_potentials]
    n_fixed = len(fixed_nodes)
    n_free = len(free_nodes)
    
    LOGGER.debug(
        f"Circuit: n={n}, n_batches={n_batches}, n_fixed={n_fixed}, "
        f"n_free={n_free}, leakage={leakage:.2e}"
    )
    
    if n_free == 0:
        # All nodes are fixed - just return the fixed values
        phi = np.zeros((n, n_batches), dtype=np.float64)
        for node_idx, voltage in fixed_potentials.items():
            phi[node_idx, :] = voltage
        return phi.squeeze(axis=1) if squeeze_output else phi
    
    # Build reduced system for free nodes only
    # The equation for free nodes becomes:
    #   L_ff @ phi_f = b - L_fc @ phi_c
    # where phi_c are the fixed (clamped) potentials
    
    free_mask = np.array(free_nodes)
    fixed_mask = np.array(fixed_nodes)
    
    # Extract submatrices
    L_ff = L[np.ix_(free_mask, free_mask)]  # Free-Free coupling
    L_fc = L[np.ix_(free_mask, fixed_mask)]  # Free-Clamped coupling
    
    # Build RHS for each batch
    rhs = current_sources[free_mask, :]  # Start with current sources at free nodes
    
    # Subtract contribution from fixed potentials
    if n_fixed > 0:
        phi_fixed = np.zeros((n_fixed, n_batches), dtype=np.float64)
        for row_idx, node_idx in enumerate(fixed_nodes):
            v = fixed_potentials[node_idx]
            if isinstance(v, np.ndarray):
                if v.ndim != 1:
                    raise ValueError(
                        "fixed_potentials ndarray values must be 1D vectors of length n_batches"
                    )
                if int(v.shape[0]) != n_batches:
                    raise ValueError(
                        f"fixed_potentials[{node_idx}] has length {int(v.shape[0])}, expected n_batches={n_batches}"
                    )
                phi_fixed[row_idx, :] = v
            else:
                phi_fixed[row_idx, :] = float(v)

        # L_fc @ phi_fixed yields (n_free, n_batches)
        fixed_contribution = L_fc @ phi_fixed
        rhs = rhs - np.asarray(fixed_contribution)
    
    # Solve for free nodes using GMRES
    phi_free = np.zeros((n_free, n_batches), dtype=np.float64)

    M: Optional[LinearOperator] = None
    if gmres_preconditioner is not None:
        if gmres_preconditioner != "jacobi":
            raise ValueError(
                f"Unsupported gmres_preconditioner={gmres_preconditioner!r}; supported: 'jacobi'"
            )
        diag = L_ff.diagonal().astype(np.float64, copy=False)
        inv_diag = np.zeros_like(diag)
        nz = diag != 0
        inv_diag[nz] = 1.0 / diag[nz]

        M = _JacobiInversePreconditioner(inv_diag)

    def _make_callback(batch_idx: Optional[int]):
        if gmres_log_every is None or gmres_log_every <= 0:
            return None, None

        log_every = int(gmres_log_every)

        start_t = time.perf_counter()
        it_ctr = 0

        def _cb(pr_norm: float) -> None:
            nonlocal it_ctr
            it_ctr += 1
            if it_ctr % log_every == 0:
                elapsed = time.perf_counter() - start_t
                if batch_idx is None:
                    LOGGER.info(
                        f"GMRES: inner_iter={it_ctr}, pr_norm={pr_norm:.2e}, elapsed={elapsed:.1f}s"
                    )
                else:
                    LOGGER.info(
                        f"GMRES(batch={batch_idx}): inner_iter={it_ctr}, pr_norm={pr_norm:.2e}, elapsed={elapsed:.1f}s"
                    )

        return _cb, "pr_norm"
    
    if n_batches == 1:
        # Single solve
        LOGGER.debug("Solving circuit equation with GMRES...")
        callback, callback_type = _make_callback(batch_idx=None)
        phi_free_vec, info = gmres(
            L_ff,
            rhs.flatten(),
            rtol=gmres_tol,
            maxiter=gmres_maxiter,
            atol=0,  # Use relative tolerance only
            M=M,
            callback=callback,
            callback_type=callback_type,
        )
        if info != 0:
            LOGGER.warning(f"GMRES did not converge (info={info})")
        phi_free[:, 0] = phi_free_vec
    else:
        # Batched solve
        LOGGER.debug(f"Solving {n_batches} circuit equations with GMRES...")
        for batch_idx in tqdm(
            range(n_batches),
            desc=f"Circuit solver (n_free={n_free})",
            leave=False,
        ):
            callback, callback_type = _make_callback(batch_idx=batch_idx)
            phi_free_vec, info = gmres(
                L_ff,
                rhs[:, batch_idx],
                rtol=gmres_tol,
                maxiter=gmres_maxiter,
                atol=0,
                M=M,
                callback=callback,
                callback_type=callback_type,
            )
            if info != 0:
                LOGGER.warning(f"GMRES batch {batch_idx} did not converge (info={info})")
            phi_free[:, batch_idx] = phi_free_vec
    
    # Reconstruct full solution
    phi = np.zeros((n, n_batches), dtype=np.float64)
    phi[free_mask, :] = phi_free
    for node_idx, voltage in fixed_potentials.items():
        if isinstance(voltage, np.ndarray):
            phi[node_idx, :] = voltage
        else:
            phi[node_idx, :] = float(voltage)
    
    if squeeze_output:
        return phi.squeeze(axis=1)
    return phi


def compute_current_flow(
    adj: sp.csr_matrix,
    potentials: NDArray[np.floating],
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Compute the current flow magnitude through each node from potentials.
    
    For a resistor network, the current flowing through node i is the sum of
    absolute edge currents incident on that node. This measures how much of
    the total source→target current passes through each node.
    
    For normalized adjacency (PageRank mode), edge currents are:
        I_{i→j} = A_norm[i,j] * (φ_i - φ_j)
    
    The node current is half the sum of absolute incident edge currents:
        I_node[i] = 0.5 * Σ_j |I_{i→j}| + |I_{j→i}|
    
    Args:
        adj: Adjacency matrix (will be normalized if normalize=True)
        potentials: Node potentials from compute_graph_potential
        normalize: Whether to column-normalize adjacency (should match solver setting)
        
    Returns:
        Array of current flow magnitudes, one per node. Higher values indicate
        nodes that are more "central" to the source→target path.
    """
    n = adj.shape[0]
    
    # Normalize adjacency if requested (should match solver setting)
    if normalize:
        out_degree = np.array(adj.sum(axis=0)).flatten()
        out_degree = np.where(out_degree > 0, out_degree, 1.0)
        D_inv = sp.diags(1.0 / out_degree)
        adj_norm = adj @ D_inv
    else:
        adj_norm = adj
    
    # Convert to COO for efficient edge iteration
    adj_coo = adj_norm.tocoo()
    
    # Compute edge currents: I_{i←j} = A[i,j] * (φ_j - φ_i)
    # (current flows from high to low potential)
    phi_diff = potentials[adj_coo.col] - potentials[adj_coo.row]
    edge_currents = adj_coo.data * phi_diff
    
    # Sum absolute edge currents at each node
    # Each edge contributes to both endpoints
    node_currents = np.zeros(n, dtype=np.float64)
    np.add.at(node_currents, adj_coo.row, np.abs(edge_currents))
    np.add.at(node_currents, adj_coo.col, np.abs(edge_currents))
    
    # Divide by 2 since each edge is counted twice
    node_currents *= 0.5
    
    return node_currents


def build_batched_potential_boundary_conditions(
    n_samples: int,
    num_topics: int,
    leaf_to_cluster: NDArray[np.int64],
) -> Tuple[Dict[int, NDArray[np.floating]], Dict[int, NDArray[np.floating]]]:
    """Build batched boundary conditions for inter-topic dependency computation.
    
    For each topic i (column in the batch):
    - Topic i's leaves are set to 1V (Goal/Target)
    - All other topics' leaves are set to 0V (Known/Ground)
    - Unassigned nodes (cluster=-1) remain free (no constraint)
    
    Returns two dictionaries mapping node indices to voltage arrays:
    - sources: Nodes to set to 1V (shape: (num_topics,), one per batch)
    - sinks: Nodes to set to 0V (shape: (num_topics,), one per batch)
    
    These can be combined into a single fixed_potentials dict for the solver.
    
    Args:
        n_samples: Total number of nodes in the graph.
        num_topics: Number of frontier topics.
        leaf_to_cluster: Array mapping node indices to cluster IDs.
            Values of -1 indicate unassigned nodes (not in any topic).
        
    Returns:
        Tuple of (sources_dict, sinks_dict) where each maps node_idx to voltage array.
    """
    # For each node, build a vector of voltages across batches
    # node_voltages[node_idx] = [V_batch0, V_batch1, ..., V_batch_{num_topics-1}]
    
    sources_dict: Dict[int, NDArray[np.floating]] = {}
    sinks_dict: Dict[int, NDArray[np.floating]] = {}
    
    for node_idx in range(n_samples):
        cluster_id = leaf_to_cluster[node_idx]
        
        if cluster_id == -1:
            # Unassigned node - no constraint, skip
            continue
        
        # Build voltage vector for this node across all batches
        voltages = np.zeros(num_topics, dtype=np.float64)
        
        # In batch i (solving for topic i dependencies):
        # - If node belongs to topic i: set to 1V (source)
        # - If node belongs to other topic: set to 0V (sink)
        voltages[:] = 0.0  # Default: sink (0V)
        voltages[cluster_id] = 1.0  # Source (1V) when batch == node's cluster
        
        # Categorize based on primary role (majority vote across batches)
        n_source = (voltages == 1.0).sum()
        n_sink = (voltages == 0.0).sum()
        
        if n_source > 0:
            sources_dict[node_idx] = voltages
        if n_sink > 0:
            # Actually, every assigned node will be in both dicts with different values
            # per batch. This is fine - we'll merge them in the solver.
            pass
    
    # Actually, simpler approach: just return a single dict with the voltage pattern
    # Since each node has a specific voltage per batch, we should return:
    # fixed_potentials[node_idx] = array of shape (num_topics,)
    
    # Let me refactor to return a simpler structure
    fixed_potentials_batched: Dict[int, NDArray[np.floating]] = {}
    
    for node_idx in range(n_samples):
        cluster_id = leaf_to_cluster[node_idx]
        
        if cluster_id == -1:
            continue
        
        # Voltage pattern: 1V when batch==cluster_id, 0V otherwise
        voltages = np.zeros(num_topics, dtype=np.float64)
        voltages[cluster_id] = 1.0
        
        fixed_potentials_batched[node_idx] = voltages
    
    # Return dummy dicts for now (keeping interface, but not used)
    return {}, fixed_potentials_batched


def compute_frontier_adjacency_potential(
    adj: sp.csr_matrix,
    frontier_nodes: NDArray[np.int64],
    leaf_to_cluster: NDArray[np.int64],
    leakage: float = 1e-6,
    beta: Optional[float] = None,
    normalize: bool = True,
    gmres_tol: float = 1e-6,
    gmres_maxiter: Optional[int] = None,
    gmres_preconditioner: Optional[str] = None,
    gmres_log_every: Optional[int] = None,
) -> NDArray[np.floating]:
    """Compute inter-topic dependency matrix via batched circuit solving.
    
    For each source topic (batch), we clamp that topic's nodes to 1V and all
    other assigned nodes to 0V. Unassigned nodes (cluster=-1) remain free.
    We then measure the total potential at each target topic's nodes.
    
    Uses batched solving: all source topics are solved simultaneously by passing
    voltage vectors (one value per batch) for each clamped node.
    
    Args:
        adj: Sparse citation adjacency matrix (A[i,j] = 1 means i cites j).
        frontier_nodes: Array of frontier node indices.
        leaf_to_cluster: Array mapping node indices to cluster IDs.
            Values of -1 indicate unassigned nodes (intermediaries only).
        leakage: Global ground conductance.
        beta: Alternative parameterization (overrides leakage if provided).
        normalize: Whether to column-normalize adjacency (PageRank mode).
        gmres_tol: GMRES convergence tolerance.
        gmres_maxiter: Maximum GMRES iterations.
        gmres_preconditioner: Optional preconditioner for GMRES.
        gmres_log_every: If set, log GMRES progress every N iterations.
        
    Returns:
        Frontier adjacency matrix of shape (num_topics, num_topics) where
        entry [i,j] represents how much topic i depends on topic j.
    """
    num_topics = len(frontier_nodes)
    n = adj.shape[0]
    
    LOGGER.info(f"Computing frontier adjacency for {num_topics} topics (batched circuit solver)...")

    # Build batched fixed_potentials: for each assigned node, voltage is 1V
    # when batch index == node's cluster, 0V otherwise.
    # Unassigned nodes (cluster == -1) are free.
    fixed_potentials: Dict[int, NDArray[np.floating]] = {}
    
    for node_idx in range(n):
        cluster_id = leaf_to_cluster[node_idx]
        if cluster_id == -1:
            continue  # Unassigned node - leave free
        
        # Voltage pattern: 1V when batch == cluster_id, 0V otherwise
        voltages = np.zeros(num_topics, dtype=np.float64)
        voltages[cluster_id] = 1.0
        fixed_potentials[node_idx] = voltages
    
    # Solve all batches simultaneously
    # phi has shape (n, num_topics) where phi[:, b] is the solution for batch b
    phi = compute_graph_potential(
        adj,
        fixed_potentials=fixed_potentials,
        current_sources=None,
        leakage=leakage,
        beta=beta,
        normalize=normalize,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_preconditioner=gmres_preconditioner,
        gmres_log_every=gmres_log_every,
    )
    
    # Compute current flowing INTO each node from the network.
    # For voltage-clamped nodes at 0V, current_in = sum of A[j,i] * phi[j]
    # where j are neighbors that cite node i.
    #
    # The solver internally transposes the adjacency, so we need to be careful.
    # In the original adj: A[i,j] = 1 means i cites j.
    # Current into node i from citing papers: I_in[i] = sum_j A[j,i] * (phi[j] - phi[i])
    # For target nodes clamped at 0V: I_in[i] = sum_j A[j,i] * phi[j]
    #
    # Using adj.T: (adj.T)[i,j] = A[j,i] = 1 means j cites i
    # So: I_in = adj.T @ phi (for each batch column)
    
    # Note: The solver transposes internally for PageRank-style flow direction.
    # We need to match that here. Since we're measuring "how much influence 
    # from source reaches target", we compute current INTO target nodes.
    adj_T = sp.csr_matrix(adj.T)  # adj_T[i,j] = 1 means j cites i
    
    # For each batch, compute current into each node
    # current_in[i, b] = sum_j adj_T[i,j] * phi[j, b] - in_degree[i] * phi[i, b]
    # For clamped nodes at 0V, phi[i,b] = 0 so: current_in[i,b] = sum_j adj_T[i,j] * phi[j,b]
    
    # Compute in-degree for non-clamped contribution (we'll use it for all nodes)
    in_degree = np.asarray(adj_T.sum(axis=1)).flatten()
    
    # current_in = adj_T @ phi - diag(in_degree) @ phi
    # For clamped 0V nodes, second term is 0; for source 1V nodes, we don't care
    current_in = np.asarray(adj_T @ phi) - in_degree[:, np.newaxis] * phi
    
    # Compute frontier adjacency: for each (source, target) pair,
    # sum the current flowing into target nodes in the source's batch.
    adj_frontier = np.zeros((num_topics, num_topics), dtype=np.float64)
    
    # Get topic leaves for each topic
    topic_leaves_list = [np.where(leaf_to_cluster == t)[0] for t in range(num_topics)]
    
    for source_idx in range(num_topics):
        for target_idx in range(num_topics):
            if target_idx == source_idx:
                continue
            target_leaves = topic_leaves_list[target_idx]
            if target_leaves.size == 0:
                continue
            # Sum current INTO target nodes from the source batch
            # Positive current = flow into the node (from high to low potential)
            adj_frontier[source_idx, target_idx] = current_in[target_leaves, source_idx].sum()
    
    return adj_frontier
