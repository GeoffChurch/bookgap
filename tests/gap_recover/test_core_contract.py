import numpy as np
import scipy.sparse as sp
from artifacts_scripts.curriculum.gap_recover.core import kernels
from artifacts_scripts.curriculum.gap_recover.core.data import Data


def test_fg_neighbor_table_masked_rescore():
    rng = np.random.default_rng(0)
    fg = rng.standard_normal((40, 8)).astype(np.float32)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True)
    cand = rng.standard_normal((10, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    tbl = kernels.FgNeighborTable(cand, fg, m=20, k=5)
    intact = tbl.r_F_k(None)
    dmask = np.zeros(40, dtype=bool); dmask[:10] = True
    after = tbl.r_F_k(dmask)
    assert np.all(after >= intact - 1e-5)   # deleting foreground can only push the k-th distance out
    assert kernels.mann_whitney_auc(np.array([3., 4., 5.]), np.array([0., 1., 2.])) == 1.0


def _write_synthetic(tmp_path, n_concepts=6, thms_per=30, dim=16, seed=0):
    """data.npz where each B-concept sits in its own embedding cluster surrounded by a
    cluster of F-operations that cite it. Deleting a concept's operations orphans that
    concept's region and nobody else's."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_concepts, dim)) * 5.0
    names, kinds, mods, embs, isF, isB, concept_pref = [], [], [], [], [], [], []
    edges_src, edges_dst = [], []
    node_id = {}
    def add(name, kind, mod, vec, f, b, pref):
        node_id[name] = len(names)
        names.append(name); kinds.append(kind); mods.append(mod)
        embs.append(vec); isF.append(f); isB.append(b); concept_pref.append(pref)
        return node_id[name]
    for c in range(n_concepts):
        cid = add(f"M{c}.Concept{c}", "class", f"M{c}", centers[c], False, True, True)
        for t in range(thms_per):
            v = centers[c] + rng.standard_normal(dim) * 0.3
            tid = add(f"M{c}.op{c}_{t}", "function", f"M{c}", v, True, False, False)
            edges_src.append(tid); edges_dst.append(cid)   # op cites concept (operational user)
    emb = np.array(embs, dtype=np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    N = len(names)
    struct = np.zeros(N, dtype=np.int64); oper = np.zeros(N, dtype=np.int64)
    isB_arr = np.array(isB); isF_arr = np.array(isF)
    src = np.array(edges_src); dst = np.array(edges_dst)
    np.add.at(struct, dst, isB_arr[src].astype(np.int64))
    np.add.at(oper, dst, isF_arr[src].astype(np.int64))
    sub = sp.csr_matrix((np.ones(src.size, np.int8), (src, dst)), shape=(N, N)); sub.data[:] = 1
    np.savez(tmp_path / "data.npz", emb=emb, is_F=isF_arr, is_B=isB_arr,
             kinds=np.array(kinds, object), mods=np.array(mods, object),
             names=np.array(names, object), structural_users=struct,
             operational_users=oper, concept_pref=np.array(concept_pref))
    sp.save_npz(tmp_path / "subgraph.npz", sub)
    return tmp_path


def test_data_loads_contract(tmp_path):
    _write_synthetic(tmp_path)
    data = Data(tmp_path)
    assert data.is_B.sum() == 6 and data.is_F.sum() == 180
    assert set(data.operational_users[data.B_idx].tolist()) == {30}
    assert set(data.structural_users[data.B_idx].tolist()) == {0}
