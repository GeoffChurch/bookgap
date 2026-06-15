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


def test_plant_recovers_known_region(tmp_path):
    import numpy as np
    from artifacts_scripts.curriculum.gap_recover.core import plant_recover as pr
    from artifacts_scripts.curriculum.gap_recover.core.kernels import knn_dists, FgNeighborTable
    _write_synthetic(tmp_path, n_concepts=8, thms_per=40, seed=1)
    data = pr.Data(tmp_path)
    r_B = knn_dists(data.Zb, data.Zb, min(pr.K_NN, data.B_idx.size), exclude_self=True)[:, -1].astype(np.float32)
    tbl = FgNeighborTable(data.Zb, data.Zf, m=128, k=pr.K_NN)
    c0 = int(data.B_idx[0])
    cluster = pr.concept_theorem_cluster(data, c0)
    region, dmask = pr.region_for_deletion(data, tbl, cluster, c0)
    assert region.size >= 1
    scores = pr.score_all(data, tbl, r_B, dmask)
    rest = np.setdiff1d(np.arange(data.B_idx.size), region)
    auc = pr.mann_whitney_auc(scores["M_emb"][region], scores["M_emb"][rest])
    assert auc >= 0.8   # clean synthetic clusters -> strong embedding recovery


def test_temporal_calibration_is_chance(tmp_path):
    import argparse
    import numpy as np
    from artifacts_scripts.curriculum.gap_recover.core import temporal_holdout as th
    # Large n so the single-draw calibration AUC concentrates near chance.
    _write_synthetic(tmp_path, n_concepts=100, thms_per=8, seed=2)
    d = np.load(tmp_path / "data.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    is_B = d["is_B"]
    N = len(names)
    # 5 tags (0..4); evaluate at T="v1" (Ti=1). Construct births so that HALF the
    # present concepts get future (in-window) arrivals and half get none -> a genuinely
    # MIXED arrival label set, against which a RANDOM ranking must score ~chance.
    # (A uniform-random birth assignment makes every dense concept attract an arrival,
    # collapsing the negative class and yielding a degenerate nan AUC.)
    birth = np.zeros(N, dtype=np.int16)
    for i, nm in enumerate(names):
        grp = int(nm.split(".")[0][1:])      # "M7.Concept7"/"M7.op7_3" -> 7
        if is_B[i]:
            birth[i] = 0                      # all concepts present from the start
        else:
            birth[i] = 3 if (grp % 2 == 0) else 0   # even groups' ops arrive AFTER T
    np.savez(tmp_path / "births.npz", birth=birth,
             tags=np.array([f"v{i}" for i in range(5)], object))
    data = th.Data(tmp_path).load_births(tmp_path)
    args = argparse.Namespace(topfrac=0.1, min_deg=0, nb=5, boot=200, seed=0)
    out = th.run_for_T(data, "v1", args, np.random.default_rng(0))
    cal = out["citation_all"]["calibration"]["auc"]
    assert not np.isnan(cal)                  # mixed labels -> a finite calibration AUC
    assert abs(cal - 0.5) < 0.15              # random ranking does NOT predict arrivals
