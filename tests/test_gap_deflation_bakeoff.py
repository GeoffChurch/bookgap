import numpy as np
import pytest


def test_greedy_hard_byte_matches_oracle():
    """greedy_hard must reproduce _greedy_kball_gaps_inline exactly."""
    import numpy as np
    from artifacts_scripts.kball_gap_probe import _greedy_kball_gaps_inline
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(3)
    art = rs.randn(300, 8).astype(np.float32)
    art /= np.linalg.norm(art, axis=1, keepdims=True)
    bk = rs.randn(20, 8).astype(np.float32)
    bk /= np.linalg.norm(bk, axis=1, keepdims=True)

    oracle = _greedy_kball_gaps_inline(
        book_emb=bk, article_emb=art, radius_fraction=0.75, k=1,
        num_gaps=50, min_gap_size=0)
    cands = build_kball_candidates(art, bk, radius_fraction=0.75, k=1)
    got = deflate(cands, "greedy_hard", art, num_gaps=50, min_gap_size=0)

    assert len(got) == len(oracle)
    for (gi_o, gs_o, ids_o), (gi_g, gs_g, ids_g) in zip(oracle, got):
        assert gi_o == gi_g and gs_o == gs_g
        np.testing.assert_array_equal(np.sort(ids_o), np.sort(ids_g))


def test_build_kball_candidates_shape():
    from artifacts_scripts.gap_deflation_bakeoff import build_kball_candidates
    rs = np.random.RandomState(0)
    art = rs.randn(200, 8).astype(np.float32)
    art /= np.linalg.norm(art, axis=1, keepdims=True)
    bk = rs.randn(15, 8).astype(np.float32)
    bk /= np.linalg.norm(bk, axis=1, keepdims=True)
    cands = build_kball_candidates(art, bk, radius_fraction=0.75, k=1)
    assert len(cands) == 200
    c = cands[0]
    assert c.idx == 0
    assert c.members.dtype == np.int64
    assert c.centroid.shape == (8,)
    assert all(0 <= m < 200 for m in c.members)


def test_greedy_geom_differs_and_no_redundancy():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(5)
    art = rs.randn(300, 8).astype(np.float32); art /= np.linalg.norm(art,axis=1,keepdims=True)
    bk = rs.randn(20, 8).astype(np.float32); bk /= np.linalg.norm(bk,axis=1,keepdims=True)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "greedy_geom", art, num_gaps=20, min_gap_size=0,
                   book_emb=bk, radius_fraction=0.75)
    seen = set()
    sizes = []
    for _i, sz, ids in gaps:
        s = set(ids.tolist())
        assert not (s & seen)
        seen |= s
        sizes.append(sz)
    assert sizes == sorted(sizes, reverse=True) or len(sizes) <= 1


def test_submodular_fl_monotone_nonincreasing_gains():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_marginal_trace,
    )
    rs = np.random.RandomState(7)
    art = rs.randn(250, 8).astype(np.float32); art /= np.linalg.norm(art,axis=1,keepdims=True)
    bk = rs.randn(15, 8).astype(np.float32); bk /= np.linalg.norm(bk,axis=1,keepdims=True)
    sigma = np.full(250, 0.5, dtype=np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "submodular_fl", art, num_gaps=15,
                   min_gap_size=0, sigma=sigma)
    gains = _fl_marginal_trace()
    assert all(gains[i] >= gains[i+1] - 1e-6 for i in range(len(gains)-1))
    assert len(gaps) >= 1
    # element-0 is the sequential emission rank
    assert [g[0] for g in gaps] == list(range(len(gaps)))


def test_dpp_picks_diverse_capped():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(9)
    art = rs.randn(300, 8).astype(np.float32); art /= np.linalg.norm(art,axis=1,keepdims=True)
    bk = rs.randn(20, 8).astype(np.float32); bk /= np.linalg.norm(bk,axis=1,keepdims=True)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "dpp", art, num_gaps=10, min_gap_size=0,
                   dpp_pool_cap=200)
    assert [g[0] for g in gaps] == list(range(len(gaps)))   # emission rank
    sets = [set(ids.tolist()) for _i, _s, ids in gaps]
    if len(sets) >= 2:
        js = [len(a & b)/len(a | b) for i,a in enumerate(sets)
              for b in sets[i+1:] if (a|b)]
        assert np.mean(js) < 0.5     # DPP suppresses heavy overlap


def test_kball_seeds_accepts_deflation_kwarg():
    import inspect
    from artifacts_scripts.gap_denoiser_bakeoff import kball_seeds_from_state
    sig = inspect.signature(kball_seeds_from_state)
    assert "deflation" in sig.parameters
    assert sig.parameters["deflation"].default == "greedy_hard"


def test_inter_gap_metrics_basic():
    from artifacts_scripts.gap_deflation_bakeoff import inter_gap_metrics
    survivors = [
        {"refined_member_indices": list(range(0, 50))},
        {"refined_member_indices": list(range(40, 90))},
        {"refined_member_indices": list(range(200, 260))},
    ]
    m = inter_gap_metrics(survivors, top_k=20)
    assert 0.0 <= m["mean_pairwise_jaccard_top20"] <= 1.0
    assert m["unique_coverage_at_20"] == len(set(range(0,90)) | set(range(200,260)))


def test_heldout_mask_deterministic_and_recall():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        heldout_book_split, heldout_recall_spread,
    )
    a = heldout_book_split(100, frac=0.2, seed=42)
    b = heldout_book_split(100, frac=0.2, seed=42)
    np.testing.assert_array_equal(a, b)        # deterministic
    assert len(a) == 20
    assert np.all((a >= 0) & (a < 100)) and len(set(a.tolist())) == 20

    heldout_books_emb = np.zeros((2, 4), dtype=np.float32)
    article_emb = np.zeros((30, 4), dtype=np.float32)
    article_emb[:10] = 0.01                    # 10 articles near book 0
    article_emb[10:] = 10.0                    # articles 10-29 far from book 0
    gaps_members = [list(range(10)), list(range(20, 30))]
    rec, spread = heldout_recall_spread(
        heldout_books_emb[:1], article_emb, gaps_members,
        k_near=10, cover_frac=0.5)
    assert rec == 1.0
    assert spread == 0.0                        # 1 recalled book → entropy 0


def test_deflation_main_guard_is_last_toplevel():
    import ast, pathlib
    src = pathlib.Path("artifacts_scripts/gap_deflation_bakeoff.py").read_text()
    mod = ast.parse(src)
    gi = None
    for i, n in enumerate(mod.body):
        if isinstance(n, ast.If) and "__main__" in ast.dump(n.test):
            gi = i
    assert gi is not None, "no __main__ guard"
    after = [n for n in mod.body[gi+1:]
             if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    assert not after, f"defs after __main__ guard: {[n.name for n in after]}"


def test_deflation_driver_emits_per_arm_rows(tmp_path, monkeypatch):
    import csv
    import artifacts_scripts.gap_deflation_bakeoff as dfl

    def fake_run_arm(scope, deflation, out_dir, **kw):
        return {"deflation": deflation, "scope": scope,
                "mean_pairwise_jaccard_top20": 0.1,
                "unique_coverage_at_20": 500,
                "heldout_book_recall_at_20": 0.4,
                "heldout_spread": 1.2, "runtime_s": 1.0}
    monkeypatch.setattr(dfl, "run_arm", fake_run_arm)
    monkeypatch.setattr("sys.argv", [
        "prog", "--scope", "17/1702/T10181",
        "--arms", "greedy_hard,greedy_geom,submodular_fl,dpp",
        "--out", str(tmp_path)])
    dfl.main()
    rows = list(csv.DictReader(open(tmp_path / "deflation_scorecard.csv")))
    assert {r["deflation"] for r in rows} == {
        "greedy_hard","greedy_geom","submodular_fl","dpp"}
    assert all(r["scope"]=="17/1702/T10181" for r in rows)


def test_softmax_mc_partition_and_emission_rank():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(13)
    d = 16
    cen = np.zeros((3, d), np.float32)
    for i in range(3): cen[i, i] = 8.0
    parts = [cen[i] + 0.1*rs.randn(50, d).astype(np.float32) for i in range(3)]
    parts.append(0.1*rs.randn(80, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = (0.1*rs.randn(8, d).astype(np.float32))           # books near origin bg
    sigma = np.full(len(art), 0.5, dtype=np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "softmax_mc", art, num_gaps=20, min_gap_size=5,
                   sigma=sigma, book_emb=bk)
    assert [g[0] for g in gaps] == list(range(len(gaps)))   # emission rank
    seen = set()
    for _i, sz, ids in gaps:                                # partition: disjoint
        s = set(ids.tolist())
        assert not (s & seen)
        seen |= s
        assert sz == len(ids) >= 5
    assert len(gaps) >= 1


@pytest.mark.parametrize("strategy", [
    "greedy_hard", "greedy_geom", "submodular_fl", "dpp"])
def test_synthetic_implant_recovers_clusters(strategy):
    """3 tight implanted article clusters, NO books nearby → recovered
    as the top-3 gaps with ~no mutual redundancy.

    Fixture notes (test-fixture design; NOT changes to arm logic):
    - d=32 so that axis-aligned cluster centers (10·e_i) are far apart in raw
      Euclidean space (inter-cluster dist ≈ 14.1) without L2 normalisation.
      L2-normalising collapses the geometry: post-normalisation inter-cluster
      cosine dist (~1.1) falls below the book-article angular dist (~1.58),
      making every k-ball span all 3 clusters — DPP then sees all top balls as
      identical and cannot diversify.
    - Books near the origin (scale 0.5) so the 75th-pct book distance from any
      cluster article ≈ 10.8, which is < 14.1 (inter-cluster), giving each seed
      a per-cluster ball of exactly 60 members.
    - sigma=0.5 ≈ intra-cluster RMS distance (≈ 0.56 for σ_noise=0.1, d=32) so
      that the submodular_fl Gaussian kernel produces non-trivial sim values.
      sigma=0.1 (smaller than intra-cluster dist) drives all sim weights to ~0
      and causes submodular_fl to emit 0 gaps.
    """
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(11)
    d = 32
    # Three axis-aligned cluster centers, well-separated in raw Euclidean space
    centers = np.eye(3, d, dtype=np.float32) * 10.0
    parts = []
    for c in centers:
        parts.append(c + 0.1 * rs.randn(60, d).astype(np.float32))  # 3×60 tight
    parts.append(rs.randn(120, d).astype(np.float32) * 0.5)          # diffuse bg near origin
    art = np.vstack(parts).astype(np.float32)
    # No L2 normalisation — preserves the Euclidean cluster geometry
    bk = rs.randn(10, d).astype(np.float32) * 0.5                    # books near origin, NOT normalised
    sigma = np.full(len(art), 0.5, dtype=np.float32)                  # ≈ intra-cluster RMS dist

    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, strategy, art, num_gaps=3, min_gap_size=10,
                   sigma=sigma, book_emb=bk, radius_fraction=0.75)
    assert len(gaps) == 3, f"{strategy}: expected 3 gaps, got {len(gaps)}"
    sets = [set(ids.tolist()) for _i, _s, ids in gaps]
    for i in range(3):
        for j in range(i + 1, 3):
            u = len(sets[i] | sets[j])
            jac = (len(sets[i] & sets[j]) / u) if u else 0.0
            assert jac < 0.2, f"{strategy}: gaps {i},{j} overlap jac={jac:.2f}"


def test_kball_seeds_passes_article_sliced_sigma(monkeypatch):
    """BLOCKER 1 regression: deflate must receive sigma sliced to the
    article space (length A, == state['sigma'][article_global_idx]),
    not the raw global N_total sigma."""
    import numpy as np
    import artifacts_scripts.gap_denoiser_bakeoff as gdb
    captured = {}
    def fake_deflate(cands, strategy, article_emb, *, num_gaps,
                     min_gap_size, sigma=None, **kw):
        captured["sigma"] = None if sigma is None else np.asarray(sigma).copy()
        return []
    monkeypatch.setattr(gdb, "deflate", fake_deflate)
    N = 60
    art_gi = np.arange(0, 40)            # 40 articles at global 0..39
    bk_gi = np.arange(40, 60)            # 20 books at global 40..59
    emb = np.random.RandomState(0).randn(N, 6).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sig = (np.arange(N) * 0.01 + 0.1).astype(np.float32)  # NON-constant
    state = {"emb": emb, "article_global_idx": art_gi,
             "book_global_idx": bk_gi, "sigma": sig, "topic_id": "T"}
    gdb.kball_seeds_from_state(state, deflation="submodular_fl")
    got = captured["sigma"]
    assert got is not None and len(got) == 40
    np.testing.assert_array_equal(got, sig[art_gi])   # article-sliced, not raw


def test_global_to_article_local_maps_and_drops():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _global_to_article_local
    agi = np.array([10, 20, 30, 40])      # global ids of 4 articles
    # gm has global ids; 99 is a non-article id and must be dropped
    out = _global_to_article_local([[20, 40, 99], [10]], agi)
    assert out == [[1, 3], [0]]


def test_fl_bookdemand_recovers_book_anchored_clusters():
    """3 tight article clusters, one book just outside each → fl_bookdemand
    selects 3 gaps each covering a distinct book's neighborhood, disjoint,
    emission-rank element-0.

    Books at 1.2*cen (outside but close to each cluster centroid) so that each
    cluster article's nearest book is the cluster's own book and the k-ball
    radius (~1.5) captures all 60 cluster neighbors. No diffuse background
    noise — background articles would acquire huge k-balls spanning all
    clusters, saturating book-demand coverage in a single pick.
    """
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import build_kball_candidates, deflate
    rs=np.random.RandomState(21); d=32
    cen=np.zeros((3,d),np.float32)
    for i in range(3): cen[i,i]=10.0
    # 3 tight clusters only — no diffuse background (avoids huge cross-cluster balls)
    parts=[cen[i]+0.1*rs.randn(60,d).astype(np.float32) for i in range(3)]
    art=np.vstack(parts).astype(np.float32)
    # books just outside each cluster centroid + 5 far decoy books
    bk=np.vstack([cen*1.2, 40.0+rs.randn(5,d).astype(np.float32)]).astype(np.float32)
    cands=build_kball_candidates(art, bk, 0.75, 1)
    gaps=deflate(cands,"fl_bookdemand",art,num_gaps=3,min_gap_size=10,book_emb=bk)
    assert len(gaps)==3
    assert [g[0] for g in gaps]==[0,1,2]                      # emission rank
    sets=[set(ids.tolist()) for _i,_s,ids in gaps]
    for i in range(3):
        for j in range(i+1,3):
            u=len(sets[i]|sets[j]); assert (len(sets[i]&sets[j])/u if u else 0)<0.2


def test_fl_bookdemand_monotone_and_nobook_edge():
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_trace)
    rs=np.random.RandomState(22)
    a=rs.randn(250,8).astype(np.float32); a/=np.linalg.norm(a,axis=1,keepdims=True)
    b=rs.randn(15,8).astype(np.float32); b/=np.linalg.norm(b,axis=1,keepdims=True)
    c=build_kball_candidates(a,b,0.75,1)
    g=deflate(c,"fl_bookdemand",a,num_gaps=15,min_gap_size=0,book_emb=b)
    gn=_fl_bookdemand_trace()
    assert all(gn[i]>=gn[i+1]-1e-9 for i in range(len(gn)-1))   # submodular
    # no-books edge → []
    assert deflate(c,"fl_bookdemand",a,num_gaps=5,min_gap_size=0,book_emb=None)==[]
    import numpy as _np
    assert deflate(c,"fl_bookdemand",a,num_gaps=5,min_gap_size=0,
                   book_emb=_np.zeros((0,8),_np.float32))==[]


def test_fl_bookdemand_no_saturation_with_background():
    """Regression guard for the saturation defect: a diffuse background ball
    with large members-count must NOT exhaust all book-demand in a single
    greedy pick (which collapses output to ~1 gap).

    Fixture: 3 tight clusters (noise=0.1) at axis-aligned corners + a diffuse
    background (120 articles near origin).  One book just outside each cluster
    + 5 far decoy books.  The background articles have large k-balls that span
    many books' Nb sets → with the unnormalized cov = |intersection|/k_book
    the background ball dominates and only 1 gap is emitted.  Jaccard cov
    = |intersection|/|union| caps this at ≤10/500≈0.02 and lets all 3 tight
    cluster balls win.

    MUST FAIL on pre-fix (unnormalized) cov; MUST PASS after Jaccard fix.
    """
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import build_kball_candidates, deflate
    rs = np.random.RandomState(31); d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))  # diffuse background near origin
    art = np.vstack(parts).astype(np.float32)
    # One book just outside each cluster + 5 far decoy books
    bk = np.vstack([cen * 1.2, 40.0 + rs.randn(5, d).astype(np.float32)]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "fl_bookdemand", art, num_gaps=3, min_gap_size=10, book_emb=bk)
    # Must NOT collapse to ~1 gap — saturation defect would give len(gaps)==1
    assert len(gaps) == 3, (
        f"SATURATION: expected 3 gaps (one per cluster), got {len(gaps)}. "
        "Pre-fix unnormalized cov causes background ball to dominate."
    )
    # Each gap should map to a distinct cluster (check via centroid argmax)
    centers = np.zeros((3, d), np.float32)
    for i in range(3):
        centers[i, i] = 10.0
    cluster_labels = []
    for _rank, _sz, ids in gaps:
        if len(ids) == 0:
            cluster_labels.append(-1)
            continue
        gap_centroid = art[ids].mean(axis=0)
        dists = np.linalg.norm(centers - gap_centroid, axis=1)
        cluster_labels.append(int(np.argmin(dists)))
    assert len(set(cluster_labels)) == 3, (
        f"Gaps did not map to 3 distinct clusters; labels={cluster_labels}"
    )


# ---------------------------------------------------------------------------
# fl_bookdemand_eb — empirical-Bayes shrinkage demand (books ∪ background)
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
# ---------------------------------------------------------------------------

def _eb_cluster_labels(art, ids, centers):
    """argmax-nearest-center label of a gap's article centroid."""
    if len(ids) == 0:
        return -1
    gap_centroid = art[np.asarray(ids)].mean(axis=0)
    dists = np.linalg.norm(centers - gap_centroid, axis=1)
    return int(np.argmin(dists))


def test_fl_bookdemand_eb_book_rich_recovers_clusters():
    """Book-rich limit: 3 dense clusters, dense books per cluster, NO
    diffuse background → EB behaves like fl_bookdemand and recovers 3
    distinct cluster-aligned gaps with low pairwise jaccard."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(21)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    # 3 tight clusters only — no diffuse background (avoids huge cross-cluster balls)
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    art = np.vstack(parts).astype(np.float32)
    # dense books just outside each cluster centroid + 5 far decoy books
    bk = np.vstack([
        cen[0] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[1] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[2] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        40.0 + rs.randn(5, d).astype(np.float32),
    ]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "fl_bookdemand_eb", art, num_gaps=3,
                   min_gap_size=10, book_emb=bk)
    assert len(gaps) == 3, f"expected 3 gaps, got {len(gaps)}"
    assert [g[0] for g in gaps] == [0, 1, 2]      # emission rank
    labels = [_eb_cluster_labels(art, ids, cen) for _i, _s, ids in gaps]
    assert len(set(labels)) == 3, f"labels not distinct: {labels}"
    sets = [set(ids.tolist()) for _i, _s, ids in gaps]
    for i in range(3):
        for j in range(i + 1, 3):
            u = len(sets[i] | sets[j])
            jac = (len(sets[i] & sets[j]) / u) if u else 0.0
            assert jac < 0.2, f"gaps {i},{j} overlap jac={jac:.2f}"


def test_fl_bookdemand_eb_book_sparse_no_collapse():
    """CORE HYPOTHESIS: 3 clusters + 120-article diffuse background +
    only 1 book total → EB does NOT collapse: emits >= 3 gaps mapping to
    3 distinct clusters.  Plain fl_bookdemand collapses to < 3 here
    because the single book's demand is exhausted in one pick."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(31)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))  # diffuse bg
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2]).astype(np.float32)         # ONLY 1 book
    cands = build_kball_candidates(art, bk, 0.75, 1)

    # Contrast: plain fl_bookdemand collapses (< 3) on this exact geometry.
    g_bd = deflate(cands, "fl_bookdemand", art, num_gaps=10,
                   min_gap_size=5, book_emb=bk)
    assert len(g_bd) < 3, (
        f"contrast precondition: fl_bookdemand should collapse <3, got {len(g_bd)}"
    )

    g_eb = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                   min_gap_size=5, book_emb=bk)
    assert len(g_eb) >= 3, f"EB collapsed: got {len(g_eb)} gaps"
    labels = {_eb_cluster_labels(art, ids, cen) for _i, _s, ids in g_eb}
    assert len({l for l in labels if l >= 0}) >= 3, (
        f"EB gaps did not map to 3 distinct clusters; labels={labels}"
    )


def test_fl_bookdemand_eb_shrinkage_monotone():
    """Same geometry, vary only #books (dense vs sparse). Sparse run
    must have strictly larger ratio_bg_book AND shrinkage than dense
    (data-driven empirical-Bayes monotonicity, no tuned λ)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_diag,
    )
    rs = np.random.RandomState(41)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)

    bk_dense = np.vstack([
        cen[0] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
        cen[1] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
        cen[2] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
    ]).astype(np.float32)
    bk_sparse = np.vstack([cen[0] * 1.2]).astype(np.float32)

    c_dense = build_kball_candidates(art, bk_dense, 0.75, 1)
    deflate(c_dense, "fl_bookdemand_eb", art, num_gaps=10,
            min_gap_size=5, book_emb=bk_dense)
    diag_dense = _fl_bookdemand_eb_diag()

    c_sparse = build_kball_candidates(art, bk_sparse, 0.75, 1)
    deflate(c_sparse, "fl_bookdemand_eb", art, num_gaps=10,
            min_gap_size=5, book_emb=bk_sparse)
    diag_sparse = _fl_bookdemand_eb_diag()

    assert diag_sparse["n_books"] < diag_dense["n_books"]
    assert diag_sparse["M"] == diag_dense["M"]   # M depends only on A
    assert diag_sparse["ratio_bg_book"] > diag_dense["ratio_bg_book"]
    assert diag_sparse["shrinkage"] > diag_dense["shrinkage"]


def test_fl_bookdemand_eb_no_book_graceful():
    """Zero books → EB must NOT return [] (that was fl_bookdemand). It
    runs with background customers only and emits >= 3 distinct gaps."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(51)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    # candidates still need *some* book_emb to build k-balls; deflation
    # itself is run with no books (book_emb=None and empty array).
    bk_for_cands = np.vstack([cen * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk_for_cands, 0.75, 1)

    g_none = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                     min_gap_size=5, book_emb=None)
    assert g_none != [], "EB returned [] with no books (must degrade gracefully)"
    assert len(g_none) >= 3, f"expected >=3 gaps with no books, got {len(g_none)}"
    labels = {_eb_cluster_labels(art, ids, cen) for _i, _s, ids in g_none}
    assert len({l for l in labels if l >= 0}) >= 3, f"labels={labels}"

    g_empty = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                      min_gap_size=5,
                      book_emb=np.zeros((0, d), np.float32))
    assert len(g_empty) >= 3, f"empty book_emb: expected >=3, got {len(g_empty)}"
    diag = __import__(
        "artifacts_scripts.gap_deflation_bakeoff", fromlist=["x"]
    )._fl_bookdemand_eb_diag()
    assert diag["n_books"] == 0


def test_fl_bookdemand_eb_monotone_gains():
    """Greedy marginal gains (submodular) must be non-increasing."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_trace,
    )
    rs = np.random.RandomState(61)
    a = rs.randn(250, 8).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rs.randn(15, 8).astype(np.float32)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = build_kball_candidates(a, b, 0.75, 1)
    g = deflate(c, "fl_bookdemand_eb", a, num_gaps=15,
                min_gap_size=0, book_emb=b)
    gn = _fl_bookdemand_eb_trace()
    assert len(gn) == len(g)
    assert all(gn[i] >= gn[i + 1] - 1e-9 for i in range(len(gn) - 1)), gn
    assert [x[0] for x in g] == list(range(len(g)))   # emission rank


def test_fl_bookdemand_eb_determinism():
    """Two runs must be byte-identical (FlatL2 exact, KMeans seeded)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(71)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2, cen[1] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    g1 = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                 min_gap_size=5, book_emb=bk)
    g2 = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                 min_gap_size=5, book_emb=bk)
    assert len(g1) == len(g2) and len(g1) >= 1
    for (i1, s1, m1), (i2, s2, m2) in zip(g1, g2):
        assert i1 == i2 and s1 == s2
        np.testing.assert_array_equal(m1, m2)


def test_eb_target_shrinkage_pins_ratio():
    """n_books-keyed s* variants pin M = clip(round(n_books*s/(1-s)),
    8, min(4000, A)) and shrinkage ≈ s, INDEPENDENT of A.

    Two fixtures with the SAME n_books (40) but different A (different
    background point counts); for fixed s, M must be identical (ratio
    keyed to n_books, not A)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_diag,
    )

    def _make(extra_noise, seed):
        rs = np.random.RandomState(seed)
        d = 24
        cen = np.zeros((4, d), np.float32)
        for i in range(4):
            cen[i, i] = 12.0
        parts = [cen[i] + 0.1 * rs.randn(80, d).astype(np.float32)
                 for i in range(4)]
        if extra_noise:
            parts.append(8.0 * rs.randn(extra_noise, d).astype(np.float32))
        art = np.vstack(parts).astype(np.float32)
        # 40 books: 10 per cluster, just outside each centroid
        bk = np.vstack([cen[i] * 1.2 for i in range(4)
                        for _ in range(10)]).astype(np.float32)
        return art, bk

    for s, nn in ((0.30, "30"), (0.50, "50"), (0.70, "70")):
        name = "fl_bookdemand_eb_s" + nn
        M_seen = []
        for extra in (0, 2000):
            art, bk = _make(extra, seed=11)
            cands = build_kball_candidates(art, bk, 0.75, 1)
            deflate(cands, name, art, num_gaps=20, min_gap_size=3,
                    book_emb=bk)
            diag = _fl_bookdemand_eb_diag()
            A = len(art)
            n_books = diag["n_books"]
            assert n_books == 40, f"{name}: n_books={n_books}"
            M_exp = max(8, min(int(round(n_books * s / (1.0 - s))),
                               4000, A))
            assert diag["M"] == M_exp, (
                f"{name} A={A}: M={diag['M']} != expected {M_exp}")
            assert abs(diag["shrinkage"] - s) < 0.05, (
                f"{name} A={A}: shrinkage={diag['shrinkage']} not ≈ {s}")
            assert diag["mode"] == "nbooks-keyed", diag
            assert diag["s_target"] == s, diag
            M_seen.append(diag["M"])
        # A-invariance: same n_books, two different A → identical M
        assert M_seen[0] == M_seen[1], (
            f"{name}: M not A-invariant: {M_seen}")


def test_eb_target_shrinkage_nbooks0_fallback():
    """s set but n_books == 0 → fall back to A-keyed M, run
    background-only, emit >= 3 gaps (NOT []), diag mode
    "A-keyed-fallback" (a variant silently degraded — distinct from the
    deliberate s-is-None "A-keyed" control)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_diag,
    )
    rs = np.random.RandomState(51)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk_for_cands = np.vstack([cen * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk_for_cands, 0.75, 1)
    A = len(art)

    g = deflate(cands, "fl_bookdemand_eb_s50", art, num_gaps=10,
                min_gap_size=5, book_emb=None)
    assert g != [], "s50 returned [] with no books (must degrade)"
    assert len(g) >= 3, f"expected >=3 gaps, got {len(g)}"
    diag = _fl_bookdemand_eb_diag()
    assert diag["n_books"] == 0, diag
    assert diag["mode"] == "A-keyed-fallback", diag
    M_akeyed = min(int(np.clip(A // 10, 64, 4000)), A)
    assert diag["M"] == M_akeyed, (
        f"fallback M={diag['M']} != A-keyed {M_akeyed}")


def test_eb_diag_mode_three_way_reachable():
    """All THREE diag `mode` values are reachable and correct on one
    fixture:
      (i)   fl_bookdemand_eb_s50 + books>0  → "nbooks-keyed"
      (ii)  fl_bookdemand_eb (None) + books>0 → "A-keyed"
      (iii) fl_bookdemand_eb_s50 + book_emb=None → "A-keyed-fallback"
    The s-given/no-books case must NOT collapse onto the deliberate
    s-is-None A-keyed control — that conflation is the spec bug."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_diag,
    )
    rs = np.random.RandomState(9)
    d = 20
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    art = np.vstack([cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
                     for i in range(3)]).astype(np.float32)
    bk = np.vstack([cen[i] * 1.2 for i in range(3)
                    for _ in range(8)]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)

    deflate(cands, "fl_bookdemand_eb_s50", art, num_gaps=10,
            min_gap_size=3, book_emb=bk)
    assert _fl_bookdemand_eb_diag()["mode"] == "nbooks-keyed", \
        _fl_bookdemand_eb_diag()

    deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
            min_gap_size=3, book_emb=bk)
    assert _fl_bookdemand_eb_diag()["mode"] == "A-keyed", \
        _fl_bookdemand_eb_diag()

    deflate(cands, "fl_bookdemand_eb_s50", art, num_gaps=10,
            min_gap_size=3, book_emb=None)
    assert _fl_bookdemand_eb_diag()["mode"] == "A-keyed-fallback", \
        _fl_bookdemand_eb_diag()


def test_eb_variants_registered_and_distinct():
    """The 3 _sNN names are registered, each deterministic across two
    runs, and at least one differs from fl_bookdemand_eb (None path) on
    a fixture where shrinkage matters."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _STRATEGIES,
    )
    for nn in ("30", "50", "70"):
        assert "fl_bookdemand_eb_s" + nn in _STRATEGIES

    rs = np.random.RandomState(41)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    # moderate book count + background → shrinkage materially differs
    bk = np.vstack([
        cen[0] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[1] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[2] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
    ]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)

    def _run(name):
        return deflate(cands, name, art, num_gaps=20, min_gap_size=5,
                       book_emb=bk)

    def _key(gaps):
        return [(r, s, tuple(sorted(ids.tolist()))) for r, s, ids in gaps]

    for nn in ("30", "50", "70"):
        name = "fl_bookdemand_eb_s" + nn
        assert _key(_run(name)) == _key(_run(name)), f"{name} non-determ"

    base = _key(_run("fl_bookdemand_eb"))
    diffs = [nn for nn in ("30", "50", "70")
             if _key(_run("fl_bookdemand_eb_s" + nn)) != base]
    assert diffs, (
        "no _sNN variant differs from fl_bookdemand_eb — param inert")


def test_fl_bookdemand_eb_none_path_unchanged():
    """Regression guard: fl_bookdemand_eb (None path) must produce the
    SAME gaps as the pre-change code (captured from CURRENT code before
    the n_books-keyed change). The refuted control must not move."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_eb_diag,
    )
    rs = np.random.RandomState(71)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2, cen[1] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    g = deflate(cands, "fl_bookdemand_eb", art, num_gaps=10,
                min_gap_size=5, book_emb=bk)
    # Hard-coded snapshot captured by running the PRE-change code once.
    # NOTE: build_kball_candidates' HNSW index is order-nondeterministic
    # ACROSS processes (the in-array member ORDER of a gap varies run to
    # run — a pre-existing property unrelated to this change), but the
    # SORTED member SET of each gap is byte-stable cross-process and is
    # the correct regression anchor. We pin sha1(sorted-members) plus
    # ngaps / sizes / ranks / diag — together these catch any movement
    # of the None (A-keyed) path.
    import hashlib
    assert len(g) == 3
    assert [x[0] for x in g] == [0, 1, 2]
    assert [x[1] for x in g] == [60, 60, 180]

    def _h(ids):
        return hashlib.sha1(
            repr(tuple(sorted(int(x) for x in ids))).encode()).hexdigest()

    assert _h(g[0][2]) == "7f76ce77d235f93245c8b30d358f1f38722debc7"
    assert _h(g[1][2]) == "8f9b6a4babdf9010a23b901429dc5d2613817a26"
    assert _h(g[2][2]) == "c7784c21193a1a2e7f1d2ec9ebef8a6aecdd014f"
    diag = _fl_bookdemand_eb_diag()
    assert diag["n_books"] == 2
    assert diag["M"] == 64                       # A-keyed value unchanged
    assert diag["mode"] == "A-keyed", diag
    assert diag["s_target"] is None, diag


def test_fl_bookdemand_eb_memory_bounded():
    """The EB peak must be incompatible with a dense (A,K) or (C,K)
    float64 allocation — the exact regression this guard exists for.

    The previous guard (peak < 0.25*8*A*K*d, and peak < 4*peak_bd) could
    NOT catch that regression: the spurious `d` factor inflated the
    ceiling d-fold (a dense (A,K) float64 matrix is only 8*A*K bytes,
    which sails under 0.25*8*A*K*d), and the 4*peak_bd factor was
    calibrated to pass current code, not to fail a concrete dense alloc.

    This rewrite makes the k-ball mass *genuinely* sparse, so a TIGHT
    byte budget can sit strictly below BOTH forbidden dense allocations.
    The test then asserts its own threshold sits below 8*A*K and 8*C*K,
    so by construction any dense (A,K)/(C,K) allocation in the sparse-FL
    family would push peak_eb above the threshold and fail this test.

    Fixture: ~600 tight tiny clusters of 8 points each, spaced far apart
    (sep=80, intra-cluster jitter=0.02), one book per cluster offset by
    0.3 so a p=0.75 k-ball ≈ exactly one cluster (8 members). Total
    candidate member mass S = sum(len(c.members)) is therefore O(A) and
    ≪ A*K and ≪ C*K — an explicit, self-checked sparsity precondition.
    Deterministic (seeded), no network, ~17s.
    """
    import tracemalloc
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )

    # --- small-ball fixture: many tight, well-separated micro-clusters ----
    rs = np.random.RandomState(81)
    d = 16
    n_clusters, pts_per = 600, 8
    centers = (rs.randn(n_clusters, d).astype(np.float32)) * 80.0   # far apart
    parts = [centers[i] + 0.02 * rs.randn(pts_per, d).astype(np.float32)
             for i in range(n_clusters)]
    art = np.vstack(parts).astype(np.float32)            # A = 4800
    # One book per cluster, offset just enough that a p=0.75 ball captures
    # its own (and only its own) tight cluster -> per-candidate ball ~= 8.
    bk = (centers + 0.3).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    A = len(art)
    K = len(cands)

    # Total candidate member mass (the true working-set size of the
    # sparse machinery). Must be ≪ A*K and ≪ C*K for the tight guard to
    # cleanly separate legit-sparse from a dense (A,K)/(C,K) allocation.
    S = sum(int(len(c.members)) for c in cands)
    assert S < 0.05 * A * K, (
        f"sparsity precondition violated: S={S} not << A*K={A * K} "
        f"(S/(A*K)={S / (A * K):.5f}); fixture k-balls are not sparse"
    )

    # Customer count C mirrors _fl_bookdemand_eb exactly:
    #   M = clip(A // k_book, 64, 4000), clamped M <= A;  C = n_books + M.
    k_book = 10
    M = int(np.clip(A // k_book, 64, 4000))
    M = min(M, A)
    n_books = len(bk)
    C = n_books + M

    tracemalloc.start()
    deflate(cands, "fl_bookdemand_eb", art, num_gaps=50,
            min_gap_size=5, book_emb=bk)
    peak_eb = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Expected sparse footprint: S sparse {customer:cov} entries (8 bytes
    # value + a generous per-entry Python dict overhead) + the 1-D dense
    # best_cov / cust_weight vectors of length C. DICT_OVERHEAD is a
    # deliberately generous slack for CPython dict-entry + boxed-int/float
    # bookkeeping; SPARSE_MULT and FIXED_MB absorb the FlatL2 index, the
    # bounded KMeans coreset over (A,d), and interpreter/import noise.
    DICT_OVERHEAD = 80                       # bytes per sparse dict entry
    SPARSE_MULT = 6                          # modest multiplicative slack
    FIXED_MB = 4 * 1024 * 1024               # fixed slack: index + coreset
    sparse_bytes = S * (8 + DICT_OVERHEAD) + C * 8
    threshold = SPARSE_MULT * sparse_bytes + FIXED_MB

    # The two forbidden dense regressions (NO spurious `d` factor): a
    # dense (A,K) float64 matrix = 8*A*K bytes, a dense (C,K) float64
    # matrix = 8*C*K bytes.
    dense_AK = 8 * A * K
    dense_CK = 8 * C * K

    # Self-documenting catch proof: the threshold sits STRICTLY below
    # both forbidden dense allocations. Hence any code path that
    # materialises a dense (A,K) or (C,K) float64 array necessarily
    # pushes peak_eb above `threshold` and fails the primary assert
    # below — this is exactly the "would FAIL on a real dense (A,K) or
    # (C,K) allocation" property the spec requires.
    assert threshold < dense_AK and threshold < dense_CK, (
        f"guard not below both dense allocs: S={S} C={C} "
        f"threshold={threshold} dense_AK={dense_AK} dense_CK={dense_CK}"
    )

    # Primary memory assertion: the real sparse run stays under the tight
    # budget (and therefore far under either dense allocation).
    assert peak_eb < threshold, (
        f"fl_bookdemand_eb peak={peak_eb} exceeds tight sparse budget "
        f"threshold={threshold} (S={S} C={C} A={A} K={K}); "
        f"dense_AK={dense_AK} dense_CK={dense_CK} — a dense (A,K)/(C,K) "
        f"allocation likely crept into the sparse-FL machinery"
    )


# ---------------------------------------------------------------------------
# fl_bookdemand_resid — intrinsic residual demand (no `s`, no fitted constant)
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
#       (Addendum 2, 2026-05-18). Background coreset built ONLY over
#       book-orphan articles, so the operating point falls out of orphan
#       mass with zero tuned knobs.
# ---------------------------------------------------------------------------

def test_resid_book_rich_reduces_to_bookdemand():
    """Book-rich limit: 3 tight clusters, dense books per cluster, NO
    diffuse background → books cover the manifold, |O|≈0 → M_bg==0,
    mode 'book-only', 3 distinct cluster-aligned gaps (fl_bookdemand-like).

    Geometry caveat: 3 tight 60-pt clusters (0.1 jitter); sparse books at
    cen*1.2 give the cluster-spanning k-balls (the proven ball-forming
    layout from the eb suite). DENSE per-article coverage books (sampled
    at every article position) make the union of book k_book-NN cover ALL
    articles → |O|==0 → M_bg==0 → mode 'book-only' (fl_bookdemand limit).
    5 far decoy books produce no candidates of their own."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )
    rs = np.random.RandomState(121)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
             for i in range(3)]
    art = np.vstack(parts).astype(np.float32)
    # Coverage books: one per article (slightly perturbed) so every
    # article is within some book's k_book nearest → orphan set empty.
    coverage_books = (art + 0.3 * rs.randn(*art.shape).astype(np.float32)
                      ).astype(np.float32)
    bk = np.vstack([
        cen[0] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[1] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        cen[2] * 1.2 + 0.05 * rs.randn(8, d).astype(np.float32),
        coverage_books,
        40.0 + rs.randn(5, d).astype(np.float32),
    ]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    gaps = deflate(cands, "fl_bookdemand_resid", art, num_gaps=3,
                   min_gap_size=10, book_emb=bk)
    diag = _fl_bookdemand_resid_diag()
    assert diag["M_bg"] == 0, diag
    assert diag["mode"] == "book-only", diag
    assert len(gaps) == 3, f"expected 3 gaps, got {len(gaps)}"
    assert [g[0] for g in gaps] == [0, 1, 2]      # emission rank
    labels = [_eb_cluster_labels(art, ids, cen) for _i, _s, ids in gaps]
    assert len(set(labels)) == 3, f"labels not distinct: {labels}"


def test_resid_book_sparse_no_collapse():
    """CORE HYPOTHESIS: 3 clusters + 120-article diffuse background +
    only 1 book → most articles orphaned → orphan_frac high, M_bg>0,
    emits >= 3 gaps mapping to 3 distinct clusters (does NOT collapse).

    Contrast: plain fl_bookdemand collapses to < 3 here because the
    single book's demand is exhausted in one pick (asserted — it is
    stable on this seeded geometry)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )
    rs = np.random.RandomState(131)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))  # diffuse bg
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2]).astype(np.float32)         # ONLY 1 book
    cands = build_kball_candidates(art, bk, 0.75, 1)

    # Contrast: plain fl_bookdemand collapses (< 3) on this exact geometry.
    g_bd = deflate(cands, "fl_bookdemand", art, num_gaps=10,
                   min_gap_size=5, book_emb=bk)
    assert len(g_bd) < 3, (
        f"contrast precondition: fl_bookdemand should collapse <3, got {len(g_bd)}"
    )

    g_resid = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                      min_gap_size=5, book_emb=bk)
    diag = _fl_bookdemand_resid_diag()
    assert diag["M_bg"] > 0, diag
    assert diag["orphan_frac"] > 0.5, diag
    assert diag["mode"] == "residual", diag
    assert len(g_resid) >= 3, f"resid collapsed: got {len(g_resid)} gaps"
    labels = {_eb_cluster_labels(art, ids, cen) for _i, _s, ids in g_resid}
    assert len({l for l in labels if l >= 0}) >= 3, (
        f"resid gaps did not map to 3 distinct clusters; labels={labels}"
    )


def test_resid_no_book_graceful():
    """0 books (book_emb None AND empty) → O = all articles →
    background-only, emits >= 3 gaps, NOT []. mode 'background-only'."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )
    rs = np.random.RandomState(141)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    art = np.vstack(parts).astype(np.float32)
    # Candidates still need SOME book to seed k-balls; the strategy is run
    # with book_emb None / empty so there are zero book customers.
    bk_for_cands = np.vstack([cen[0] * 1.2, cen[1] * 1.2,
                              cen[2] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk_for_cands, 0.75, 1)

    for be in (None, np.zeros((0, d), np.float32)):
        g = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                    min_gap_size=5, book_emb=be)
        diag = _fl_bookdemand_resid_diag()
        assert diag["mode"] == "background-only", (be is None, diag)
        assert diag["n_books"] == 0, diag
        assert len(g) >= 3, f"no-book run collapsed: {len(g)} gaps (be={be!r})"
        labels = {_eb_cluster_labels(art, ids, cen) for _i, _s, ids in g}
        assert len({l for l in labels if l >= 0}) >= 3, (
            f"no-book gaps not distinct; labels={labels}"
        )


def test_resid_orphan_monotone():
    """Vary ONLY book count (dense vs sparse) on the same geometry. The
    intrinsic regulator must be monotone in book sparsity with NO fitted
    constant: orphan_frac(sparse) > orphan_frac(dense) AND M_bg strictly
    larger for sparse."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )
    rs = np.random.RandomState(151)
    d = 32
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)

    bk_dense = np.vstack([
        cen[0] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
        cen[1] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
        cen[2] * 1.2 + 0.05 * rs.randn(20, d).astype(np.float32),
    ]).astype(np.float32)
    bk_sparse = np.vstack([cen[0] * 1.2]).astype(np.float32)

    c_dense = build_kball_candidates(art, bk_dense, 0.75, 1)
    deflate(c_dense, "fl_bookdemand_resid", art, num_gaps=10,
            min_gap_size=5, book_emb=bk_dense)
    diag_dense = _fl_bookdemand_resid_diag()

    c_sparse = build_kball_candidates(art, bk_sparse, 0.75, 1)
    deflate(c_sparse, "fl_bookdemand_resid", art, num_gaps=10,
            min_gap_size=5, book_emb=bk_sparse)
    diag_sparse = _fl_bookdemand_resid_diag()

    assert diag_sparse["n_books"] < diag_dense["n_books"]
    assert diag_sparse["orphan_frac"] > diag_dense["orphan_frac"], (
        diag_sparse, diag_dense)
    assert diag_sparse["M_bg"] > diag_dense["M_bg"], (
        diag_sparse, diag_dense)


def test_resid_M_bg_normalized_per_book():
    """Isolates the per-book normalization in
    M_bg = |O| // max(k_book·n_books, k_book) (Addendum 2 fix that escapes
    A-keyed-EB equivalence). Distinct from test_resid_orphan_monotone
    (which varies orphan mass via book sparsity): here ALL books are
    CO-LOCATED on one cluster, so the orphan set is held ~constant while
    only n_books changes. M_bg must be STRICTLY smaller when n_books is
    larger (denominator grows with n_books) — proving the regulator is
    orphan density PER BOOK, not absolute |O| (the latter would be the
    refuted A-keyed EB)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )
    rs = np.random.RandomState(207)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    parts = [cen[i] + 0.1 * rs.randn(80, d).astype(np.float32)
             for i in range(3)]
    parts.append(0.1 * rs.randn(200, d).astype(np.float32))  # diffuse bg
    art = np.vstack(parts).astype(np.float32)

    def _run(n_books):
        # ALL books clustered onto cluster 0 → their k_book-NN union is
        # ~constant regardless of n_books, so |O| stays ~fixed; only the
        # per-book denominator changes.
        bk = np.vstack([
            cen[0] * 1.2 + 0.02 * rs.randn(d).astype(np.float32)
            for _ in range(n_books)
        ]).astype(np.float32)
        cands = build_kball_candidates(art, bk, 0.75, 1)
        deflate(cands, "fl_bookdemand_resid", art, num_gaps=15,
                min_gap_size=3, book_emb=bk)
        return _fl_bookdemand_resid_diag()

    diag_few = _run(3)
    diag_many = _run(60)

    # Orphan mass comparable (clustered books don't blanket the manifold).
    assert abs(diag_few["n_orphans"] - diag_many["n_orphans"]) <= 0.05 * (
        diag_few["n_orphans"]), (diag_few, diag_many)
    # Per-book normalization active: more books ⇒ strictly smaller M_bg
    # despite comparable orphan mass.
    assert diag_many["M_bg"] < diag_few["M_bg"], (
        f"per-book normalization inactive (looks A-keyed): "
        f"few={diag_few} many={diag_many}")


def test_resid_no_s_knob():
    """Parameter-free: the signature has NO eb_target_shrinkage / s-style
    operating-point parameter, and a stray such kwarg passed via **params
    is harmless/ignored (output byte-identical with vs without it)."""
    import inspect
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid,
    )
    sig = inspect.signature(_fl_bookdemand_resid)
    forbidden = {"eb_target_shrinkage", "s", "s_target", "target_shrinkage",
                 "shrinkage", "lam", "lambda_"}
    assert not (forbidden & set(sig.parameters)), (
        f"resid must be parameter-free; found {forbidden & set(sig.parameters)}"
    )
    rs = np.random.RandomState(161)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)

    g0 = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                 min_gap_size=5, book_emb=bk)
    g1 = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                 min_gap_size=5, book_emb=bk, eb_target_shrinkage=0.5)
    assert len(g0) == len(g1)
    for (r0, s0, m0), (r1, s1, m1) in zip(g0, g1):
        assert r0 == r1 and s0 == s1
        assert np.array_equal(np.asarray(m0), np.asarray(m1)), (
            "stray eb_target_shrinkage kwarg changed output"
        )


def test_resid_determinism():
    """Two runs must be byte-identical (FlatL2 exact, KMeans seeded
    random_state=0 n_init=1, deterministic argmax tie-break)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(171)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2, cen[1] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)

    g1 = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                 min_gap_size=3, book_emb=bk)
    g2 = deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
                 min_gap_size=3, book_emb=bk)
    assert len(g1) == len(g2)
    for (r1, s1, m1), (r2, s2, m2) in zip(g1, g2):
        assert r1 == r2 and s1 == s2
        assert np.array_equal(np.asarray(m1), np.asarray(m2)), (
            "resid not deterministic across runs"
        )


def test_resid_monotone_gains():
    """Greedy marginal gains (submodular FL) must be non-increasing."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_trace,
    )
    rs = np.random.RandomState(181)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 10.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32) for i in range(3)]
    parts.append(0.1 * rs.randn(120, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[0] * 1.2]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    deflate(cands, "fl_bookdemand_resid", art, num_gaps=10,
            min_gap_size=3, book_emb=bk)
    gn = _fl_bookdemand_resid_trace()
    assert len(gn) >= 2, f"need >=2 gains to check monotonicity, got {gn}"
    for a, b in zip(gn, gn[1:]):
        assert b <= a + 1e-9, f"gains not non-increasing: {gn}"


def test_resid_memory_bounded():
    """GENUINE memory guard, mirroring test_fl_bookdemand_eb_memory_bounded.

    Many tight, well-separated micro-clusters so the candidate member
    mass S = Σ|members| is O(A) and ≪ A*K and ≪ C*K (C = n_books+M_bg).
    The forbidden dense baselines are 8*A*K and 8*C*K float64 matrices
    (NO spurious `d` factor). The tight sparse threshold is constructed
    to sit STRICTLY below BOTH, so any dense (A,K)/(C,K) allocation in
    the sparse-FL machinery necessarily pushes peak above it and fails.
    """
    import tracemalloc
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand_resid_diag,
    )

    rs = np.random.RandomState(181)
    d = 16
    n_clusters, pts_per = 5000, 4
    centers = (rs.randn(n_clusters, d).astype(np.float32)) * 80.0   # far apart
    parts = [centers[i] + 0.02 * rs.randn(pts_per, d).astype(np.float32)
             for i in range(n_clusters)]
    art = np.vstack(parts).astype(np.float32)            # A = 20000
    # SEED books (one per cluster) only seed the k-balls: a p=0.75 ball
    # captures its own (and only its own) tight cluster → per-candidate
    # ball ~= 4, so S = Σ|members| is O(A) and ≪ A*K and ≪ C*K.
    seed_bk = (centers + 0.3).astype(np.float32)
    cands = build_kball_candidates(art, seed_bk, 0.75, 1)
    A = len(art)
    K = len(cands)
    # CUSTOMER books passed to the resid run are a SMALL clustered subset
    # (real books cluster — they never blanket the manifold). Under the
    # corrected per-book formula M_bg = |O| // max(k_book·n_books, k_book),
    # ~all clusters are orphaned (|O| ≈ A) so M_bg is large and
    # C = n_books + M_bg stays large enough that 8*C*K comfortably exceeds
    # the tight sparse threshold (the "guard below both dense allocs"
    # self-proof holds, and any real dense (C,K)/(A,K) allocation in the
    # sparse-FL machinery still fails the primary assert). Decoupling the
    # seed books from the customer books is deliberate: it isolates the
    # sparse-ball requirement (needs many seeds) from the large-orphan-mass
    # requirement (needs few, clustered customer books) which the corrected
    # per-book formula otherwise puts in tension.
    bk = (centers[:4] + 0.3).astype(np.float32)

    S = sum(int(len(c.members)) for c in cands)
    assert S < 0.05 * A * K, (
        f"sparsity precondition violated: S={S} not << A*K={A * K} "
        f"(S/(A*K)={S / (A * K):.5f}); fixture k-balls are not sparse"
    )

    tracemalloc.start()
    deflate(cands, "fl_bookdemand_resid", art, num_gaps=50,
            min_gap_size=3, book_emb=bk)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Customer count C = n_books + M_bg, taken from the diag the run just
    # wrote (M_bg is intrinsic to orphan mass — cannot be precomputed
    # without re-deriving the orphan set, so read it back).
    diag = _fl_bookdemand_resid_diag()
    n_books = len(bk)
    M_bg = int(diag["M_bg"])
    C = n_books + M_bg

    DICT_OVERHEAD = 80                       # bytes per sparse dict entry
    SPARSE_MULT = 6                          # modest multiplicative slack
    FIXED_MB = 4 * 1024 * 1024               # fixed slack: index + coreset
    sparse_bytes = S * (8 + DICT_OVERHEAD) + C * 8
    threshold = SPARSE_MULT * sparse_bytes + FIXED_MB

    dense_AK = 8 * A * K
    dense_CK = 8 * C * K

    assert threshold < dense_AK and threshold < dense_CK, (
        f"guard not below both dense allocs: S={S} C={C} "
        f"threshold={threshold} dense_AK={dense_AK} dense_CK={dense_CK}"
    )

    assert peak < threshold, (
        f"fl_bookdemand_resid peak={peak} exceeds tight sparse budget "
        f"threshold={threshold} (S={S} C={C} A={A} K={K} M_bg={M_bg}); "
        f"dense_AK={dense_AK} dense_CK={dense_CK} — a dense (A,K)/(C,K) "
        f"allocation likely crept into the sparse-FL machinery"
    )


# ---------------------------------------------------------------------------
# best-of-both reframes: fl_union_bd_sub (union, not mixture) and
# fl_gen_rank (decouple generation from selection).
# Spec: docs/superpowers/specs/2026-05-18-fl-bookdemand-eb-shrinkage-design.md
#       (Addendum 3, 2026-05-18). A mixed demand objective dilutes; these
#       arms stop mixing — A unions the two pure signals, B uses
#       submodular_fl as a robust generator then ranks by book demand.
# ---------------------------------------------------------------------------

def _gr_key(gaps):
    """Order-stable key: list of (rank, size, sorted-member-tuple)."""
    return [(int(r), int(s), tuple(sorted(int(x) for x in ids)))
            for r, s, ids in gaps]


def _gr_setkey(gaps):
    """Order-agnostic key: list of frozenset member sets."""
    return [frozenset(int(x) for x in ids) for _, _, ids in gaps]


def test_union_is_concatenation():
    """fl_union_bd_sub returns exactly the _fl_bookdemand gaps followed
    by the _submodular_fl gaps: count == |bd|+|sub|, the bd-half member
    sets equal _fl_bookdemand's output (in order), the sub-half equal
    _submodular_fl's output (in order), element-0 renumbered 0..N-1."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_bookdemand, _submodular_fl,
    )
    rs = np.random.RandomState(17)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
             for i in range(3)]
    parts.append(0.1 * rs.randn(150, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[i] * 1.2 for i in range(3) for _ in range(6)]
                   ).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    sigma = np.ones(len(art), dtype=np.float64)

    g_bd = _fl_bookdemand(cands, art, num_gaps=200, min_gap_size=3,
                          book_emb=bk)
    g_sub = _submodular_fl(cands, art, num_gaps=200, min_gap_size=3,
                           sigma=sigma)
    g_union = deflate(cands, "fl_union_bd_sub", art, num_gaps=200,
                      min_gap_size=3, sigma=sigma, book_emb=bk)

    assert len(g_union) == len(g_bd) + len(g_sub)
    # element-0 renumbered 0..N-1 over the concatenation
    assert [x[0] for x in g_union] == list(range(len(g_union)))
    n_bd = len(g_bd)
    # bd-half member sets equal _fl_bookdemand output, IN ORDER
    assert _gr_setkey(g_union[:n_bd]) == _gr_setkey(g_bd)
    # sub-half member sets equal _submodular_fl output, IN ORDER
    assert _gr_setkey(g_union[n_bd:]) == _gr_setkey(g_sub)
    # sizes preserved per tuple
    assert [x[1] for x in g_union[:n_bd]] == [x[1] for x in g_bd]
    assert [x[1] for x in g_union[n_bd:]] == [x[1] for x in g_sub]


def test_genrank_no_book_is_submodular_order():
    """book_emb=None → fl_gen_rank emits the _submodular_fl pool in
    submodular order (member sets AND order identical), mode 'gen-only',
    NOT []."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _submodular_fl, _fl_gen_rank_diag,
    )
    rs = np.random.RandomState(17)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
             for i in range(3)]
    parts.append(0.1 * rs.randn(150, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[i] * 1.2 for i in range(3) for _ in range(6)]
                   ).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    sigma = np.ones(len(art), dtype=np.float64)

    g_sub = _submodular_fl(cands, art, num_gaps=200, min_gap_size=3,
                           sigma=sigma)
    g_gr = deflate(cands, "fl_gen_rank", art, num_gaps=200,
                   min_gap_size=3, sigma=sigma, book_emb=None)

    assert len(g_gr) > 0
    # member sets AND order identical to the submodular pool
    assert _gr_setkey(g_gr) == _gr_setkey(g_sub)
    # element-0 = emission order 0..N-1
    assert [x[0] for x in g_gr] == list(range(len(g_gr)))
    diag = _fl_gen_rank_diag()
    assert diag["mode"] == "gen-only", diag
    assert diag["n_books"] == 0, diag
    assert diag["pool_size"] == len(g_sub), diag
    assert diag["n_book_shaped"] == 0, diag


def test_genrank_prioritizes_book_shaped():
    """Fixture = book-aligned clusters + a book-free coverage region.
    Gaps overlapping book neighborhoods are emitted before the
    book-free ones; diag.n_book_shaped>0; backfill (bd==0) gaps are
    still present (len unchanged vs the submodular pool — none dropped).
    """
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _submodular_fl, _fl_gen_rank_diag,
    )
    from bookgap.index import build_index, FlatL2
    rs = np.random.RandomState(23)
    d = 24
    # 3 book-aligned clusters + 1 large book-free coverage region.
    cen = np.zeros((4, d), np.float32)
    for i in range(4):
        cen[i, i] = 14.0
    parts = [cen[i] + 0.10 * rs.randn(70, d).astype(np.float32)
             for i in range(3)]               # book-aligned clusters
    parts.append(cen[3] + 0.10 * rs.randn(200, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    # Books only near the first 3 cluster centroids (NONE near cen[3]).
    bk = np.vstack([cen[i] * 1.15 + 0.03 * rs.randn(6, d).astype(np.float32)
                    for i in range(3)]).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    sigma = np.ones(len(art), dtype=np.float64)

    g_sub = _submodular_fl(cands, art, num_gaps=400, min_gap_size=3,
                           sigma=sigma)
    g_gr = deflate(cands, "fl_gen_rank", art, num_gaps=400,
                   min_gap_size=3, sigma=sigma, book_emb=bk)

    diag = _fl_gen_rank_diag()
    assert diag["mode"] == "gen-rank", diag
    assert diag["n_book_shaped"] > 0, diag
    assert diag["pool_size"] == len(g_sub), diag
    # backfill kept: nothing dropped vs the submodular pool
    assert len(g_gr) == len(g_sub), (len(g_gr), len(g_sub))

    # Determine which emitted gaps overlap a book neighborhood.
    aidx = build_index(FlatL2(), np.ascontiguousarray(art.astype(np.float32)))
    _D, Nb = aidx.search(np.ascontiguousarray(bk.astype(np.float32)), k=10)
    book_articles = set(int(x) for x in np.asarray(Nb).reshape(-1).tolist())

    def _is_book_shaped(ids):
        return len(set(int(x) for x in ids) & book_articles) > 0

    flags = [_is_book_shaped(ids) for _, _, ids in g_gr]
    # No book-free gap may precede a book-shaped one (book-shaped first,
    # then backfill in submodular order).
    last_true = max((i for i, f in enumerate(flags) if f), default=-1)
    first_false = next((i for i, f in enumerate(flags) if not f),
                       len(flags))
    assert last_true < first_false or last_true == -1, (
        f"book-shaped gap appears after a book-free one: flags={flags}")
    assert any(flags), "no book-shaped gap surfaced at all"


def test_genrank_determinism():
    """Two fl_gen_rank runs are byte-identical (ranks, sizes, member
    order)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(17)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
             for i in range(3)]
    parts.append(0.1 * rs.randn(150, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[i] * 1.2 for i in range(3) for _ in range(6)]
                   ).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    sigma = np.ones(len(art), dtype=np.float64)

    def _run():
        g = deflate(cands, "fl_gen_rank", art, num_gaps=200,
                    min_gap_size=3, sigma=sigma, book_emb=bk)
        return [(int(r), int(s), tuple(int(x) for x in ids))
                for r, s, ids in g]

    assert _run() == _run()


def test_genrank_memory_bounded():
    """GENUINE memory guard, mirroring test_fl_bookdemand_eb_memory_bounded.

    Many tight, well-separated micro-clusters so the gen-rank working
    set — the POOL member mass Sp = Σ_p|pool_members_p| (NOT the full
    candidate mass S; gen-rank's inverted index/per-gap dicts are keyed
    by the tiny submodular pool, not all candidates) — is ≪ A*K and ≪
    A*pool_size. The forbidden dense baselines are 8*A*K (article×
    candidate) and 8*A*pool_size (article×pool scoring) float64 matrices
    (NO spurious `d` factor; 8*C*K is NOT meaningful here — C is the
    tiny pool count, not a k-means coreset as in the eb/resid family).
    The tight sparse threshold is constructed to sit STRICTLY below
    BOTH, so any dense (A,K)/(A,pool_size) allocation in the gen-rank
    scoring machinery necessarily pushes peak above it and fails.
    """
    import tracemalloc
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _fl_gen_rank_diag,
    )

    rs = np.random.RandomState(181)
    d = 16
    n_clusters, pts_per = 5000, 4
    centers = (rs.randn(n_clusters, d).astype(np.float32)) * 80.0
    parts = [centers[i] + 0.02 * rs.randn(pts_per, d).astype(np.float32)
             for i in range(n_clusters)]
    art = np.vstack(parts).astype(np.float32)            # A = 20000
    seed_bk = (centers + 0.3).astype(np.float32)
    cands = build_kball_candidates(art, seed_bk, 0.75, 1)
    A = len(art)
    K = len(cands)
    # A SMALL clustered customer-book subset (real books cluster).
    bk = (centers[:4] + 0.3).astype(np.float32)

    S = sum(int(len(c.members)) for c in cands)
    assert S < 0.05 * A * K, (
        f"sparsity precondition violated: S={S} not << A*K={A * K} "
        f"(S/(A*K)={S / (A * K):.5f}); fixture k-balls are not sparse"
    )

    # sigma is ALWAYS supplied on the production path
    # (kball_seeds_from_state passes state['sigma']); the embedded
    # _submodular_fl generator requires it.
    sigma = np.ones(A, dtype=np.float64)
    tracemalloc.start()
    deflate(cands, "fl_gen_rank", art, num_gaps=50,
            min_gap_size=3, sigma=sigma, book_emb=bk)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # gen-rank's genuine working set is the POOL member mass, NOT the
    # full candidate mass S: the inverted index is article→pool-gap and
    # the per-gap sparse {book:isect} dict is keyed by the (tiny) pool,
    # so the true sparse footprint scales with
    #   Sp = Σ_p |pool_members_p|  (≤ num_gaps · ball-size ≪ S ≪ A·K).
    # Re-derive the pool exactly as gen-rank does (deterministic) to get
    # Sp and pool_size — C = pool_size + n_books is correspondingly tiny.
    from artifacts_scripts.gap_deflation_bakeoff import _submodular_fl
    pool = _submodular_fl(cands, art, num_gaps=50, min_gap_size=3,
                          sigma=sigma)
    Sp = sum(int(len(ids)) for _, _, ids in pool)
    diag = _fl_gen_rank_diag()
    n_books = len(bk)
    pool_size = int(diag["pool_size"])
    assert pool_size == len(pool)
    C = pool_size + n_books

    # Sparse precondition: the gen-rank working set is O(pool mass),
    # vastly below any dense article×candidate or article×pool matrix.
    assert Sp < 0.05 * A * K, (
        f"gen-rank sparsity precondition violated: Sp={Sp} not << "
        f"A*K={A * K} (Sp/(A*K)={Sp / (A * K):.6f})"
    )

    DICT_OVERHEAD = 80                       # bytes per sparse dict entry
    SPARSE_MULT = 6                          # modest multiplicative slack
    # Fixed slack absorbs the legitimately-bounded _submodular_fl
    # generator working set (one FlatL2 index over (A,d) float32, the
    # length-A best_cov vector, and the per-candidate sim_vecs whose
    # TOTAL is Σ|cand.members|=S — sparse, bounded by ball mass, NOT a
    # dense A×K matrix) plus interpreter/import noise. It stays FAR
    # below the forbidden dense (A,pool_size) ceiling (8·A·pool_size).
    FIXED_MB = 6 * 1024 * 1024               # fixed slack: index + gen pool
    # Sparse footprint: the inverted index + per-pool-gap {book:isect}
    # dicts (≤ Sp + pool_size·n_books entries) + the 1-D length-pool_size
    # `bd` vector. Bounded by Sp (NOT S, NOT A·K, NOT A·pool_size).
    sparse_bytes = (Sp + pool_size * n_books) * (8 + DICT_OVERHEAD) + C * 8
    threshold = SPARSE_MULT * sparse_bytes + FIXED_MB

    # Forbidden dense regressions (NO spurious `d` factor): a dense
    # article×candidate (A,K) float64 matrix = 8·A·K bytes, and a dense
    # article×pool scoring (A,pool_size) float64 matrix = 8·A·pool_size
    # bytes. Spec Addendum 3 additionally pins the canonical 8·C·K pair
    # (C = pool_size + n_books, same K = |build_kball_candidates|): even
    # though C is the tiny pool+books count here (not a k-means coreset
    # as in the eb/resid family), the (C,K) dense matrix is still a
    # spec-required forbidden baseline the sparse guard must undercut.
    dense_AK = 8 * A * K
    dense_CK = 8 * C * K
    dense_Apool = 8 * A * pool_size

    # Self-documenting catch proof: the threshold sits STRICTLY below
    # ALL THREE forbidden dense allocations — the spec-required pair
    # (8·A·K, 8·C·K) plus the extra, stricter arm-relevant
    # 8·A·pool_size — so any code path materialising a dense (A,K),
    # (C,K), or (A,pool_size) float64 array necessarily pushes peak
    # above `threshold` and fails the primary assert.
    assert (
        threshold < dense_AK
        and threshold < dense_CK
        and threshold < dense_Apool
    ), (
        f"guard not below all dense allocs: Sp={Sp} C={C} K={K} "
        f"threshold={threshold} dense_AK={dense_AK} "
        f"dense_CK={dense_CK} dense_Apool={dense_Apool}"
    )

    assert peak < threshold, (
        f"fl_gen_rank peak={peak} exceeds tight sparse budget "
        f"threshold={threshold} (Sp={Sp} C={C} A={A} K={K} "
        f"pool_size={pool_size}); dense_AK={dense_AK} "
        f"dense_CK={dense_CK} dense_Apool={dense_Apool} — a dense "
        f"(A,K)/(C,K)/(A,pool) allocation likely crept into gen-rank"
    )


def test_union_genrank_registered_distinct():
    """Both names are in _STRATEGIES, each deterministic, and each
    differs from plain submodular_fl AND fl_bookdemand on a fixture
    where the difference must show (union longer than either; gen_rank
    order != submodular order when books are present)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate, _STRATEGIES,
        _fl_bookdemand, _submodular_fl,
    )
    assert "fl_union_bd_sub" in _STRATEGIES
    assert "fl_gen_rank" in _STRATEGIES

    rs = np.random.RandomState(17)
    d = 24
    cen = np.zeros((3, d), np.float32)
    for i in range(3):
        cen[i, i] = 12.0
    parts = [cen[i] + 0.1 * rs.randn(60, d).astype(np.float32)
             for i in range(3)]
    parts.append(0.1 * rs.randn(150, d).astype(np.float32))
    art = np.vstack(parts).astype(np.float32)
    bk = np.vstack([cen[i] * 1.2 for i in range(3) for _ in range(6)]
                   ).astype(np.float32)
    cands = build_kball_candidates(art, bk, 0.75, 1)
    sigma = np.ones(len(art), dtype=np.float64)

    def _u():
        return deflate(cands, "fl_union_bd_sub", art, num_gaps=200,
                       min_gap_size=3, sigma=sigma, book_emb=bk)

    def _r():
        return deflate(cands, "fl_gen_rank", art, num_gaps=200,
                       min_gap_size=3, sigma=sigma, book_emb=bk)

    # deterministic
    assert _gr_key(_u()) == _gr_key(_u())
    assert _gr_key(_r()) == _gr_key(_r())

    g_bd = _fl_bookdemand(cands, art, num_gaps=200, min_gap_size=3,
                          book_emb=bk)
    g_sub = _submodular_fl(cands, art, num_gaps=200, min_gap_size=3,
                           sigma=sigma)
    g_u = _u()
    g_r = _r()

    # union strictly longer than either pure arm
    assert len(g_u) > len(g_bd)
    assert len(g_u) > len(g_sub)
    assert _gr_key(g_u) != _gr_key(g_bd)
    assert _gr_key(g_u) != _gr_key(g_sub)

    # gen_rank is a PERMUTATION of the submodular pool (same BAG of
    # member sets — gen_rank keeps each pool gap's members intact, only
    # the emission ORDER changes when book demand re-ranks).
    assert sorted(_gr_setkey(g_r), key=lambda s: sorted(s)) == \
        sorted(_gr_setkey(g_sub), key=lambda s: sorted(s))
    # ...but the ORDER differs from plain submodular order (books rerank).
    assert [t[2] for t in _gr_key(g_r)] != [t[2] for t in _gr_key(g_sub)], (
        "gen_rank order identical to submodular order despite books")
    assert _gr_key(g_r) != _gr_key(g_bd)


# ---------------------------------------------------------------------------
# Ranking-stage book-aware selection (the BINDING stage).
# Spec: docs/superpowers/specs/2026-05-18-ranking-stage-book-aware-design.md
# `_rank_survivors` is a strictly-additive axis over run_arm; rank="effsize"
# (default) MUST be byte-identical to the prior
#   sorted(survivors, key=lambda g: g.get("effective_size", 0.0), reverse=True)
# line — that no-op-refactor property is the additive-proof gate.
# ---------------------------------------------------------------------------


def test_rank_registered():
    """The 3 rank names dispatch; an unknown rank raises a clear
    ValueError naming the bad rank."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors
    s = [{"effective_size": 2.0, "refined_member_indices": [0]},
         {"effective_size": 5.0, "refined_member_indices": [1]}]
    art = np.zeros((4, 3), dtype=np.float32)
    for r in ("effsize", "bookdemand", "effsize_x_bd"):
        out = _rank_survivors(s, r, book_emb=None,
                              article_emb_for_rank=art, k_book=10)
        assert isinstance(out, list) and len(out) == len(s)
    with pytest.raises(ValueError, match="unknown rank 'nope'"):
        _rank_survivors(s, "nope", book_emb=None,
                        article_emb_for_rank=art, k_book=10)


def test_rank_effsize_is_identity():
    """rank="effsize" returns EXACTLY
       sorted(s, key=lambda g: g.get("effective_size",0.0), reverse=True)
    — same order AND same object identities (incl. ties + a survivor
    missing the key → treated as 0.0). This proves the run_arm line
    swap is a no-op refactor (the additive-proof gate)."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors
    s = [
        {"effective_size": 3.0, "refined_member_indices": [1, 2]},
        {"effective_size": 9.0, "refined_member_indices": [3]},
        {"effective_size": 3.0, "refined_member_indices": [4, 5, 6]},
        {"refined_member_indices": [0]},          # missing key → 0.0
        {"effective_size": 9.0, "refined_member_indices": []},
        {"effective_size": 0.0, "refined_member_indices": [2]},
    ]
    art = np.zeros((7, 4), dtype=np.float32)
    expected = sorted(s, key=lambda g: g.get("effective_size", 0.0),
                      reverse=True)
    got = _rank_survivors(s, "effsize", book_emb=None,
                          article_emb_for_rank=art, k_book=10)
    assert got == expected
    assert [id(x) for x in got] == [id(x) for x in expected]
    # also identity-stable when a (degenerate) book_emb is supplied —
    # the effsize branch must NEVER consult books.
    got2 = _rank_survivors(s, "effsize",
                           book_emb=np.zeros((3, 4), dtype=np.float32),
                           article_emb_for_rank=art, k_book=10)
    assert [id(x) for x in got2] == [id(x) for x in expected]


def test_rank_bd_zero_reduces_to_effsize():
    """No books (None / empty) OR survivors with no book overlap ⇒
    effsize_x_bd order == effsize order, and bookdemand order ==
    effsize-stable order. Graceful, no special-casing."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors
    s = [
        {"effective_size": 4.0, "refined_member_indices": [10]},
        {"effective_size": 7.0, "refined_member_indices": [11]},
        {"effective_size": 4.0, "refined_member_indices": [12]},
        {"effective_size": 1.0, "refined_member_indices": [13]},
    ]
    art = np.zeros((4, 5), dtype=np.float32)
    eff = _rank_survivors(s, "effsize", book_emb=None,
                          article_emb_for_rank=art, k_book=10)
    eff_ids = [id(x) for x in eff]

    # book_emb None
    for be in (None, np.zeros((0, 5), dtype=np.float32)):
        bx = _rank_survivors(s, "effsize_x_bd", book_emb=be,
                             article_emb_for_rank=art, k_book=10)
        bd = _rank_survivors(s, "bookdemand", book_emb=be,
                             article_emb_for_rank=art, k_book=10)
        assert [id(x) for x in bx] == eff_ids
        assert [id(x) for x in bd] == eff_ids

    # books exist but none of the refined members land in any Nb
    # (members are global ids 10..13 which are NOT valid article-local
    # positions into a 4-article array → mapped-away → bd == 0 for all).
    bk = np.ones((3, 5), dtype=np.float32)
    bx2 = _rank_survivors(s, "effsize_x_bd", book_emb=bk,
                          article_emb_for_rank=art, k_book=10)
    bd2 = _rank_survivors(s, "bookdemand", book_emb=bk,
                          article_emb_for_rank=art, k_book=10)
    assert [id(x) for x in bx2] == eff_ids
    assert [id(x) for x in bd2] == eff_ids


def test_rank_bookaware_prioritizes_book_shaped():
    """Two survivors with EQUAL effective_size: one whose refined
    members overlap a book's Nb, one that doesn't. Under effsize_x_bd
    the book-shaped one ranks first; assert the exact bd values by
    hand calc (reused _fl_bookdemand Jaccard isect/(|m|+k_b−isect))."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors
    # 4 articles: two tight pairs far apart. Article-local ids 0..3.
    art = np.array([
        [0.0, 0.0],
        [0.01, 0.0],
        [50.0, 0.0],
        [50.0, 0.01],
    ], dtype=np.float32)
    # One book sitting on the first pair. k_book=2 → Nb[0] = {0,1}.
    bk = np.array([[0.0, 0.0]], dtype=np.float32)
    # Survivor A overlaps the book neighborhood (members {0,1});
    # survivor B is the far pair (members {2,3}) — no overlap.
    s = [
        {"effective_size": 5.0, "refined_member_indices": [2, 3]},  # B
        {"effective_size": 5.0, "refined_member_indices": [0, 1]},  # A
    ]
    out = _rank_survivors(s, "effsize_x_bd", book_emb=bk,
                          article_emb_for_rank=art, k_book=2)
    # Hand calc: k_b = min(2, A=4) = 2; Nb[0] = {0,1}.
    #   A: |m|=2, isect=2, union=2+2−2=2 → bd = 2/2 = 1.0
    #   B: |m|=2, isect=0 → bd = 0.0
    # effsize_x_bd key: A = 5*(1+1)=10 ; B = 5*(1+0)=5 → A first.
    assert out[0]["refined_member_indices"] == [0, 1], out
    assert out[1]["refined_member_indices"] == [2, 3], out

    # bookdemand: same ordering here (A has strictly larger bd).
    out_bd = _rank_survivors(s, "bookdemand", book_emb=bk,
                             article_emb_for_rank=art, k_book=2)
    assert out_bd[0]["refined_member_indices"] == [0, 1], out_bd


def test_rank_determinism():
    """Two calls byte-identical (stable sort, no RNG) for every rank."""
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors
    rs = np.random.RandomState(7)
    art = rs.randn(40, 6).astype(np.float32)
    bk = rs.randn(5, 6).astype(np.float32)
    s = [{"effective_size": float(rs.randint(0, 5)),
          "refined_member_indices": sorted(
              int(x) for x in rs.choice(40, size=4, replace=False))}
         for _ in range(30)]

    def _key(lst):
        return [(round(g.get("effective_size", 0.0), 9),
                 tuple(g.get("refined_member_indices", []))) for g in lst]

    for r in ("effsize", "bookdemand", "effsize_x_bd"):
        a = _rank_survivors(s, r, book_emb=bk,
                            article_emb_for_rank=art, k_book=10)
        b = _rank_survivors(s, r, book_emb=bk,
                            article_emb_for_rank=art, k_book=10)
        assert _key(a) == _key(b)


def test_rank_memory_bounded():
    """GENUINE memory guard mirroring test_fl_bookdemand_eb_memory_bounded.

    Many tight, well-separated micro-clusters so the rank working set —
    the survivor member mass Ss = Σ_g|members_g| — is ≪ A·N and ≪ C·N.
    The forbidden dense baselines are 8·A·N (article×survivor) and 8·C·N
    (book/customer×survivor) float64 matrices (NO spurious `d` factor).
    The tight sparse threshold is constructed to sit STRICTLY below
    BOTH, so any dense (A,N)/(C,N) allocation in the bd scoring
    machinery necessarily pushes peak above it and fails. The bd score
    reuses _fl_bookdemand's sparse inverted-index + per-survivor sparse
    {book:isect} dict (NO dense (A,N)/(C,N)/(A,N,d) array).
    """
    import tracemalloc
    import numpy as np
    from artifacts_scripts.gap_deflation_bakeoff import _rank_survivors

    rs = np.random.RandomState(811)
    d = 16
    n_clusters, pts_per = 1500, 8
    centers = (rs.randn(n_clusters, d).astype(np.float32)) * 80.0
    parts = [centers[i] + 0.02 * rs.randn(pts_per, d).astype(np.float32)
             for i in range(n_clusters)]
    art = np.vstack(parts).astype(np.float32)            # A = 12000
    A = len(art)
    # One survivor per cluster: its members are that cluster's
    # article-local ids (8 each) → genuinely sparse member mass.
    survivors = [
        {"effective_size": float(rs.randint(1, 9)),
         "refined_member_indices": list(
             range(i * pts_per, (i + 1) * pts_per))}
        for i in range(n_clusters)
    ]
    N = len(survivors)
    # Small clustered book subset (real books cluster).
    bk = (centers[:4] + 0.3).astype(np.float32)
    n_books = len(bk)

    Ss = sum(len(g["refined_member_indices"]) for g in survivors)
    assert Ss < 0.05 * A * N, (
        f"sparsity precondition violated: Ss={Ss} not << A*N={A * N} "
        f"(Ss/(A*N)={Ss / (A * N):.6f}); fixture members not sparse"
    )

    tracemalloc.start()
    _rank_survivors(survivors, "effsize_x_bd", book_emb=bk,
                    article_emb_for_rank=art, k_book=10)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    DICT_OVERHEAD = 80                       # bytes per sparse dict entry
    SPARSE_MULT = 6                          # modest multiplicative slack
    FIXED_MB = 6 * 1024 * 1024               # fixed slack: FlatL2 index
    # Sparse footprint: the inverted index (≤ Ss entries) + per-survivor
    # {book:isect} dicts (≤ N·n_books entries) + the 1-D length-N bd
    # vector. Bounded by Ss (NOT A·N, NOT C·N).
    C = N + n_books
    sparse_bytes = (Ss + N * n_books) * (8 + DICT_OVERHEAD) + C * 8
    threshold = SPARSE_MULT * sparse_bytes + FIXED_MB

    # Forbidden dense regressions (NO spurious `d` factor).
    dense_AN = 8 * A * N
    dense_CN = 8 * C * N

    assert threshold < dense_AN and threshold < dense_CN, (
        f"guard not below both dense allocs: Ss={Ss} C={C} N={N} "
        f"threshold={threshold} dense_AN={dense_AN} dense_CN={dense_CN}"
    )
    assert peak < threshold, (
        f"_rank_survivors peak={peak} exceeds tight sparse budget "
        f"threshold={threshold} (Ss={Ss} C={C} A={A} N={N}); "
        f"dense_AN={dense_AN} dense_CN={dense_CN} — a dense (A,N)/(C,N) "
        f"allocation likely crept into the bd scoring machinery"
    )
