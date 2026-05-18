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
