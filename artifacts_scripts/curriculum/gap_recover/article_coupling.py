"""Article-domain citation↔embedding coupling — the OpenAlex analog of
``core/coupling_diagnostics.py``'s ``coupling`` measurement for code substrates.

Places the *articles* (research-literature) domain on the SAME axis as the code
substrates (Lean +0.115, Python +0.108, Go +0.037, Rust +0.057, Bazel +0.161),
to test whether coupling strength — rather than modality — is what flips articles
to the embedding-modality. See ``docs/code-gap-triangulation.md``.

Definition (identical to ``coupling_diagnostics.coupling``):

    coupling = mean over sampled nodes C of
        ``cos(C, its graph-neighbors) − cos(C, random nodes)``
    + the fraction of C where neighbors are closer than random.

For articles, a "node" is an OpenAlex **work** and its "graph-neighbors" are the
works it **cites** (``referenced_works``), restricted to works that are in the
loaded sample (so they have embeddings). Cosine == dot on L2-normalised stored
embeddings.

Alignment (the crux). We use the Cloud API, which guarantees row-for-row
alignment across ``work_ids()`` / ``work_references()`` / ``embeddings()`` (all
share the same post-dedup, post-filter ``retained_positions`` order — see
``bookgap.cloud.ensure_cloud_data``). So work ``i``'s embedding is
``embeddings()[i]``, its OpenAlex id is ``work_ids()[i]``, and the ids it cites
are ``work_references()[i]``. Building ``{work_id -> row}`` over the loaded
sample maps each cited id to its row (when present), giving the in-sample
neighbour set with no manual ``row_id`` bookkeeping.

Scope: ``subfield=AI(1702)``, work-types ``(book, article)`` — a corpus already
embedded on disk with GIST-small (corpus fingerprint ``469a666eca9848a2``,
embedder fingerprint ``6bcf5b5e02946b7e``, ~1.6 M works). No fetching/embedding
is triggered; this only reads caches.

CPU only.

Regenerate (do not edit the printed numbers by hand):

    uv run python -m \
        artifacts_scripts.curriculum.gap_recover.article_coupling \
        --subfield 1702 --n-works 1500 --min-oper 5 --cap 50 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


def _build_cloud(subfield: str, work_types: tuple, snapshot_date: str):
    """Construct a Cloud for one subfield with the default (GIST + NONE-denoiser,
    normalised) settings — matching the already-embedded on-disk corpus."""
    from bookgap.cloud import Cloud
    from bookgap.configs.cloud.cloud_config import CloudConfig
    from bookgap.configs.data.data_config import SimHashDeduplicatorConfig
    from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
    from bookgap.configs.index.index_config import HNSWConfig
    from bookgap.deduplicators import Deduplicator
    from bookgap.index import IndexInfo
    from bookgap.util import Filter, FilterPair
    from hydra.utils import instantiate

    cfg = CloudConfig()  # defaults: NONE denoiser, drop_k=0, GIST-small, normalize=True, SimHash dedup
    pre = Filter(subfield=subfield, work_types=work_types)
    filter_pair = FilterPair(pre=pre, post=Filter())
    index_info: IndexInfo = instantiate(HNSWConfig())
    deduplicator: Deduplicator = instantiate(SimHashDeduplicatorConfig())
    return Cloud(
        filter_pair=filter_pair,
        index_info=index_info,
        max_works=cfg.max_works,  # None -> full subfield (matches fingerprint 469a666eca9848a2)
        deduplicator=deduplicator,
        normalize_embeddings=cfg.normalize_embeddings,
        denoiser=cfg.denoiser,
        drop_k=cfg.drop_k,
        embedder=SentenceTransformerEmbedderConfig(),
        snapshot_date=snapshot_date,
    )


def coupling(
    emb: NDArray[np.float32],
    cited_rows: List[NDArray[np.int64]],
    rng: np.random.Generator,
    n_works: int,
    min_oper: int,
    cap: int,
) -> dict:
    """Mean (over sampled works with >= min_oper in-sample cites) of
    cos(C, in-sample cited) - cos(C, equal-count random in-sample).

    ``emb`` must be L2-normalised (cos == dot). ``cited_rows[i]`` lists the
    in-sample row indices that work ``i`` cites.
    """
    n = emb.shape[0]
    all_rows = np.arange(n, dtype=np.int64)
    cand = np.array([i for i in range(n) if cited_rows[i].size >= min_oper], dtype=np.int64)
    rng.shuffle(cand)
    cand = cand[:n_works]

    lifts: List[float] = []
    closer: List[bool] = []
    csims: List[float] = []  # raw mean cosine to cited (sanity / absolute level)
    rsims: List[float] = []  # raw mean cosine to random
    for c in cand:
        citers = cited_rows[c]
        if citers.size > cap:
            citers = rng.choice(citers, cap, replace=False)
        csim = emb[citers] @ emb[c]
        rsim = emb[rng.choice(all_rows, citers.size, replace=False)] @ emb[c]
        cm, rm = float(csim.mean()), float(rsim.mean())
        lifts.append(cm - rm)
        closer.append(cm > rm)
        csims.append(cm)
        rsims.append(rm)

    lifts_a = np.asarray(lifts)
    return {
        "n": int(lifts_a.size),
        "lift_mean": float(lifts_a.mean()) if lifts_a.size else float("nan"),
        "lift_median": float(np.median(lifts_a)) if lifts_a.size else float("nan"),
        "frac_closer": float(np.mean(closer)) if closer else float("nan"),
        "raw_cos_cited": float(np.mean(csims)) if csims else float("nan"),
        "raw_cos_random": float(np.mean(rsims)) if rsims else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subfield", default="1702", help="OpenAlex subfield id (default 1702 = AI)")
    ap.add_argument("--work-types", default="book,article",
                    help="comma-separated work types defining the corpus (default book,article)")
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--n-works", type=int, default=1500,
                    help="number of sampled works with >= min-oper in-sample cites")
    ap.add_argument("--min-oper", type=int, default=5,
                    help="min in-sample cited works for a node to qualify (matches code coupling)")
    ap.add_argument("--cap", type=int, default=50, help="cap on cited works per node (matches code)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="Articles (OpenAlex AI subfield)")
    args = ap.parse_args()

    work_types = tuple(s.strip() for s in args.work_types.split(",") if s.strip())

    cloud = _build_cloud(args.subfield, work_types, args.snapshot_date)

    # Aligned, row-for-row: id[i] <-> references[i] <-> embeddings()[i].
    work_ids: List[str] = cloud.work_ids()
    work_refs: List[List[str]] = cloud.work_references()
    emb = np.asarray(cloud.embeddings(), dtype=np.float32)

    n = emb.shape[0]
    assert len(work_ids) == n == len(work_refs), (
        f"misaligned: ids={len(work_ids)} emb={n} refs={len(work_refs)}"
    )

    # Defensive L2-normalise so cosine == dot (CloudConfig already normalises,
    # but the contract of this measurement is cosine-on-unit-vectors).
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb = emb / norms

    # {work_id -> row} over the loaded sample; resolve each work's cited ids to
    # in-sample rows (drop self-cites and out-of-sample references).
    id_to_row: Dict[str, int] = {wid: i for i, wid in enumerate(work_ids)}
    cited_rows: List[NDArray[np.int64]] = []
    total_refs = 0
    total_in_sample = 0
    for i, refs in enumerate(work_refs):
        rows = []
        for rid in refs:
            total_refs += 1
            j = id_to_row.get(rid)
            if j is not None and j != i:
                rows.append(j)
        total_in_sample += len(rows)
        cited_rows.append(np.array(sorted(set(rows)), dtype=np.int64))

    n_with_cites = int(sum(1 for r in cited_rows if r.size >= 1))
    n_qualifying = int(sum(1 for r in cited_rows if r.size >= args.min_oper))

    rng = np.random.default_rng(args.seed)
    cp = coupling(emb, cited_rows, rng, args.n_works, args.min_oper, args.cap)

    print(f"[{args.label}] subfield={args.subfield} work_types={work_types} "
          f"snapshot={args.snapshot_date}")
    print(f"  corpus: {n} works, {total_refs} total references, "
          f"{total_in_sample} in-sample ({100*total_in_sample/max(total_refs,1):.1f}%); "
          f"{n_with_cites} works with >=1 in-sample cite, "
          f"{n_qualifying} with >={args.min_oper}")
    print(f"  coupling: citer−random cosine lift = {cp['lift_mean']:+.3f} "
          f"(median {cp['lift_median']:+.3f}, frac closer {cp['frac_closer']:.2f}, "
          f"n={cp['n']})")
    print(f"  raw cosine: to cited = {cp['raw_cos_cited']:+.3f}, "
          f"to random = {cp['raw_cos_random']:+.3f}")
    print()
    print("  reference (code substrates, citer−random cosine lift):")
    print("    Lean +0.115  Python +0.108  Go +0.037  Rust +0.057  Bazel +0.161")
    return 0


if __name__ == "__main__":
    sys.exit(main())
