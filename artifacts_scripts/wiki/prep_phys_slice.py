"""Materialize a Physical-Sciences slice (emb + title + abstract) as an npz for the
wiki-gap v2 detector — the CROSS-FIELD dissociation test.

NLP came up 0 gaps (Wikipedia documents it exhaustively). Hypothesis: an applied
Phys-Sci field (e.g. Materials Science) dissociates in its research long-tail —
there are millions of specific materials/compounds/processing techniques and
Wikipedia has articles for a tiny fraction.

Reads the already-ingested OpenAlex works_parquet (abstracts are reconstructed
text), filters English articles, subsamples, and embeds with GIST-small directly
(no 60 GB cloud machinery).

Regenerate:
    uv run \
        python -m artifacts_scripts.wiki.prep_phys_slice --scope 25 --n 150000 \
        --out cache/wiki/oa_phys25_slice.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="25", help="works_parquet scope: field '25' or 'field/subfield' e.g. '17/1705'")
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--min-abstract", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    # Defaulted rather than required: this writes a ~200MB npz, and a careless
    # invocation should not be able to drop that in the current directory.
    ap.add_argument("--out", type=Path,
                    default=Path("cache/wiki_spike/phys_slice.npz"))
    args = ap.parse_args()

    d = ds.dataset(f"cache/works_parquet/{args.scope}")
    filt = (pc.field("type") == "article") & (pc.field("language") == "en")
    print(f"reading scope {args.scope} (en articles) ...", flush=True)
    t = d.to_table(columns=["display_name", "abstract"], filter=filt)
    titles = t.column("display_name").to_pylist()
    absts = t.column("abstract").to_pylist()
    pairs = [(ti, ab) for ti, ab in zip(titles, absts)
             if ti and ab and len(ab) >= args.min_abstract]
    print(f"{len(pairs):,} en articles w/ abstract (>= {args.min_abstract} chars)", flush=True)
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.n:
        idx = rng.choice(len(pairs), args.n, replace=False)
        pairs = [pairs[i] for i in idx.tolist()]
    titles = [p[0] for p in pairs]
    absts = [p[1] for p in pairs]

    print(f"embedding {len(titles):,} works (GIST-small) ...", flush=True)
    from bookgap.embedder import SentenceTransformerEmbedder
    with SentenceTransformerEmbedder(SentenceTransformerEmbedder.ModelName.GIST_SMALL) as emb:
        vecs = emb.embed_texts(list(zip(titles, absts)), batch_size=args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        arts_emb=np.ascontiguousarray(vecs, dtype=np.float32),
        arts_titles=np.array(titles, dtype=object),
        arts_abstracts=np.array(absts, dtype=object),
    )
    print(f"wrote {args.out}: emb={vecs.shape}", flush=True)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# PROVENANCE: regenerate (do not hand-edit the npz):
#   uv run \
#       python -m artifacts_scripts.wiki.prep_phys_slice --scope 25 --n 150000 \
#       --out cache/wiki/oa_phys25_slice.npz
# Reads cache/works_parquet/<scope> (ingested OpenAlex, reconstructed abstracts),
# en-article filter + subsample, GIST-small embed. Feeds wiki_gap_spike2.py via
# --slice-npz for the cross-field Wikipedia-gap test.
# ---------------------------------------------------------------------------
