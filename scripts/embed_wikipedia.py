#!/usr/bin/env python
# scripts/embed_wikipedia.py
"""Build a Wikipedia leaf-corpus store in bookgap's GIST-small embedding space.

Reads the CirrusSearch content dump, materializes an articles parquet
``(page_id, title, abstract)``, embeds each article with GIST-small using the
EXACT same per-work text construction as OpenAlex ingest, and writes an
embeddings parquet aligned row-for-row with the articles, plus a provenance
``manifest.json``.

Shared-space contract
---------------------
bookgap embeds OpenAlex works by handing ``(display_name, abstract)`` pairs to
``SentenceTransformerEmbedder.embed_texts`` (bookgap/openalex_ingest.py:682),
which builds the model input as ``f"{title}\\n{abstract}"``
(bookgap/embedder.py:145). We reproduce that byte-for-byte here by passing
``(title, abstract)`` pairs through the same ``embed_texts`` call, so Wikipedia
vectors land in the identical space and can be unioned with OpenAlex corpora.

Store layout (under ``cache/wiki/store/<run_name>/``):
    articles.parquet    (page_id:int64, title:str, abstract:str)
    embeddings.parquet   row-aligned dense float32 (N, 384)
    manifest.json        provenance + embedder_fingerprint + schema version

Usage:
    python scripts/embed_wikipedia.py --out wiki_5k --limit 5000 --build-ts 2026-05-30
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bookgap.configs.embedder.embedder_config import SentenceTransformerEmbedderConfig
from bookgap.corpus import embedder_fingerprint
from bookgap.embedder import SentenceTransformerEmbedder
from bookgap.util import CACHE_DIR, LOGGER
from bookgap.wikipedia.ingest import write_articles_parquet

DEFAULT_DUMP = CACHE_DIR / "wiki" / "dumps" / "enwiki-20251229-cirrussearch-content.json.gz"
GIST_MODEL = "avsolatorio/GIST-small-Embedding-v0"
CIRRUS_DUMP_DATE = "20251229"
ID_SCHEME = "wiki:<page_id>"
EXTRACTION_METHOD = "cirrus opening_text (fallback: leading text)"
SCHEMA_VERSION = 1
# float32 embeddings; column name kept simple and self-describing.
EMBEDDINGS_SCHEMA_DIM_KEY = b"bookgap:embedding_dim"


def _source_digest(path: Path) -> dict:
    """Cheap-ish provenance for a possibly-huge (and partial) dump.

    sha256 over the whole 11+ GB file is wasteful during iteration, so we
    record size plus a sha256 of the leading 64 MiB (a stable fingerprint of
    the prefix we actually read for bounded slices).
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    prefix_bytes = 64 * 1024 * 1024
    read = 0
    with path.open("rb") as f:
        while read < prefix_bytes:
            chunk = f.read(min(1 << 20, prefix_bytes - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return {"size_bytes": size, "prefix_sha256_64MiB": h.hexdigest(), "prefix_bytes_hashed": read}


def embed_articles(articles_parquet: Path, batch_size: int = 256) -> np.ndarray:
    """Embed an articles parquet into an (N, D) float32 array, row-aligned.

    Uses the same ``embed_texts`` join as OpenAlex ingest so the vectors share
    the OpenAlex embedding space.
    """
    table = pq.read_table(articles_parquet, columns=["title", "abstract"])
    titles = table.column("title").to_pylist()
    abstracts = table.column("abstract").to_pylist()
    pairs = [(t or "", a or "") for t, a in zip(titles, abstracts)]
    n = len(pairs)
    LOGGER.info("Embedding %d Wikipedia articles with %s", n, GIST_MODEL)
    with SentenceTransformerEmbedder(model_name=GIST_MODEL) as embedder:
        dim = embedder.embedding_dimension()
        out = np.empty((n, dim), dtype=np.float32)
        for start in range(0, n, batch_size):
            chunk = pairs[start : start + batch_size]
            out[start : start + len(chunk)] = embedder.embed_texts(chunk, batch_size=batch_size)
            if start % (batch_size * 40) == 0:
                LOGGER.info("  embedded %d/%d", start + len(chunk), n)
    return out


def write_embeddings_parquet(embeddings: np.ndarray, out_path: Path) -> None:
    """Write an (N, D) float32 array as a parquet with a fixed-size list column."""
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n, d = embeddings.shape
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    fsl = pa.FixedSizeListArray.from_arrays(flat, d)
    table = pa.table({"embedding": fsl})
    table = table.replace_schema_metadata({EMBEDDINGS_SCHEMA_DIM_KEY: str(d).encode()})
    pq.write_table(table, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run_name; store goes to cache/wiki/store/<run_name>/")
    ap.add_argument("--limit", type=int, default=None, help="cap number of articles (bounded slice)")
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="path to cirrus content dump .json.gz")
    ap.add_argument("--build-ts", default="2026-05-30", help="build timestamp recorded in the manifest")
    ap.add_argument("--batch-size", type=int, default=256, help="embedding batch size")
    args = ap.parse_args()

    store_dir = CACHE_DIR / "wiki" / "store" / args.out
    store_dir.mkdir(parents=True, exist_ok=True)
    articles_path = store_dir / "articles.parquet"
    embeddings_path = store_dir / "embeddings.parquet"
    manifest_path = store_dir / "manifest.json"

    LOGGER.info("Materializing articles parquet -> %s (limit=%s)", articles_path, args.limit)
    n_articles = write_articles_parquet(args.dump, articles_path, limit=args.limit)
    LOGGER.info("Wrote %d articles", n_articles)

    embeddings = embed_articles(articles_path, batch_size=args.batch_size)
    assert embeddings.shape[0] == n_articles, (embeddings.shape, n_articles)
    write_embeddings_parquet(embeddings, embeddings_path)
    LOGGER.info("Wrote embeddings parquet %s shape=%s", embeddings_path, embeddings.shape)

    embedder_cfg = SentenceTransformerEmbedderConfig(model_name=GIST_MODEL)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.out,
        "n_articles": n_articles,
        "embedding_dim": int(embeddings.shape[1]),
        "source": {
            "cirrus_dump_filename": Path(args.dump).name,
            "cirrus_dump_date": CIRRUS_DUMP_DATE,
            **_source_digest(Path(args.dump)),
        },
        "embedder": {
            "model_name": GIST_MODEL,
            "embedder_fingerprint": embedder_fingerprint(embedder_cfg),
        },
        "id_scheme": ID_SCHEME,
        "extraction_method": EXTRACTION_METHOD,
        "text_construction": "embed_texts join: f'{title}\\n{abstract}' (matches OpenAlex)",
        "build_ts": args.build_ts,
        "limit": args.limit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    LOGGER.info("Wrote manifest %s", manifest_path)
    LOGGER.info(
        "DONE: store=%s articles=%d dim=%d fp=%s",
        store_dir, n_articles, embeddings.shape[1], manifest["embedder"]["embedder_fingerprint"],
    )


if __name__ == "__main__":
    main()
