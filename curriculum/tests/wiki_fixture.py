"""Load the cached wiki subgraph as a plain :class:`InMemoryGraphSource`.

The parity gate uses ``cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz``
(a saved ``(graph, embeddings, ids, titles, abstracts)`` artifact) as a pure
GraphSource fixture — no bookgap import. This loader mirrors
``wiki_curriculum_demo.load_subgraph`` exactly, then wraps the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from curriculum.graph_source import InMemoryGraphSource

# Seed titles that anchor the NLP curriculum (== wiki_curriculum_demo.SEED_TITLES).
SEED_TITLES = [
    "Natural language processing",
    "Machine learning",
    "Language model",
    "Word embedding",
    "Parsing",
    "Computational linguistics",
]

DEFAULT_FIXTURE = Path(
    "cache/wiki/store/enwiki_full/curriculum_subgraph_n40000.npz"
)


def load_raw(path: Path = DEFAULT_FIXTURE) -> Tuple[
    sp.csr_matrix, List[str], List[str], np.ndarray
]:
    """Return ``(adj, titles, texts, embeddings)`` from the npz fixture.

    ``texts`` are ``"title. abstract"`` (matching the committed doc, which was
    generated with ``--use-abstracts``).
    """
    z = np.load(path, allow_pickle=True)
    adj = sp.csr_matrix(
        (z["adj_data"], z["adj_indices"], z["adj_indptr"]),
        shape=tuple(int(x) for x in z["adj_shape"]),
    ).tocsr()
    titles = [str(s) for s in z["titles"]]
    abstracts = [str(s) for s in z["abstracts"]]
    emb = np.ascontiguousarray(z["emb"], dtype=np.float32)
    n = len(titles)
    if any(abstracts):
        texts = [f"{titles[i]}. {abstracts[i]}".strip() for i in range(n)]
    else:
        texts = list(titles)
    return adj, titles, texts, emb


def load_wiki_graph_source(
    path: Path = DEFAULT_FIXTURE,
) -> Optional[InMemoryGraphSource]:
    """Wrap the fixture as a GraphSource, or ``None`` if the fixture is absent.

    Ids are the titles (the fixture's natural node identity); seeds are resolved
    from :data:`SEED_TITLES`.
    """
    if not path.exists():
        return None
    adj, titles, texts, emb = load_raw(path)
    title_to_idx = {t: i for i, t in enumerate(titles)}
    seeds = np.array(
        [title_to_idx[t] for t in SEED_TITLES if t in title_to_idx], dtype=np.int64
    )
    return InMemoryGraphSource(
        adj=adj, id_list=titles, text=texts, seeds=seeds, embeddings=emb,
    )
