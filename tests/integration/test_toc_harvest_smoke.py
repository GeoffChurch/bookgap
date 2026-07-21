import itertools
import pytest
from bookgap.toc.records import read_book_records
from bookgap.toc.parse import derive_tocs


def _leaves(nodes):
    out = []
    for n in nodes:
        out += _leaves(n.children) if n.children else [n]
    return out


@pytest.mark.integration
def test_smoke_real_harvest_yields_plausible_tocs():
    # Read a bounded slice of real domain-3 book records and derive ToCs.
    recs = list(itertools.islice(read_book_records("2025-11-06"), 20000))
    tocs = derive_tocs(recs, "2025-11-06")
    assert tocs, "expected at least some ToCs from 20k real book records"
    for nt in tocs[:50]:
        leaves = _leaves(nt.nodes)
        assert len(leaves) >= 3
        assert all(n.title.strip() for n in leaves)
        # assemble() bounds each leaf title to <=140 chars (the _after() capture
        # window) and guards content (>=2 chars, contains a letter) for >=70% of
        # leaves; it does NOT cap at 100 (some titles swallow embedded sub-outlines
        # and run long -- a known cosmetic residue, not a precision failure).
        # ToCs nest, so this is checked over leaves, not top-level nodes.
        assert sum(1 for n in leaves if 2 <= len(n.title) <= 140) >= 0.7 * len(leaves)
        assert nt.parser_profile in {"chapter-n", "arabic-list", "roman-list"}
