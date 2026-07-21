import itertools
import pytest
from bookgap.toc.records import read_book_records
from bookgap.toc.parse import derive_tocs

@pytest.mark.integration
def test_smoke_real_harvest_yields_plausible_tocs():
    # Read a bounded slice of real domain-3 book records and derive ToCs.
    recs = list(itertools.islice(read_book_records("2025-11-06"), 20000))
    tocs = derive_tocs(recs, "2025-11-06")
    assert tocs, "expected at least some ToCs from 20k real book records"
    for nt in tocs[:50]:
        assert len(nt.nodes) >= 3
        assert all(2 <= len(n.title) <= 100 for n in nt.nodes)
        assert nt.parser_profile in {"chapter-n", "arabic-list", "roman-list"}
