from bookgap.toc.records import BookRecord
from bookgap.toc.parse import derive_tocs, NeutralToc

def _rec(oid, title, abstract, lang="en"):
    return BookRecord(openalex_id=oid, title=title, abstract=abstract,
                      primary_topic_id="T1", primary_subfield_id="17", language=lang)

_FIVE = ("1. Power Electronics 2. Power Diodes 3. Power Transistors "
         "4. Thyristor Devices 5. Rectifiers")

def test_derive_emits_neutral_toc_with_profile_and_nodes():
    out = derive_tocs([_rec("https://openalex.org/W1", "Power Electronics", _FIVE)], "2025-11-06")
    assert len(out) == 1
    nt = out[0]
    assert nt.parser_profile == "arabic-list"
    assert nt.snapshot_date == "2025-11-06"
    assert [n["title"] for n in nt.to_dict()["nodes"]][0] == "Power Electronics"
    assert nt.to_dict()["schema"] == "openalex-toc/v1"

def test_derive_drops_records_without_a_toc():
    assert derive_tocs([_rec("https://openalex.org/W9", "Prose", "A plain abstract.")], "s") == []

def test_derive_dedups_by_normalized_title_keeping_richest():
    short = "CHAPTER 1 Alpha CHAPTER 2 Beta CHAPTER 3 Gamma"   # 3 nodes (sane titles)
    rich = _FIVE                                            # 5 nodes, same title (punct/case differ)
    out = derive_tocs([
        _rec("https://openalex.org/W1", "Power: Electronics!", short),
        _rec("https://openalex.org/W2", "power electronics", rich),
    ], "s")
    assert len(out) == 1 and len(out[0].nodes) == 5
    assert out[0].openalex_id == "https://openalex.org/W2"
