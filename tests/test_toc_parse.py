from bookgap.toc.records import BookRecord
from bookgap.toc.profiles import TocNodeRaw
from bookgap.toc.parse import derive_tocs, NeutralToc, _leaf_count

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

def test_leaf_count_counts_nested_children_not_top_level_nodes():
    nested = (
        TocNodeRaw(title="Part One", kind="part", children=(
            TocNodeRaw(title="Intro"), TocNodeRaw(title="Middle"), TocNodeRaw(title="End"),
        )),
    )
    flat = (TocNodeRaw(title="Alpha"), TocNodeRaw(title="Beta"))
    assert len(nested) == 1 and len(flat) == 2  # top-level count would (wrongly) favor flat
    assert _leaf_count(nested) == 3
    assert _leaf_count(flat) == 2
    assert _leaf_count(nested) > _leaf_count(flat)

_NESTED = "PART ONE: Basics CHAPTER 1 Intro CHAPTER 2 Middle CHAPTER 3 End PART TWO: Advanced"
_FLAT3 = "CHAPTER 1 Xray CHAPTER 2 Yankee CHAPTER 3 Zulu"

def test_derive_dedup_keeps_nested_parse_with_more_leaves_over_flatter_top_level():
    # W1: flat parse with 3 top-level chapter nodes (3 leaves).
    # W2: nested parse with only 2 top-level part nodes but 4 leaves total
    #     (3 chapters under PART ONE + the childless PART TWO leaf).
    # Top-level node count would wrongly favor W1 (3 > 2); leaf count must favor W2 (4 > 3).
    out = derive_tocs([
        _rec("https://openalex.org/W1", "Basics and Advanced", _FLAT3),
        _rec("https://openalex.org/W2", "basics and advanced", _NESTED),
    ], "s")
    assert len(out) == 1
    nt = out[0]
    assert nt.openalex_id == "https://openalex.org/W2"
    assert len(nt.nodes) == 2  # top-level count is fewer than the flat parse's 3
    assert nt.nodes[0].kind == "part" and len(nt.nodes[0].children) == 3
