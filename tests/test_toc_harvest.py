import json
from pathlib import Path
from bookgap.toc.parse import NeutralToc
from bookgap.toc.profiles import TocNodeRaw
from bookgap.toc.harvest import write_corpus, build_argparser

def _nt(oid, title, prof="arabic-list"):
    return NeutralToc(openalex_id=f"https://openalex.org/{oid}", title=title,
                      primary_topic_id="T1", primary_subfield_id="17", language="en",
                      snapshot_date="2025-11-06", parser_profile=prof,
                      nodes=(TocNodeRaw(title="A"), TocNodeRaw(title="B"), TocNodeRaw(title="C")))

def test_write_corpus_writes_per_toc_json_and_manifest(tmp_path):
    manifest = write_corpus([_nt("W1", "Alpha"), _nt("W2", "Beta", prof="chapter-n")], tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == ["W1.json", "W2.json", "_manifest.json"]
    w1 = json.loads((tmp_path / "W1.json").read_text())
    assert w1["schema"] == "openalex-toc/v1" and w1["title"] == "Alpha"
    assert manifest["total"] == 2
    assert manifest["by_profile"] == {"arabic-list": 1, "chapter-n": 1}

def test_argparser_defaults():
    ns = build_argparser().parse_args(["--out", "/tmp/x"])
    assert ns.domain == "3" and ns.snapshot == "2025-11-06" and ns.language == "en"
    assert ns.profiles == "chapter-n,arabic-list,roman-list"
