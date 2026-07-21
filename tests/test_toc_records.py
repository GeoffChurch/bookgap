from bookgap.toc.records import BookRecord, _records_from_rows

_ROWS = [
    {"id": "https://openalex.org/W1", "display_name": "Real Book", "abstract": "1. A 2. B",
     "primary_topic_id": "T1", "primary_subfield_id": "17", "language": "en"},
    {"id": "https://openalex.org/W2", "display_name": "No Abstract", "abstract": None,
     "primary_topic_id": "T1", "primary_subfield_id": "17", "language": "en"},
    {"id": "https://openalex.org/W3", "display_name": "French Book", "abstract": "1. A",
     "primary_topic_id": "T1", "primary_subfield_id": "17", "language": "fr"},
]

def test_records_from_rows_drops_empty_abstract_and_filters_language():
    recs = list(_records_from_rows(_ROWS, language="en"))
    assert [r.openalex_id for r in recs] == ["https://openalex.org/W1"]  # W2 empty, W3 non-en
    assert recs[0].title == "Real Book"

def test_records_from_rows_language_none_keeps_all_nonempty():
    recs = list(_records_from_rows(_ROWS, language=None))
    assert [r.openalex_id for r in recs] == ["https://openalex.org/W1", "https://openalex.org/W3"]
