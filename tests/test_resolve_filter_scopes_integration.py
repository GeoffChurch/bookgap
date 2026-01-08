from bookgap.cloud import _resolve_filter_scopes
from bookgap.util import Filter, WorkType


def test_resolve_filter_scopes_live_counts():
    cases = [
        (Filter(field="17"), 302),
        (Filter(subfield="1711"), 10),
        (Filter(topic="T10828"), 1),
        (Filter(field="17", subfield="1711"), 10),
        (Filter(field="17", topic="T10828"), 1),
        (Filter(subfield="1711", topic="T10828"), 1),
        (Filter(field="17", subfield="1711", topic="T10828"), 1),
    ]

    # Add BOOK and ARTICLE work types to each filter
    work_types = (str(WorkType.BOOK), str(WorkType.ARTICLE))
    cases = [(
        Filter(
            field=filt.field,
            subfield=filt.subfield,
            topic=filt.topic,
            work_types=work_types),
        expected * len(work_types))
        for filt, expected in cases]

    for filt, expected in cases:
        scopes = _resolve_filter_scopes(filt)
        assert len(scopes) == expected
