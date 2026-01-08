import pytest

from bookgap.openalex_scopes import resolve_filter_upward
from bookgap.util import Filter


def test_resolve_filter_upward_derives_missing_parts():
    # Using known hierarchy: field=17, subfield=1711, topic=T10828
    cases = [
        (Filter(field="17"), ("17", None, None)),
        (Filter(subfield="1711"), ("17", "1711", None)),
        (Filter(topic="T10828"), ("17", "1711", "T10828")),
        (Filter(field="17", subfield="1711"), ("17", "1711", None)),
        (Filter(field="17", topic="T10828"), ("17", "1711", "T10828")),
        (Filter(subfield="1711", topic="T10828"), ("17", "1711", "T10828")),
        (Filter(field="17", subfield="1711", topic="T10828"), ("17", "1711", "T10828")),
    ]

    for work_filter, (exp_field, exp_subfield, exp_topic) in cases:
        resolved = resolve_filter_upward(work_filter)
        assert resolved.field == exp_field
        assert resolved.subfield == exp_subfield
        assert resolved.topic == exp_topic


def test_resolve_filter_upward_detects_inconsistencies():
    # Mismatched field with subfield
    with pytest.raises(ValueError):
        resolve_filter_upward(Filter(field="18", subfield="1711"))

    # Mismatched field with topic
    with pytest.raises(ValueError):
        resolve_filter_upward(Filter(field="18", topic="T10828"))

    # Mismatched subfield with topic (1710 != 1711)
    with pytest.raises(ValueError):
        resolve_filter_upward(Filter(subfield="1710", topic="T10828"))
