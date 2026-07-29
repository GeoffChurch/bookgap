"""Scope resolution against the pinned snapshot taxonomy.

Previously this test called ``_resolve_filter_scopes(filt)`` with no snapshot
and was named ``..._live_counts``: it exercised the live-OpenAlex backfill that
commit dba74f6 ("pin taxonomy and work ingest to a snapshot date, kill
live-API backfill") removed. It had been failing since April with
``TypeError: missing a required argument: 'snapshot_date'``.

The per-scope counts below are unchanged from the live-API era -- the
2025-11-06 snapshot taxonomy reproduces them exactly (302 topics in field 17,
10 in subfield 1711, 1 for topic T10828, each times the two work types). They
are now stable rather than dependent on whatever OpenAlex returns today.
"""
from pathlib import Path

import pytest

from bookgap.cloud import _resolve_filter_scopes
from bookgap.util import CACHE_DIR, Filter, WorkType

# The snapshot this repo has ingested. Resolution reads
# cache/snapshot_taxonomy_<date>.json, written by bookgap.snapshot_ingest.run.
SNAPSHOT_DATE = "2025-11-06"

WORK_TYPES = (str(WorkType.BOOK), str(WorkType.ARTICLE))

# (filter kwargs, expected number of resolved topics before the work-type fan-out)
CASES = [
    (dict(field="17"), 302),
    (dict(subfield="1711"), 10),
    (dict(topic="T10828"), 1),
    (dict(field="17", subfield="1711"), 10),
    (dict(field="17", topic="T10828"), 1),
    (dict(subfield="1711", topic="T10828"), 1),
    (dict(field="17", subfield="1711", topic="T10828"), 1),
]


def _taxonomy_path() -> Path:
    return CACHE_DIR / f"snapshot_taxonomy_{SNAPSHOT_DATE}.json"


pytestmark = pytest.mark.skipif(
    not _taxonomy_path().exists(),
    reason=(
        f"no snapshot taxonomy at {_taxonomy_path()}; "
        "run bookgap.snapshot_ingest to build one"
    ),
)


@pytest.mark.parametrize("filter_kwargs,expected_topics", CASES)
def test_resolve_filter_scopes_snapshot_counts(filter_kwargs, expected_topics):
    """Each filter resolves to one scope per (topic, work_type) pair."""
    filt = Filter(work_types=WORK_TYPES, **filter_kwargs)
    scopes = _resolve_filter_scopes(filt, SNAPSHOT_DATE)
    assert len(scopes) == expected_topics * len(WORK_TYPES)


def test_narrowing_a_filter_never_widens_the_scope_set():
    """field > subfield > topic must be monotonically narrowing.

    Guards the resolution order independently of the hardcoded counts above,
    so a taxonomy refresh that changes the numbers still fails loudly if the
    containment relationship breaks.
    """
    by_field = _resolve_filter_scopes(
        Filter(field="17", work_types=WORK_TYPES), SNAPSHOT_DATE)
    by_subfield = _resolve_filter_scopes(
        Filter(field="17", subfield="1711", work_types=WORK_TYPES), SNAPSHOT_DATE)
    by_topic = _resolve_filter_scopes(
        Filter(field="17", subfield="1711", topic="T10828", work_types=WORK_TYPES),
        SNAPSHOT_DATE)

    assert set(by_topic) <= set(by_subfield) <= set(by_field)
    assert len(by_topic) == len(WORK_TYPES)
