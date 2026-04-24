import gzip
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bookgap.snapshot_ingest import (
    SnapshotState,
    load_state,
    save_state,
    check_disk_or_abort,
    DiskFullError,
    make_s3_client,
    list_shards,
    SpillSink,
    _normalize_snapshot_work,
    ScopeKey,
    run,
)


def test_load_state_missing_returns_empty(tmp_path):
    state = load_state(tmp_path / "missing.json", snapshot_date="2026-04-01")
    assert state.snapshot_date == "2026-04-01"
    assert state.shards_completed == []


def test_load_state_wipes_on_date_change(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"snapshot_date": "2026-03-01", "shards_completed": ["a", "b"]}))
    state = load_state(path, snapshot_date="2026-04-01")
    assert state.shards_completed == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = SnapshotState(snapshot_date="2026-04-01", shards_completed=["shard1.gz"])
    save_state(path, state)
    loaded = load_state(path, snapshot_date="2026-04-01")
    assert loaded.shards_completed == ["shard1.gz"]


def test_disk_guard_aborts_at_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 96
        free = 4
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    with pytest.raises(DiskFullError):
        check_disk_or_abort(tmp_path, abort_pct=95.0)


def test_disk_guard_aborts_exactly_at_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 95
        free = 5
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    with pytest.raises(DiskFullError):
        check_disk_or_abort(tmp_path, abort_pct=95.0)


def test_disk_guard_passes_under_threshold(tmp_path, monkeypatch):
    import shutil
    class FakeUsage:
        total = 100
        used = 50
        free = 50
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())
    check_disk_or_abort(tmp_path, abort_pct=95.0)  # no raise


# ---------------------------------------------------------------------------
# Helpers for S3 tests
# ---------------------------------------------------------------------------

class FakePaginator:
    """Mimics boto3 paginator for list_objects_v2."""

    def __init__(self, pages):
        # pages: list of dicts, each with optional "Contents" key
        self._pages = pages

    def paginate(self, Bucket, Prefix):
        return iter(self._pages)


class FakeClient:
    """Minimal boto3 client stub."""

    def __init__(self, pages):
        self._paginator = FakePaginator(pages)

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return self._paginator


# ---------------------------------------------------------------------------
# S3 client tests
# ---------------------------------------------------------------------------

def test_make_s3_client_returns_unsigned_client():
    """make_s3_client() must produce a boto3 client with UNSIGNED signature."""
    from botocore import UNSIGNED
    client = make_s3_client()
    assert client.meta.config.signature_version == UNSIGNED


# ---------------------------------------------------------------------------
# list_shards tests
# ---------------------------------------------------------------------------

def test_list_shards_filters_newer_than_snapshot():
    """Keys whose updated_date partition is after snapshot_date must be excluded."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-03-15/part_000.gz"},  # before -> keep
            {"Key": "data/works/updated_date=2026-04-02/part_001.gz"},  # after -> drop
            {"Key": "data/works/updated_date=2026-04-01/part_002.gz"},  # equal -> keep
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == [
        "data/works/updated_date=2026-03-15/part_000.gz",
        "data/works/updated_date=2026-04-01/part_002.gz",
    ]


def test_list_shards_filters_non_gz_objects():
    """Non-.gz objects (e.g. manifest files) must be excluded."""
    pages = [
        {"Contents": [
            {"Key": "data/works/manifest"},
            {"Key": "data/works/updated_date=2026-03-01/part_000.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == ["data/works/updated_date=2026-03-01/part_000.gz"]


def test_list_shards_returns_sorted_order():
    """Results must be sorted regardless of page order."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-03-20/part_002.gz"},
            {"Key": "data/works/updated_date=2026-03-01/part_000.gz"},
            {"Key": "data/works/updated_date=2026-03-10/part_001.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == sorted(result)
    assert len(result) == 3


def test_list_shards_empty_when_no_match():
    """Return an empty list when all keys are newer than snapshot_date."""
    pages = [
        {"Contents": [
            {"Key": "data/works/updated_date=2026-05-01/part_000.gz"},
            {"Key": "data/works/updated_date=2026-06-15/part_001.gz"},
        ]}
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == []


def test_list_shards_handles_empty_pages():
    """Pages with no Contents key must be silently skipped."""
    pages = [
        {},
        {"Contents": [{"Key": "data/works/updated_date=2026-03-01/part_000.gz"}]},
    ]
    client = FakeClient(pages)
    result = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert result == ["data/works/updated_date=2026-03-01/part_000.gz"]


def test_partition_date_from_key_rejects_empty_value():
    from bookgap.snapshot_ingest import _partition_date_from_key
    assert _partition_date_from_key("data/works/updated_date=/part.gz") is None


def test_partition_date_from_key_rejects_time_suffix():
    from bookgap.snapshot_ingest import _partition_date_from_key
    # Anything with a time component is not an exact YYYY-MM-DD date partition.
    assert _partition_date_from_key("data/works/updated_date=2026-04-01T12:00:00/part.gz") is None


def test_list_shards_drops_keys_with_malformed_partition():
    # A key with an empty or time-suffixed partition should never make it
    # into the final shard list, regardless of snapshot_date.
    client = FakeClient(pages=[{
        "Contents": [
            {"Key": "data/works/updated_date=/part.gz"},
            {"Key": "data/works/updated_date=2026-04-01T12:00/part.gz"},
            {"Key": "data/works/updated_date=2026-04-01/part.gz"},
        ]
    }])
    shards = list_shards(client, "openalex", "data/works/", snapshot_date="2026-04-01")
    assert shards == ["data/works/updated_date=2026-04-01/part.gz"]


# ---------------------------------------------------------------------------
# Helpers for stream_shard / Bucketer tests
# ---------------------------------------------------------------------------


def _make_gzipped_jsonl(works: list[dict]) -> io.BytesIO:
    """Build an in-memory gzipped JSONL blob from a list of work dicts."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for work in works:
            gz.write((json.dumps(work) + "\n").encode())
    buf.seek(0)
    return buf


def _work(field_id: str, subfield_id: str, topic_id: str, wtype: str = "article") -> dict:
    """Convenience builder for a minimal OpenAlex work dict."""
    return {
        "id": f"https://openalex.org/W{topic_id}999",
        "type": wtype,
        "primary_topic": {
            "id": f"https://openalex.org/T{topic_id}",
            "field": {"id": f"https://openalex.org/fields/{field_id}"},
            "subfield": {"id": f"https://openalex.org/subfields/{subfield_id}"},
        },
    }


class RecordingSink:
    def __init__(self):
        self.calls = []  # list of (scope_key, list[dict])

    def __call__(self, scope_key, rows):
        self.calls.append((scope_key, list(rows)))


# ---------------------------------------------------------------------------
# stream_shard tests
# ---------------------------------------------------------------------------

def test_stream_shard_yields_scope_keyed_works():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        _work("31", "2100", "10805"),          # field 31 -> match
        _work("31", "2100", "10806"),          # field 31 -> match
        _work("99", "9900", "99001"),          # field 99 -> no match
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields={"31"}))

    assert len(results) == 2
    for key, work in results:
        field, subfield, topic, wtype = key
        assert field == "31"
        assert subfield == "2100"
        assert isinstance(topic, str) and topic
        assert wtype == "article"


def test_stream_shard_drops_works_without_primary_topic():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        {"id": "https://openalex.org/W1", "type": "article"},          # no primary_topic
        _work("31", "2100", "10805"),
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields=None))

    assert len(results) == 1
    assert results[0][0][0] == "31"


def test_stream_shard_drops_malformed_json_lines():
    from bookgap.snapshot_ingest import stream_shard

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"not json at all\n")
        gz.write((json.dumps(_work("31", "2100", "10805")) + "\n").encode())
    buf.seek(0)

    results = list(stream_shard(buf, target_fields=None))
    assert len(results) == 1


def test_stream_shard_no_target_fields_yields_all():
    from bookgap.snapshot_ingest import stream_shard

    works = [
        _work("31", "2100", "10805"),
        _work("55", "3300", "20001"),
        _work("77", "4400", "30002", wtype="book"),
    ]
    blob = _make_gzipped_jsonl(works)
    results = list(stream_shard(blob, target_fields=None))

    assert len(results) == 3
    field_ids = {key[0] for key, _ in results}
    assert field_ids == {"31", "55", "77"}


def test_stream_shard_prefilter_patterns_drop_unmatched_lines():
    """Bytes prefilter should skip lines that don't contain any anchor
    pattern, while still yielding works whose lines do contain a pattern.
    Result set with prefilter must be a subset of the result set without."""
    from bookgap.snapshot_ingest import stream_shard

    works = [
        _work("31", "3103", "10805"),     # physics — match
        _work("55", "3300", "20001"),     # some other field — no match
        _work("31", "3107", "11234"),     # physics — match
        _work("27", "2700", "30002"),     # medicine — no match
    ]
    blob1 = _make_gzipped_jsonl(works)

    # Without prefilter: target_fields still filters to field 31 (2 works).
    baseline = list(stream_shard(blob1, target_fields={"31"}))
    assert {k[0] for k, _ in baseline} == {"31"}
    assert len(baseline) == 2

    # With a prefilter that matches the raw bytes of field 31 URLs, result
    # set must be identical — the prefilter is advisory, not additive.
    blob2 = _make_gzipped_jsonl(works)
    prefiltered = list(
        stream_shard(
            blob2,
            target_fields={"31"},
            prefilter_patterns=[b'/fields/31"'],
        )
    )
    assert len(prefiltered) == len(baseline)
    assert {k[0] for k, _ in prefiltered} == {"31"}

    # Prefilter with a pattern that matches nothing should yield zero works
    # (still correct, since it proves the prefilter can filter at the byte level).
    blob3 = _make_gzipped_jsonl(works)
    empty = list(
        stream_shard(
            blob3,
            target_fields={"31"},
            prefilter_patterns=[b'/fields/99999"'],
        )
    )
    assert empty == []


# ---------------------------------------------------------------------------
# Bucketer tests
# ---------------------------------------------------------------------------

def test_bucketer_flushes_when_threshold_reached():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    threshold = 5
    bucketer = Bucketer(sink=sink, flush_threshold=threshold, max_resident=10_000)

    key = ("31", "2100", "10805", "article")
    work = _work("31", "2100", "10805")

    for _ in range(threshold):
        bucketer.insert(key, work)

    # Exactly at threshold -> one flush with all threshold rows
    assert len(sink.calls) == 1
    assert len(sink.calls[0][1]) == threshold
    assert sink.calls[0][0] == key


def test_bucketer_evicts_largest_bucket_when_resident_cap_hit():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    # max_resident = 5; threshold high so per-bucket flush won't fire
    bucketer = Bucketer(sink=sink, flush_threshold=1_000, max_resident=5)

    key_a = ("31", "2100", "10805", "article")
    key_b = ("55", "3300", "20001", "article")
    work_a = _work("31", "2100", "10805")
    work_b = _work("55", "3300", "20001")

    # Insert 3 into A, 2 into B (total = 5, exactly at cap — one more triggers eviction)
    for _ in range(3):
        bucketer.insert(key_a, work_a)
    for _ in range(2):
        bucketer.insert(key_b, work_b)

    # No flush yet (resident == max_resident, not exceeding)
    assert len(sink.calls) == 0

    # One more into B brings total to 6 > max_resident -> largest (A, size 3) flushed
    bucketer.insert(key_b, work_b)
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == key_a
    assert len(sink.calls[0][1]) == 3


def test_bucketer_flush_all_empties_all_buckets():
    from bookgap.snapshot_ingest import Bucketer

    sink = RecordingSink()
    bucketer = Bucketer(sink=sink, flush_threshold=1_000, max_resident=1_000)

    key_a = ("31", "2100", "10805", "article")
    key_b = ("55", "3300", "20001", "book")
    work_a = _work("31", "2100", "10805")
    work_b = _work("55", "3300", "20001", wtype="book")

    bucketer.insert(key_a, work_a)
    bucketer.insert(key_a, work_a)
    bucketer.insert(key_b, work_b)

    bucketer.flush_all()

    # Both buckets flushed; sink called twice
    assert len(sink.calls) == 2
    flushed_keys = {call[0] for call in sink.calls}
    assert flushed_keys == {key_a, key_b}

    # Totals match what was inserted
    total_rows = sum(len(call[1]) for call in sink.calls)
    assert total_rows == 3

    # Subsequent flush_all is a no-op
    sink.calls.clear()
    bucketer.flush_all()
    assert len(sink.calls) == 0


# ---------------------------------------------------------------------------
# _normalize_snapshot_work tests (Part B)
# ---------------------------------------------------------------------------


def _snapshot_work_raw(
    field_id="31",
    subfield_id="3102",
    topic_id="T99999",
    wtype="article",
    abstract_inverted_index=None,
    openalex_id="https://openalex.org/W12345",
) -> dict:
    """Build a raw snapshot JSON work record."""
    return {
        "ids": {"openalex": openalex_id, "mag": "99"},
        "type": wtype,
        "display_name": "A Great Paper",
        "abstract_inverted_index": abstract_inverted_index or {"Hello": [0], "world": [1]},
        "primary_topic": {
            "id": f"https://openalex.org/{topic_id}",
            "field": {"id": f"https://openalex.org/fields/{field_id}"},
            "subfield": {"id": f"https://openalex.org/subfields/{subfield_id}"},
        },
    }


def test_normalize_snapshot_work_happy_path():
    scope_key: ScopeKey = ("31", "3102", "T99999", "article")
    raw = _snapshot_work_raw()
    result = _normalize_snapshot_work(raw, scope_key)

    assert result is not None
    # abstract decoded from inverted index
    assert result["abstract"] == "Hello world"
    assert "abstract_inverted_index" not in result
    # id extracted from ids.openalex
    assert result["id"] == "https://openalex.org/W12345"
    assert "ids" not in result
    # scope labels set
    assert result["topic"] == "T99999"
    assert result["field_id"] == "31"
    assert result["subfield_id"] == "3102"
    assert result["topic_id"] == "T99999"


def test_normalize_snapshot_work_drops_no_abstract():
    scope_key: ScopeKey = ("31", "3102", "T99999", "article")
    raw = _snapshot_work_raw(abstract_inverted_index=None)
    raw.pop("abstract_inverted_index")
    assert _normalize_snapshot_work(raw, scope_key) is None


def test_normalize_snapshot_work_drops_empty_abstract():
    scope_key: ScopeKey = ("31", "3102", "T99999", "article")
    raw = _snapshot_work_raw()
    raw["abstract_inverted_index"] = {}  # falsy
    assert _normalize_snapshot_work(raw, scope_key) is None


# ---------------------------------------------------------------------------
# SpillSink tests (Part C)
# ---------------------------------------------------------------------------


def _normalized_work(work_id: str, **overrides) -> dict:
    """Build a minimal normalized work dict matching SCOPE_PARQUET_SCHEMA."""
    base = {
        "id": work_id,
        "display_name": "Test Work",
        "abstract": "Some abstract text",
        "type": "article",
        "publication_date": "2024-01-15",
        "created_date": "2024-01-15",
        "updated_date": "2024-01-15",
        "language": "en",
        "topic": "T10805",
        "primary_topic": {"id": "https://openalex.org/T10805", "subfield": {"id": "https://openalex.org/subfields/3102"}},
        "referenced_works": [],
    }
    base.update(overrides)
    return base


def test_spill_sink_writes_one_file_per_scope(tmp_path):
    sink = SpillSink(tmp_path / "spill")
    key_a: ScopeKey = ("31", "3102", "T99999", "article")
    key_b: ScopeKey = ("17", "1702", "T10181", "book")
    sink(key_a, [_normalized_work("W1"), _normalized_work("W2")])
    sink(key_b, [_normalized_work("W10", type="book")])

    path_a = sink._spill_path(key_a, 0)
    path_b = sink._spill_path(key_b, 0)
    assert path_a.exists()
    assert path_b.exists()

    table_a = pq.read_table(path_a)
    table_b = pq.read_table(path_b)
    assert table_a.num_rows == 2
    assert table_b.num_rows == 1
    assert table_a.column("id").to_pylist() == ["W1", "W2"]
    assert table_b.column("id").to_pylist() == ["W10"]


def test_spill_sink_writes_one_shard_per_call(tmp_path):
    sink = SpillSink(tmp_path / "spill")
    key: ScopeKey = ("31", "3102", "T99999", "article")
    sink(key, [_normalized_work("W1"), _normalized_work("W2")])
    sink(key, [_normalized_work("W3")])

    paths = sink._shards_for(key)
    assert len(paths) == 2
    combined = pa.concat_tables([pq.read_table(p) for p in paths])
    assert combined.column("id").to_pylist() == ["W1", "W2", "W3"]


def test_spill_sink_discovers_existing_shards_on_init(tmp_path):
    """SpillSink must pre-populate _files_written and _next_seq from existing shards.

    Covers resume: a prior interrupted run wrote shards but never finalized;
    a fresh SpillSink must pick them up AND continue numbering correctly.
    """
    spill_dir = tmp_path / "spill"
    key: ScopeKey = ("31", "3102", "T99999", "article")

    scope_dir = spill_dir / "31" / "3102" / "T99999"
    scope_dir.mkdir(parents=True, exist_ok=True)
    # Write two synthetic prior shards at sequence 0 and 1.
    from bookgap.openalex_ingest import works_to_table
    table = works_to_table([_normalized_work("W1")])
    pq.write_table(table, scope_dir / "article_0000.parquet")
    pq.write_table(table, scope_dir / "article_0001.parquet")

    sink = SpillSink(spill_dir)
    assert key in sink._files_written
    # Next write should land at sequence 2 (0 and 1 are taken).
    sink(key, [_normalized_work("W2")])
    assert (scope_dir / "article_0002.parquet").exists()


# ---------------------------------------------------------------------------
# _normalize_snapshot_work malformed abstract test
# ---------------------------------------------------------------------------


def test_normalize_snapshot_work_handles_malformed_abstract(tmp_path):
    """_normalize_snapshot_work must return None (not raise) on a malformed abstract.

    resolve_abstract_inverted_index asserts that words contain no spaces.
    A word with a space in the inverted index triggers AssertionError; we confirm
    _normalize_snapshot_work catches it and returns None gracefully.
    """
    scope_key: ScopeKey = ("31", "3102", "T99999", "article")
    raw = _snapshot_work_raw(
        abstract_inverted_index={"hello world": [0]},  # space in word -> AssertionError
    )
    result = _normalize_snapshot_work(raw, scope_key)
    assert result is None


# ---------------------------------------------------------------------------
# end-to-end run() test (Part D + E)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ingest(tmp_path, monkeypatch):
    """Redirect CACHE_DIR and ARTIFACTS_DIR to tmp_path for ingest modules."""
    import bookgap.openalex_ingest as oi_mod
    import bookgap.snapshot_ingest as si_mod
    import bookgap.util as util_mod

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()

    monkeypatch.setattr(oi_mod, "CACHE_DIR", cache)
    monkeypatch.setattr(oi_mod, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(si_mod, "CACHE_DIR", cache)
    monkeypatch.setattr(util_mod, "CACHE_DIR", cache)
    monkeypatch.setattr(util_mod, "ARTIFACTS_DIR", artifacts)

    return tmp_path


def _make_gzipped_jsonl_bytes(works: list[dict]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for work in works:
            gz.write((json.dumps(work) + "\n").encode())
    return buf.getvalue()


class FakeS3Client:
    """Minimal boto3 client stub for end-to-end test."""

    def __init__(self, shard_key: str, shard_bytes: bytes):
        self._shard_key = shard_key
        self._shard_bytes = shard_bytes

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"

        class _Paginator:
            def paginate(self_, Bucket, Prefix):
                return iter([{
                    "Contents": [{"Key": self._shard_key}]
                }])

        return _Paginator()

    def download_file(self, bucket, key, local_path):
        Path(local_path).write_bytes(self._shard_bytes)


def test_run_entrypoint_end_to_end(tmp_ingest, monkeypatch):
    """Full pipeline: shard → stream → bucket → spill → filter → dedup → parquet."""
    import bookgap.openalex_ingest as oi_mod
    from bookgap.openalex_ingest import ResolvedScope
    from bookgap.util import Filter

    # The scope we'll target
    FIELD = "31"
    SUBFIELD = "3102"
    TOPIC = "T99999"
    WTYPE = "article"
    scope_key: ScopeKey = (FIELD, SUBFIELD, TOPIC, WTYPE)

    # Monkeypatch _resolve_filter_scopes to return the single concrete scope
    fake_scope = ResolvedScope(field=FIELD, subfield=SUBFIELD, topic=TOPIC, work_type=WTYPE)
    monkeypatch.setattr(oi_mod, "_resolve_filter_scopes", lambda f, snapshot_date: [fake_scope])

    # Build a synthetic shard: one matching work, one in a different field
    matching_work = {
        "ids": {"openalex": "https://openalex.org/W99001"},
        "type": WTYPE,
        "display_name": "Quantum Computing Fundamentals",
        "abstract_inverted_index": {"Quantum": [0], "computing": [1], "is": [2], "interesting": [3]},
        "primary_topic": {
            "id": f"https://openalex.org/{TOPIC}",
            "field": {"id": f"https://openalex.org/fields/{FIELD}"},
            "subfield": {"id": f"https://openalex.org/subfields/{SUBFIELD}"},
        },
        "language": "en",
        "publication_date": "2024-01-15",
        "created_date": "2024-01-15",
        "updated_date": "2024-01-15",
        "referenced_works": [],
    }
    other_field_work = {
        "ids": {"openalex": "https://openalex.org/W99002"},
        "type": "article",
        "display_name": "Economics Paper",
        "abstract_inverted_index": {"Economics": [0], "stuff": [1]},
        "primary_topic": {
            "id": "https://openalex.org/T11111",
            "field": {"id": "https://openalex.org/fields/99"},
            "subfield": {"id": "https://openalex.org/subfields/9901"},
        },
        "language": "en",
        "publication_date": "2024-01-15",
        "created_date": "2024-01-15",
        "updated_date": "2024-01-15",
        "referenced_works": [],
    }

    shard_key = f"data/works/updated_date=2026-04-01/part_000.gz"
    shard_bytes = _make_gzipped_jsonl_bytes([matching_work, other_field_work])

    fake_client = FakeS3Client(shard_key=shard_key, shard_bytes=shard_bytes)

    cache_dir = tmp_ingest / "cache"
    state_path = cache_dir / "snapshot_state.json"
    spill_dir = cache_dir / "snapshot_tmp"

    run(
        work_filter=Filter(field=FIELD, topic=TOPIC, work_types=(WTYPE,)),
        snapshot_date="2026-04-01",
        state_path=state_path,
        spill_dir=spill_dir,
        s3_client=fake_client,
        shard_row_flush_threshold=100,
        max_resident_rows=10_000,
    )

    # State file should have finished_at set and the shard in shards_completed
    state = load_state(state_path, snapshot_date="2026-04-01")
    assert state.finished_at is not None
    assert shard_key in state.shards_completed

    # Parquet file should exist and be readable
    import bookgap.openalex_ingest as oi_mod
    scope = ResolvedScope(field=FIELD, subfield=SUBFIELD, topic=TOPIC, work_type=WTYPE)
    parquet_path = oi_mod._scope_parquet_path(scope)
    assert parquet_path.exists(), f"Expected parquet at {parquet_path}"

    table = pq.read_table(parquet_path)
    assert table.num_rows >= 1

    # The target work must be present; the off-target work must NOT be present.
    ids = table.column("id").to_pylist()
    assert "https://openalex.org/W99001" in ids, "Target work missing from parquet"
    assert "https://openalex.org/W99002" not in ids, "Off-target work must not appear in parquet"

    # Parquet schema must include all expected columns.
    expected_columns = {
        "id", "display_name", "abstract", "type",
        "publication_date", "created_date", "updated_date",
        "language", "topic_label", "primary_topic_id",
        "primary_subfield_id", "referenced_works",
    }
    assert expected_columns <= set(table.schema.names), (
        f"Missing columns: {expected_columns - set(table.schema.names)}"
    )

    # Per-scope artifact files must exist.
    artifacts_dir = tmp_ingest / "artifacts" / str(scope)
    assert (artifacts_dir / "invalid_works.txt").exists(), (
        f"invalid_works.txt not found under {artifacts_dir}"
    )
    assert (artifacts_dir / "invalid_works_stats.json").exists(), (
        f"invalid_works_stats.json not found under {artifacts_dir}"
    )


def test_run_skips_completed_shards(tmp_ingest, monkeypatch):
    """If a shard is already in shards_completed, run() must skip it."""
    import bookgap.openalex_ingest as oi_mod
    from bookgap.openalex_ingest import ResolvedScope
    from bookgap.util import Filter

    FIELD = "31"
    fake_scope = ResolvedScope(field=FIELD, subfield="3102", topic="T99999", work_type="article")
    monkeypatch.setattr(oi_mod, "_resolve_filter_scopes", lambda f, snapshot_date: [fake_scope])

    shard_key = "data/works/updated_date=2026-04-01/part_000.gz"

    # Prime the state with the shard already completed
    cache_dir = tmp_ingest / "cache"
    state_path = cache_dir / "snapshot_state.json"
    state = SnapshotState(
        snapshot_date="2026-04-01",
        shards_completed=[shard_key],
        started_at="2026-04-01T00:00:00+00:00",
    )
    save_state(state_path, state)

    downloads: list[str] = []

    class TrackingClient:
        def get_paginator(self, op):
            class _P:
                def paginate(self_, Bucket, Prefix):
                    return iter([{"Contents": [{"Key": shard_key}]}])
            return _P()

        def download_file(self, bucket, key, local_path):
            downloads.append(key)
            # Write an empty gzip so the file is valid if accidentally opened
            Path(local_path).write_bytes(_make_gzipped_jsonl_bytes([]))

    spill_dir = cache_dir / "snapshot_tmp"
    run(
        work_filter=Filter(field=FIELD, work_types=("article",)),
        snapshot_date="2026-04-01",
        state_path=state_path,
        spill_dir=spill_dir,
        s3_client=TrackingClient(),
    )

    # The shard was already done — download_file must never have been called
    assert downloads == [], f"Expected no downloads, got {downloads}"


def test_run_raises_when_work_types_is_none(tmp_path, monkeypatch):
    """run() must raise ValueError with a helpful message when work_filter.work_types is None."""
    from bookgap import snapshot_ingest
    from bookgap.util import Filter
    import shutil

    class FakeUsage:
        total = 100
        used = 10
        free = 90
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())

    with pytest.raises(ValueError, match="work_types must be provided"):
        snapshot_ingest.run(
            work_filter=Filter(domain="3"),  # work_types defaults to None
            snapshot_date="2026-04-01",
            state_path=tmp_path / "state.json",
            spill_dir=tmp_path / "spill",
            s3_client=None,  # never reached
        )


def test_run_accepts_list_of_filters_and_unions_scopes(tmp_path, monkeypatch):
    """run() should accept a list of filters and union their resolved scopes,
    so one streaming pass can cover multiple non-contiguous fields."""
    from bookgap import snapshot_ingest
    import bookgap.openalex_ingest
    from bookgap.openalex_ingest import ResolvedScope
    from bookgap.util import Filter
    import shutil

    class FakeUsage:
        total = 100
        used = 10
        free = 90
    monkeypatch.setattr(shutil, "disk_usage", lambda p: FakeUsage())

    resolved_by_field = {
        "23": [
            ResolvedScope(field="23", subfield="2301", topic="T11111", work_type="article"),
            ResolvedScope(field="23", subfield="2301", topic="T11111", work_type="book"),
        ],
        "25": [
            ResolvedScope(field="25", subfield="2500", topic="T22222", work_type="article"),
        ],
        "26": [
            ResolvedScope(field="26", subfield="2600", topic="T33333", work_type="preprint"),
        ],
    }

    def fake_resolve(f, snapshot_date):
        return resolved_by_field[f.field]

    monkeypatch.setattr(bookgap.openalex_ingest, "_resolve_filter_scopes", fake_resolve)

    # Record what list_shards returns (we don't actually download anything).
    monkeypatch.setattr(snapshot_ingest, "list_shards", lambda *a, **kw: [])

    captured = {}
    orig_make_client = snapshot_ingest.make_s3_client
    monkeypatch.setattr(snapshot_ingest, "make_s3_client", lambda: object())

    # Patch LOGGER.info to capture the target_fields/scopes summary.
    import bookgap.util
    infos: list = []
    monkeypatch.setattr(bookgap.util.LOGGER, "info", lambda *a, **kw: infos.append((a, kw)))

    snapshot_ingest.run(
        work_filter=[
            Filter(field="23", work_types=("article", "book")),
            Filter(field="25", work_types=("article",)),
            Filter(field="26", work_types=("preprint",)),
        ],
        snapshot_date="2025-11-06",
        state_path=tmp_path / "state.json",
        spill_dir=tmp_path / "spill",
    )

    # The info log should mention all three filters by their string form.
    summary_line = next(
        (a for a, _ in infos if "filter(s)" in a[0] and "filters=" in a[0]),
        None,
    )
    assert summary_line is not None, f"no 'filter(s)' summary in {infos}"
    # a[1] is the filter count
    assert summary_line[1] == 3
    # a[3] is the list of filter str() representations
    filter_strs = summary_line[3]
    assert any("field=23" in s for s in filter_strs)
    assert any("field=25" in s for s in filter_strs)
    assert any("field=26" in s for s in filter_strs)


# ---------------------------------------------------------------------------
# run_worker snapshot dispatch test (Task 6)
# ---------------------------------------------------------------------------


def test_run_worker_snapshot_action_dispatches_correctly(tmp_path, monkeypatch):
    """run_worker('snapshot', ...) should build a Filter and invoke snapshot_ingest.run."""
    import resource

    import bookgap.scope_ops
    import bookgap.snapshot_ingest

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(bookgap.snapshot_ingest, "run", fake_run)

    # Patch manifest IO so we don't touch disk
    import bookgap.jobs
    monkeypatch.setattr(bookgap.jobs, "read_manifest", lambda _: {"pid": 0, "action": "snapshot"})
    monkeypatch.setattr(bookgap.jobs, "write_manifest", lambda *_: None)

    # Disable RLIMIT_AS — it's a process-wide side effect unsafe for pytest.
    monkeypatch.setattr(resource, "setrlimit", lambda *a, **kw: None)

    bookgap.scope_ops.run_worker(
        action="snapshot",
        scope_str="31/3102/T10805/article",
        model=None,
        job_id="test_job",
        work_types=None,
        snapshot_date="2026-04-01",
    )

    assert called["snapshot_date"] == "2026-04-01"
    assert called["work_filter"].field == "31"
    assert called["work_filter"].subfield == "3102"
    assert called["work_filter"].topic == "T10805"
    assert called["work_filter"].work_types == ("article",)


def test_run_worker_snapshot_3part_scope_builds_topic_filter(monkeypatch):
    """3-part scope (field/subfield/topic) should build topic-level filter."""
    import resource

    import bookgap.scope_ops
    import bookgap.snapshot_ingest

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(bookgap.snapshot_ingest, "run", fake_run)
    import bookgap.jobs
    monkeypatch.setattr(bookgap.jobs, "read_manifest", lambda _: {"pid": 0, "action": "snapshot"})
    monkeypatch.setattr(bookgap.jobs, "write_manifest", lambda *_: None)
    monkeypatch.setattr(resource, "setrlimit", lambda *a, **kw: None)

    bookgap.scope_ops.run_worker(
        action="snapshot",
        scope_str="31/3102/T10805",
        model=None,
        job_id="test_job",
        work_types=["article"],
        snapshot_date="2026-04-01",
    )

    f = called["work_filter"]
    assert f.field == "31"
    assert f.subfield == "3102"
    assert f.topic == "T10805"
    assert f.work_types == ("article",)


def test_run_worker_snapshot_2part_scope_builds_subfield_filter(monkeypatch):
    """2-part scope (field/subfield) should build subfield-level filter."""
    import resource

    import bookgap.scope_ops
    import bookgap.snapshot_ingest

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(bookgap.snapshot_ingest, "run", fake_run)
    import bookgap.jobs
    monkeypatch.setattr(bookgap.jobs, "read_manifest", lambda _: {"pid": 0, "action": "snapshot"})
    monkeypatch.setattr(bookgap.jobs, "write_manifest", lambda *_: None)
    monkeypatch.setattr(resource, "setrlimit", lambda *a, **kw: None)

    bookgap.scope_ops.run_worker(
        action="snapshot",
        scope_str="31/3102",
        model=None,
        job_id="test_job",
        work_types=["article", "book"],
        snapshot_date="2026-04-01",
    )

    f = called["work_filter"]
    assert f.field == "31"
    assert f.subfield == "3102"
    assert f.topic is None
    assert set(f.work_types) == {"article", "book"}


def test_run_worker_snapshot_1part_domain_scope_builds_domain_filter(monkeypatch):
    """1-part scope (domain) should build domain-level filter."""
    import resource

    import bookgap.scope_ops
    import bookgap.snapshot_ingest

    called = {}

    def fake_run(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(bookgap.snapshot_ingest, "run", fake_run)
    import bookgap.jobs
    monkeypatch.setattr(bookgap.jobs, "read_manifest", lambda _: {"pid": 0, "action": "snapshot"})
    monkeypatch.setattr(bookgap.jobs, "write_manifest", lambda *_: None)
    monkeypatch.setattr(resource, "setrlimit", lambda *a, **kw: None)

    bookgap.scope_ops.run_worker(
        action="snapshot",
        scope_str="3",
        model=None,
        job_id="test_job",
        work_types=["article"],
        snapshot_date="2026-04-01",
    )

    f = called["work_filter"]
    assert f.domain == "3"
    assert f.field is None
    assert f.work_types == ("article",)

# ---------------------------------------------------------------------------
# Derived taxonomy + reproducibility tests
# ---------------------------------------------------------------------------

def test_load_taxonomy_raises_when_missing(tmp_path, monkeypatch):
    """load_taxonomy must error clearly if snapshot_ingest hasn't run for this date."""
    import bookgap.snapshot_ingest as si
    monkeypatch.setattr(si, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_TAXONOMY_CACHE", {})
    with pytest.raises(FileNotFoundError, match="Derived taxonomy missing"):
        si.load_taxonomy("2025-11-06")


def test_load_taxonomy_reads_derived_file(tmp_path, monkeypatch):
    """load_taxonomy should parse the per-snapshot taxonomy JSON and return it."""
    import bookgap.snapshot_ingest as si
    monkeypatch.setattr(si, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_TAXONOMY_CACHE", {})
    (tmp_path / "snapshot_taxonomy_2025-11-06.json").write_text(json.dumps({
        "snapshot_date": "2025-11-06",
        "fields": {"31": {"display_name": "Physics"}},
        "subfields": {"3102": {"display_name": "X"}},
        "topics": {"T10805": {"display_name": "Y"}},
    }))
    tax = si.load_taxonomy("2025-11-06")
    assert tax.snapshot_date == "2025-11-06"
    assert tax.fields["31"]["display_name"] == "Physics"
    assert tax.subfields["3102"]["display_name"] == "X"
    assert tax.topics["T10805"]["display_name"] == "Y"


def test_load_taxonomy_rejects_date_mismatch(tmp_path, monkeypatch):
    """load_taxonomy must refuse a file whose snapshot_date does not match the request."""
    import bookgap.snapshot_ingest as si
    monkeypatch.setattr(si, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_TAXONOMY_CACHE", {})
    (tmp_path / "snapshot_taxonomy_2025-11-06.json").write_text(json.dumps({
        "snapshot_date": "2026-01-01",  # wrong!
        "fields": {}, "subfields": {}, "topics": {},
    }))
    with pytest.raises(ValueError, match="has snapshot_date"):
        si.load_taxonomy("2025-11-06")


def test_taxonomy_accumulator_builds_nested_hierarchy():
    """_TaxonomyAccumulator.observe should thread primary_topic into a nested dict."""
    from bookgap.snapshot_ingest import _TaxonomyAccumulator
    acc = _TaxonomyAccumulator()
    acc.observe({
        "primary_topic": {
            "id": "https://openalex.org/T10805",
            "display_name": "Y",
            "subfield": {"id": "https://openalex.org/subfields/3102", "display_name": "X"},
            "field": {"id": "https://openalex.org/fields/31", "display_name": "Physics"},
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        },
    })
    out = acc.emit()
    assert "31" in out["fields"]
    assert out["fields"]["31"]["display_name"] == "Physics"
    assert any(sf["id"].endswith("/3102") for sf in out["fields"]["31"]["subfields"])
    assert "3102" in out["subfields"]
    assert any(t["id"].endswith("/T10805") for t in out["subfields"]["3102"]["topics"])
    assert "T10805" in out["topics"]


def test_corpus_dataset_raises_when_parquet_missing(tmp_path, monkeypatch):
    """corpus_dataset must raise MissingScopeParquetError instead of falling back to API."""
    import bookgap.openalex_ingest as oi
    import bookgap.snapshot_ingest as si
    from bookgap.util import Filter

    monkeypatch.setattr(oi, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_TAXONOMY_CACHE", {})
    # Prime a derived taxonomy covering one topic.
    (tmp_path / "snapshot_taxonomy_2025-11-06.json").write_text(json.dumps({
        "snapshot_date": "2025-11-06",
        "fields": {"31": {
            "id": "https://openalex.org/fields/31",
            "display_name": "Physics",
            "domain": {"id": "https://openalex.org/domains/3"},
            "subfields": [{"id": "https://openalex.org/subfields/3102"}],
        }},
        "subfields": {"3102": {
            "id": "https://openalex.org/subfields/3102",
            "display_name": "X",
            "field": {"id": "https://openalex.org/fields/31"},
            "topics": [{"id": "https://openalex.org/T10805"}],
        }},
        "topics": {"T10805": {
            "id": "https://openalex.org/T10805",
            "display_name": "Y",
            "subfield": {"id": "https://openalex.org/subfields/3102"},
        }},
    }))
    oi.corpus_dataset.cache_clear()
    f = Filter(field="31", work_types=("article",))
    with pytest.raises(oi.MissingScopeParquetError, match="missing for filter"):
        oi.corpus_dataset(f, "2025-11-06")
