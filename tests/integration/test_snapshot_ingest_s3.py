"""
Live S3 integration tests for snapshot_ingest.

These tests hit the real OpenAlex public S3 bucket (no credentials required)
to verify that our unsigned client, list_shards, and _partition_date_from_key
actually work against production infrastructure.

Run with:
    uv run \
        pytest tests/integration/test_snapshot_ingest_s3.py -m integration -v

The expensive download test (test 4) additionally requires:
    BOOKGAP_RUN_EXPENSIVE_INTEGRATION=1
"""

import os

import pytest


@pytest.mark.integration
def test_list_shards_live_returns_non_empty():
    """Smoke test: the OpenAlex snapshot bucket actually contains shard files
    under data/works/ and our list_shards helper returns at least a few of them."""
    from bookgap.snapshot_ingest import list_shards, make_s3_client

    client = make_s3_client()
    # Use a deliberately far-future snapshot_date so we get all available shards.
    shards = list_shards(client, "openalex", "data/works/", snapshot_date="2099-12-31")
    assert len(shards) > 0, f"Expected at least one shard key; got {len(shards)}"
    assert all(s.endswith(".gz") for s in shards), (
        f"Expected all keys to end with .gz; got sample: {shards[:5]}"
    )


@pytest.mark.integration
def test_real_keys_have_parseable_partition_dates():
    """Every shard key we get back should parse to a YYYY-MM-DD partition date."""
    from bookgap.snapshot_ingest import (
        _partition_date_from_key,
        list_shards,
        make_s3_client,
    )

    client = make_s3_client()
    shards = list_shards(client, "openalex", "data/works/", snapshot_date="2099-12-31")
    # Sample up to 10 keys — no need to check every one
    sample = shards[:10]
    for key in sample:
        parsed = _partition_date_from_key(key)
        assert parsed is not None, f"Could not parse partition date from key: {key!r}"
        assert len(parsed) == 10, f"Unexpected partition format: {parsed!r} in {key!r}"


@pytest.mark.integration
def test_list_shards_excludes_partitions_newer_than_snapshot_date():
    """Passing an old snapshot_date should yield fewer shards than a future date."""
    from bookgap.snapshot_ingest import list_shards, make_s3_client

    client = make_s3_client()
    # 2020-01-01 is before OpenAlex started publishing snapshots, so
    # old_shards should be empty (or at worst much smaller than all_shards).
    old_date = "2020-01-01"
    all_shards = list_shards(client, "openalex", "data/works/", snapshot_date="2099-12-31")
    old_shards = list_shards(client, "openalex", "data/works/", snapshot_date=old_date)
    assert len(old_shards) < len(all_shards), (
        f"Expected old_shards < all_shards; got {len(old_shards)} vs {len(all_shards)}"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("BOOKGAP_RUN_EXPENSIVE_INTEGRATION"),
    reason="Set BOOKGAP_RUN_EXPENSIVE_INTEGRATION=1 to enable (downloads a real shard, possibly ~2 GB)",
)
def test_download_and_stream_one_shard(tmp_path):
    """End-to-end smoke: pull one real shard, stream it, confirm at least one
    parsable work comes out.  This is the only test that actually exercises the
    snapshot's JSONL format matching our expectations.

    WARNING: shards can be 1-2 GB each.  Check ContentLength before committing
    to a full download; we log it so you can bail if bandwidth is a concern.
    """
    from bookgap.snapshot_ingest import list_shards, make_s3_client, stream_shard

    client = make_s3_client()
    shards = list_shards(client, "openalex", "data/works/", snapshot_date="2099-12-31")
    assert shards, "Need at least one shard for the download test"

    # Take the first shard (sorted, so deterministic).
    shard_key = shards[0]

    # Log the object size before downloading so the operator can abort if needed.
    head = client.head_object(Bucket="openalex", Key=shard_key)
    content_length = head.get("ContentLength", 0)
    size_mb = content_length / (1024 * 1024)
    print(f"\n  Shard: {shard_key}")
    print(f"  Size:  {size_mb:.1f} MB")

    tmp_file = tmp_path / "shard.gz"
    client.download_file("openalex", shard_key, str(tmp_file))

    n_works = 0
    with tmp_file.open("rb") as f:
        for scope_key, work in stream_shard(f, target_fields=None):
            n_works += 1
            if n_works == 1:
                # Sanity-check the first work has the shape we expect.
                assert "primary_topic" in work, (
                    f"First work missing 'primary_topic' key; keys: {list(work.keys())}"
                )
                assert "id" in work or "ids" in work, (
                    f"First work has neither 'id' nor 'ids'; keys: {list(work.keys())}"
                )
            if n_works >= 5:
                break  # don't waste bandwidth reading the whole shard
    assert n_works > 0, f"Expected at least one streamable work in shard {shard_key!r}"
