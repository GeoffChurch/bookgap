"""Pure-local taxonomy resolution for OpenAlex filters.

All taxonomy lookups read from the snapshot-derived taxonomy at
cache/snapshot_taxonomy_<date>.json (built by bookgap.snapshot_ingest.run).
There is no live API path: scope resolution is reproducible across runs
given the same local cache.
"""
from typing import Any, Optional

from bookgap.snapshot_ingest import load_taxonomy
from bookgap.util import Filter


def _suffix(raw_id: str) -> str:
    return raw_id.rsplit("/", 1)[-1]


def resolve_filter_upward(work_filter, snapshot_date: str) -> Any:
    """Fill in missing hierarchy levels above the most specific component.

    Returns a new Filter with field/subfield/topic filled in consistently.
    Raises ValueError if provided values conflict with derived hierarchy
    or if any required entity is missing from the snapshot taxonomy.
    """
    domain_id: Optional[str] = work_filter.domain
    field_id: Optional[str] = work_filter.field
    subfield_id: Optional[str] = work_filter.subfield
    topic_id: Optional[str] = work_filter.topic

    tax = load_taxonomy(snapshot_date)

    if topic_id is not None:
        topic_data = tax.topics.get(topic_id)
        if topic_data is None:
            raise ValueError(f"Topic {topic_id} not in snapshot taxonomy ({snapshot_date})")
        resolved_subfield = _suffix(topic_data["subfield"]["id"])
        subfield_data = tax.subfields.get(resolved_subfield)
        if subfield_data is None:
            raise ValueError(f"Subfield {resolved_subfield} (parent of {topic_id}) not in snapshot taxonomy ({snapshot_date})")
        resolved_field = _suffix(subfield_data["field"]["id"])
        field_data = tax.fields.get(resolved_field)
        if field_data is None:
            raise ValueError(f"Field {resolved_field} (parent of {topic_id}) not in snapshot taxonomy ({snapshot_date})")
        resolved_domain = _suffix(field_data["domain"]["id"])

        if subfield_id is not None and subfield_id != resolved_subfield:
            raise ValueError(f"Provided subfield {subfield_id} does not match topic's subfield {resolved_subfield}")
        if field_id is not None and field_id != resolved_field:
            raise ValueError(f"Provided field {field_id} does not match topic's field {resolved_field}")
        if domain_id is not None and domain_id != resolved_domain:
            raise ValueError(f"Provided domain {domain_id} does not match topic's domain {resolved_domain}")

        return Filter(
            domain=resolved_domain,
            field=resolved_field,
            subfield=resolved_subfield,
            topic=topic_id,
            work_types=work_filter.work_types,
        )

    if subfield_id is not None:
        subfield_data = tax.subfields.get(subfield_id)
        if subfield_data is None:
            raise ValueError(f"Subfield {subfield_id} not in snapshot taxonomy ({snapshot_date})")
        resolved_field = _suffix(subfield_data["field"]["id"])
        field_data = tax.fields.get(resolved_field)
        if field_data is None:
            raise ValueError(f"Field {resolved_field} (parent of {subfield_id}) not in snapshot taxonomy ({snapshot_date})")
        resolved_domain = _suffix(field_data["domain"]["id"])

        if field_id is not None and field_id != resolved_field:
            raise ValueError(f"Provided field {field_id} does not match subfield's field {resolved_field}")
        if domain_id is not None and domain_id != resolved_domain:
            raise ValueError(f"Provided domain {domain_id} does not match subfield's domain {resolved_domain}")

        return Filter(
            domain=resolved_domain,
            field=resolved_field,
            subfield=subfield_id,
            topic=None,
            work_types=work_filter.work_types,
        )

    if field_id is not None:
        field_data = tax.fields.get(field_id)
        if field_data is None:
            raise ValueError(f"Field {field_id} not in snapshot taxonomy ({snapshot_date})")
        resolved_domain = _suffix(field_data["domain"]["id"])
        if domain_id is not None and domain_id != resolved_domain:
            raise ValueError(f"Provided domain {domain_id} does not match field's domain {resolved_domain}")

        return Filter(
            domain=resolved_domain,
            field=field_id,
            subfield=None,
            topic=None,
            work_types=work_filter.work_types,
        )

    if domain_id is not None:
        return Filter(
            domain=domain_id,
            field=None,
            subfield=None,
            topic=None,
            work_types=work_filter.work_types,
        )

    raise ValueError("At least one of domain, field, subfield, or topic must be provided")


def get_subfield_topics(subfield_id: str, snapshot_date: str) -> list[str]:
    """Return topic IDs for a given subfield, drawn from the snapshot taxonomy."""
    tax = load_taxonomy(snapshot_date)
    if subfield_id not in tax.subfields:
        raise ValueError(f"Subfield {subfield_id} not in snapshot taxonomy ({snapshot_date})")
    topic_ids = sorted({
        _suffix(t["id"])
        for t in tax.subfields[subfield_id].get("topics", [])
        if isinstance(t.get("id"), str)
    })
    if not topic_ids:
        raise ValueError(f"No topics recorded for subfield={subfield_id} at snapshot {snapshot_date}")
    return topic_ids


def get_field_topics(field_id: str, snapshot_date: str) -> list[str]:
    """Return all topic IDs for a given field by expanding its subfields."""
    tax = load_taxonomy(snapshot_date)
    if field_id not in tax.fields:
        raise ValueError(f"Field {field_id} not in snapshot taxonomy ({snapshot_date})")
    subfield_ids = sorted({
        _suffix(sf["id"])
        for sf in tax.fields[field_id].get("subfields", [])
        if isinstance(sf.get("id"), str)
    })
    if not subfield_ids:
        raise ValueError(f"No subfields recorded for field {field_id} at snapshot {snapshot_date}")
    return sorted({tid for subfield_id in subfield_ids for tid in get_subfield_topics(subfield_id, snapshot_date)})
