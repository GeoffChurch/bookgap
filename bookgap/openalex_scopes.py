from typing import Any, Optional

import pyalex

from bookgap.util import Filter


def _suffix(raw_id: str) -> str:
    return raw_id.rsplit("/", 1)[-1]


def resolve_filter_upward(work_filter) -> Any:
    """Fill in missing hierarchy levels above the most specific component.

    Returns a new Filter with field/subfield/topic filled in consistently.
    Raises ValueError if provided values conflict with derived hierarchy.
    """
    field_id: Optional[str] = work_filter.field
    subfield_id: Optional[str] = work_filter.subfield
    topic_id: Optional[str] = work_filter.topic

    if topic_id is not None:
        # Derive field and subfield from topic
        topic_data: Any = pyalex.Topics()[topic_id]  # type: ignore
        resolved_subfield = _suffix(topic_data["subfield"]["id"])
        subfield_data: Any = pyalex.Subfields()[resolved_subfield]  # type: ignore
        resolved_field = _suffix(subfield_data["field"]["id"])

        # Validate consistency
        if subfield_id is not None and subfield_id != resolved_subfield:
            raise ValueError(f"Provided subfield {subfield_id} does not match topic's subfield {resolved_subfield}")
        if field_id is not None and field_id != resolved_field:
            raise ValueError(f"Provided field {field_id} does not match topic's field {resolved_field}")
        
        return Filter(
            field=resolved_field,
            subfield=resolved_subfield,
            topic=topic_id,
            work_types=work_filter.work_types,
        )

    if subfield_id is not None:
        # Derive field from subfield
        subfield_data: Any = pyalex.Subfields()[subfield_id]  # type: ignore
        resolved_field = _suffix(subfield_data["field"]["id"])
        
        # Validate consistency
        if field_id is not None and field_id != resolved_field:
            raise ValueError(f"Provided field {field_id} does not match subfield's field {resolved_field}")
        
        return Filter(
            field=resolved_field,
            subfield=subfield_id,
            topic=None,
            work_types=work_filter.work_types,
        )

    if field_id is None:
        raise ValueError("At least one of field, subfield, or topic must be provided")

    # Field-only: nothing to resolve upward
    return work_filter


def get_subfield_topics(subfield_id: str) -> list[str]:
    """Return topic IDs for a given subfield."""
    filters: dict[str, Any] = {"subfield.id": subfield_id}

    topic_ids: list[str] = []
    for page in pyalex.Topics().filter(**filters).paginate(per_page=200):
        for topic in page:
            tid = topic.get("id")
            if isinstance(tid, str):
                topic_ids.append(tid.rsplit("/", 1)[-1])
    topic_ids = sorted(set(topic_ids))
    if not topic_ids:
        raise ValueError(f"No topics found for subfield={subfield_id}")
    return topic_ids


def get_field_topics(field_id: str) -> list[str]:
    """Return all topic IDs for a given field by expanding its subfields."""
    field_data: Any = pyalex.Fields()[field_id]  # type: ignore
    subfield_ids: list[str] = [_suffix(sf["id"]) for sf in field_data.get("subfields", [])]
    subfield_ids = sorted(set(subfield_ids))
    if not subfield_ids:
        raise ValueError(f"No subfields found for field {field_id}")

    all_topics: list[str] = []
    for subfield_id in subfield_ids:
        all_topics.extend(get_subfield_topics(subfield_id))
    return sorted(set(all_topics))
