"""Structured Hydra configs for filters.

Wraps ``bookgap.util.Filter`` and ``FilterPair`` with ``_target_`` values so
nested instantiation produces the concrete dataclass instances.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from bookgap.util import Field, SubField, WorkType


@dataclass(frozen=True)
class FilterConfig:
    """Hydra structured config for :class:`bookgap.util.Filter`."""

    _target_: str = field(default="bookgap.util.Filter", init=False)
    field: Optional[str] = None
    subfield: Optional[str] = None
    topic: Optional[str] = None
    work_types: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class FilterPairConfig:
    """Hydra structured config for :class:`bookgap.util.FilterPair`."""

    _target_: str = field(default="bookgap.util.FilterPair", init=False)
    pre: FilterConfig = field(
        default_factory=lambda: FilterConfig(
            field=Field.COMP_SCI.value,
            subfield=SubField.AI.value,
            work_types=(
                WorkType.BOOK.value,
                WorkType.ARTICLE.value,
                WorkType.BOOK_CHAPTER.value,
                WorkType.PREPRINT.value,
            ),
        )
    )
    post: FilterConfig = field(default_factory=FilterConfig)


__all__ = ["FilterConfig", "FilterPairConfig"]
