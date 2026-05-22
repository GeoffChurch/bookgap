"""Structured Hydra configs for filters.

Wraps ``bookgap.util.Filter`` and ``FilterPair`` with ``_target_`` values so
nested instantiation produces the concrete dataclass instances.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from bookgap.util import Domain, WorkType


@dataclass
class FilterConfig:
    """Hydra structured config for :class:`bookgap.util.Filter`.

    Not frozen so Hydra CLI overrides (e.g. ``cloud.filter_pair.pre.subfield=1702``)
    can mutate fields. Hashability lives on the runtime :class:`bookgap.util.Filter`
    that Hydra instantiates from this config, not on the config wrapper.
    """

    _target_: str = field(default="bookgap.util.Filter", init=False)
    domain: Optional[str] = None
    field: Optional[str] = None
    subfield: Optional[str] = None
    topic: Optional[str] = None
    work_types: Optional[Tuple[str, ...]] = None


@dataclass
class FilterPairConfig:
    """Hydra structured config for :class:`bookgap.util.FilterPair`.

    Not frozen — see :class:`FilterConfig` for rationale.
    """

    _target_: str = field(default="bookgap.util.FilterPair", init=False)
    pre: FilterConfig = field(
        default_factory=lambda: FilterConfig(
            domain=Domain.PHYSICAL_SCIENCES.value,
            # field=Field.COMP_SCI.value,
            # subfield=SubField.AI.value,
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
