"""Cloud-aware Filter subclasses.

Filter subclasses that need to inspect :class:`bookgap.cloud.Cloud`'s
column accessors live here so they can import ``Cloud`` directly. The
base :class:`bookgap.util.Filter` is intentionally Cloud-unaware to
keep the foundational module free of circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from bookgap.cloud import Cloud
from bookgap.util import Filter, WorkType


def _extract_year(date_str: Optional[str]) -> Optional[int]:
    """Extract year from an OpenAlex publication_date string.

    Accepts ``"YYYY-MM-DD"`` or ``"YYYY"``. Returns ``None`` for
    empty/None input; raises :class:`ValueError` for unparseable
    strings.
    """
    if not date_str:
        return None
    try:
        return int(date_str.split("-")[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid date string: {date_str}")


@dataclass(frozen=True)
class TemporalFilter(Filter):
    """Filter works by publication year and (optionally) work type.

    Articles and preprints are treated as interchangeable when
    ``work_type`` is set to either — matches OpenAlex's blurred line
    between the two for current-era research.
    """

    max_year: Optional[int] = None
    exact_year: Optional[int] = None
    work_type: Optional[WorkType] = None

    def check_work(self, work: dict[str, Any]) -> bool:
        if not super().check_work(work):
            return False

        pub_year = _extract_year(work.get("publication_date"))
        if pub_year is None:
            return False

        if self.max_year is not None and pub_year > self.max_year:
            return False
        if self.exact_year is not None and pub_year != self.exact_year:
            return False

        if self.work_type is not None:
            work_type_str = work.get("type", "").lower()
            target = (self.work_type.value if isinstance(self.work_type, WorkType)
                      else self.work_type)
            if target == "book" and work_type_str != "book":
                return False
            if target in ("article", "preprint") and work_type_str not in ("article", "preprint"):
                return False

        return True

    def compute_columnar_mask(self, cloud: Cloud) -> Optional[NDArray[np.bool_]]:
        """Vectorised fast path for SubCloud retention.

        Reads only ``publication_date`` and (optionally) ``type``
        columns from the parquet instead of materialising every work's
        full dict, then computes the retention mask with numpy.

        Returns ``None`` (delegating to the per-work ``check_work``
        path) when the parent :class:`Filter` constrains
        domain/field/subfield/topic/work_types — those columns are not
        yet vectorised here.
        """
        if (self.domain is not None or self.field is not None
                or self.subfield is not None or self.topic is not None
                or self.work_types is not None):
            return None

        pub_dates = cloud.work_publication_dates()
        years_int = np.array(
            [-1 if not d else (_extract_year(d) or -1) for d in pub_dates],
            dtype=np.int32,
        )
        mask = years_int != -1

        if self.max_year is not None:
            mask &= years_int <= self.max_year
        if self.exact_year is not None:
            mask &= years_int == self.exact_year

        if self.work_type is not None:
            target = (self.work_type.value if isinstance(self.work_type, WorkType)
                      else self.work_type)
            types_arr = np.array(cloud.work_types())
            types_lower = np.char.lower(types_arr.astype(str))
            if target == "book":
                mask &= types_lower == "book"
            elif target in ("article", "preprint"):
                mask &= (types_lower == "article") | (types_lower == "preprint")

        return mask
