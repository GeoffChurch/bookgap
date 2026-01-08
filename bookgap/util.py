from dataclasses import dataclass
import enum
import logging
from functools import wraps
import inspect
import re
from typing import Any, Optional, Sequence
from pathlib import Path
import time

import joblib
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import normalize as sk_normalize

class Timer:
    def __init__(self, print=print):
        self.print = print
    def time(self, formatters=None):
        if formatters is None:
            formatters = {}
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kw):
                bound = inspect.signature(f).bind(*args, **kw)
                bound.apply_defaults()
                filtered = {k: (formatters[k](v) if k in formatters else v) for k, v in bound.arguments.items()}
                filtered_s = ', '.join(f'{k}={v!r}' for k, v in filtered.items())
                self.print(f'Starting func:{f.__name__}({filtered_s})')
                start_time = time.time()
                result = f(*args, **kw)
                elapsed_time = time.time() - start_time
                # self.print(f'func:{f.__name__}({filtered_s}) took {elapsed_time:.2f} seconds')
                self.print(f'Took {elapsed_time:.2f} seconds for func:{f.__name__}({filtered_s})')
                return result
            return wrap
        return decorator


LOGGER = logging.getLogger("bookgap")

TIMER = Timer(print=LOGGER.info)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY = joblib.Memory(CACHE_DIR / "joblib", verbose=0)


class Field(enum.Enum):
    COMP_SCI = "17"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)


class SubField(enum.Enum):
    AI = "1702"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)

class Topic(enum.Enum):
    NLP = "T10181"
    ROBOTICS = "T10191"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)

class WorkType(str, enum.Enum):
    BOOK    = "book"
    BOOK_CHAPTER = "book-chapter"
    ARTICLE = "article"
    PREPRINT = "preprint"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)

@dataclass(frozen=True)
class Filter:
    field: Optional[str] = None
    subfield: Optional[str] = None
    topic: Optional[str] = None
    # Multiple work types are supported; callers must pass a tuple[str, ...]
    work_types: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        # Enforce that work_types is either None or a tuple[str, ...]
        if self.work_types is not None:
            object.__setattr__(self, "work_types", tuple(self.work_types)) # Hydra will pass a ListConfig
            if not all(isinstance(wt, str) for wt in self.work_types):
                raise TypeError("Each work type must be a string")

    def to_dict(self) -> dict[str, str | list[str]]:
        ret: dict[str, str | list[str]] = {}
        if self.field is not None:
            ret["primary_topic.field.id"] = self.field
        if self.topic is not None:
            ret["primary_topic.id"] = self.topic
        if self.subfield is not None:
            ret["primary_topic.subfield.id"] = self.subfield
        if self.work_types is not None:
            # pyalex expects a list when using OR; we'll apply filter_or in get_query
            ret["type"] = list(self.work_types)
        return ret
    
    def check_work(self, work: dict[str, Any]) -> bool:
        """Check if the work matches this filter."""
        if self.field is not None and work["primary_topic"]["field"]["id"] != f"https://openalex.org/fields/{self.field}":
            return False
        if self.topic is not None and work["primary_topic"]["id"] != f"https://openalex.org/{self.topic}":
            return False
        if self.subfield is not None and work["primary_topic"]["subfield"]["id"] != f"https://openalex.org/subfields/{self.subfield}":
            return False
        if self.work_types is not None:
            wtype = work.get("type")
            if wtype not in self.work_types:
                return False
        return True
    
    def __str__(self) -> str:
        parts = []
        if self.field is not None:
            parts.append(f"field={self.field}")
        if self.topic is not None:
            parts.append(f"topic={self.topic}")
        if self.subfield is not None:
            parts.append(f"subfield={self.subfield}")
        if self.work_types is not None:
            wt = self.work_types
            if len(wt) == 1:
                parts.append(f"work_type={wt[0]}")
            else:
                parts.append(f"work_types=[{','.join(wt)}]")
        return ", ".join(parts) if parts else "no filter"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __bool__(self) -> bool:
        return (
            self.field is not None
            or self.subfield is not None
            or self.topic is not None
            or self.work_types is not None
        )

@dataclass(frozen=True)
class FilterPair:
    pre: Filter
    post: Filter

    def __post_init__(self):
        # Allow subset relationship for work types
        if self.pre.work_types is not None and self.post.work_types is not None:
            pre_set = set(self.pre.work_types)
            post_set = set(self.post.work_types)
            if not post_set.issubset(pre_set):
                raise ValueError("Post-filter work types must be a subset of pre-filter work types")
        if self.pre.field is not None and self.post.field is not None:
            if self.pre.field != self.post.field:
                raise ValueError("Pre-filter and post-filter fields do not match")
        if self.pre.subfield is not None and self.post.subfield is not None:
            if self.pre.subfield != self.post.subfield:
                raise ValueError("Pre-filter and post-filter subfields do not match")
        if self.pre.topic is not None and self.post.topic is not None:
            if self.pre.topic != self.post.topic:
                raise ValueError("Pre-filter and post-filter topics do not match")
            
    def merged(self) -> Filter:
        """Return the merged (most specific) filter combining pre and post constraints.
        
        The merged filter represents the GLB (greatest lower bound) of pre and post,
        preferring post values when both are specified (since post narrows pre).
        """
        # Post overrides pre for hierarchy fields (more specific)
        field = self.post.field if self.post.field is not None else self.pre.field
        subfield = self.post.subfield if self.post.subfield is not None else self.pre.subfield
        topic = self.post.topic if self.post.topic is not None else self.pre.topic
        
        # For work_types, use post (subset) if specified, else pre
        work_types = self.post.work_types if self.post.work_types is not None else self.pre.work_types
        
        return Filter(
            field=field,
            subfield=subfield,
            topic=topic,
            work_types=work_types,
        )
    
    def merged_name(self) -> str:
        """Return a merged name for the filter pair."""
        parts: dict[str, str | list[str]] = {}
        if self.pre.field is not None:
            parts["field"] = self.pre.field
        if self.pre.topic is not None:
            parts["topic"] = self.pre.topic
        if self.pre.subfield is not None:
            parts["subfield"] = self.pre.subfield
        if self.pre.work_types is not None:
            parts["work_types"] = list(self.pre.work_types)
        if self.post.field is not None:
            parts["post_field"] = self.post.field
        if self.post.topic is not None:
            parts["post_topic"] = self.post.topic
        if self.post.subfield is not None:
            parts["post_subfield"] = self.post.subfield
        if self.post.work_types is not None:
            parts["post_work_types"] = list(self.post.work_types)
        return "(" + ", ".join(f"{k}={v}" for k, v in parts.items()) + ")"
            
class Filters(enum.Enum):
    # ROBOTICS = (TopicPreFilter(topic="T10191", subfield=None), lambda _: True)
    # NLP = (TopicFilter(topic=None, subfield="1702"), lambda work: work["primary_topic"]["id"] == "T10181")
    NLP = FilterPair(
        Filter(subfield=SubField.AI.value, work_types=(
            WorkType.BOOK.value,
            WorkType.ARTICLE.value,
            WorkType.BOOK_CHAPTER.value,
            WorkType.PREPRINT.value,
        )),
        Filter(topic=Topic.NLP.value),
    )
    # AI = (TopicPreFilter(topic=None, subfield="1702"), lambda _: True)

    # def work_type(self) -> str:
    #     """Return the single work type for this filter if uniquely defined; else raise."""
    #     pre = set(self.value.pre.work_types or [])
    #     post = set(self.value.post.work_types or [])
    #     s = pre or post
    #     if len(s) == 1:
    #         ret, = s
    #         return ret
    #     raise ValueError("No single work type defined for this filter")

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)

    def _validate(self):
        if self.value.pre.work_types is not None and self.value.post.work_types is not None:
            if not set(self.value.post.work_types).issubset(set(self.value.pre.work_types)):
                raise ValueError("Post-filter work types must be a subset of pre-filter work types")
        if self.value.pre.field is not None and self.value.post.field is not None:
            if self.value.pre.field != self.value.post.field:
                raise ValueError("Pre-filter and post-filter fields do not match")
        if self.value.pre.subfield is not None and self.value.post.subfield is not None:
            if self.value.pre.subfield != self.value.post.subfield:
                raise ValueError("Pre-filter and post-filter subfields do not match")
        if self.value.pre.topic is not None and self.value.post.topic is not None:
            if self.value.pre.topic != self.value.post.topic:
                raise ValueError("Pre-filter and post-filter topics do not match")

def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _\.\-=\(\),"\']', ' ', str(s)).rstrip()

def setup_logging(
    log_name: str = "",
    log_level: str = "INFO",
    include_time: bool = True,
    explicit_args: Optional[list[tuple[str, Any]]] = None,
    log_path: Optional[Path] = None,
):
    LOGGER.setLevel(log_level)
    
    found_handlers = LOGGER.hasHandlers()
    # Remove all handlers associated with the logger
    if found_handlers:
        LOGGER.warning("Clearing existing handlers from LOGGER")
        LOGGER.handlers.clear()

    if log_path is None:
        parts = []
        if include_time:
            parts.append(time.strftime('%Y-%m-%d_%H-%M-%S'))
        if log_name:
            parts.append(log_name)
        if explicit_args:
            parts.append(create_unicode_safe_filename("_".join(f"{k}={v}" for k, v in sorted(explicit_args))))
        if not parts:
            raise ValueError("At least one part must be provided for the log name")
        log_file = Path("logs") / create_unicode_safe_filename("__".join(parts) + ".log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = log_path
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s | %(funcName)-12s: %(message)s"))
    LOGGER.addHandler(file_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s | %(funcName)-12s: %(message)s'))
    LOGGER.addHandler(console)

    LOGGER.propagate = False  # Prevent log messages from being propagated to the root logger

    if found_handlers:
        LOGGER.warning("Replaced existing handlers with new ones")

def get_stats(a: NDArray) -> dict[str, float|int]:
    """Get statistics for an array."""
    assert a.ndim == 1, f"Expected 1D array, got {a.ndim}D"
    if a.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'std': float('nan'),
            'count': 0
        }
    return {
        'mean': float(np.mean(a)),
        'median': float(np.median(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'std': float(np.std(a)),
        'count': int(len(a))
    }


def normalize_rows(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalize a sparse CSR matrix (make rows sum to 1). Zero rows remain zero."""
    return sk_normalize(matrix, norm="l1", axis=1, copy=False)

@TIMER.time(formatters=dict(work_ids=lambda w: f"<{len(w)} works>",
                            work_references=lambda r: f"<{len(r)} reference lists>"))
def build_citation_adjacency_from_ids(
    work_ids: List[str],
    work_references: List[List[str]],
    add_self_loops: bool = True,
    normalize: bool = False,
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """Build a sparse citation adjacency matrix from work IDs and references.

    Args:
        work_ids: List of work IDs.
        work_references: List of reference lists (parallel to `work_ids`).
        add_self_loops: If True, add self-loops for numerical stability.
        normalize: If True, row-normalize the returned matrix so each row sums to 1.

    Returns:
        Tuple of (adjacency_matrix, id_to_idx mapping).
    """
    id_to_idx: Dict[str, int] = {wid: i for i, wid in enumerate(work_ids)}
    n = len(work_ids)

    row: List[int] = []
    col: List[int] = []

    for i, refs in enumerate(work_references):
        if add_self_loops:
            row.append(i)
            col.append(i)
        for ref_id in refs:
            if ref_id in id_to_idx:
                row.append(i)
                col.append(id_to_idx[ref_id])

    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))

    if normalize:
        adj = sk_normalize(adj, norm="l1", axis=1, copy=False)

    LOGGER.info(
        f"Built citation adjacency: {n} works, {adj.nnz} edges "
        f"(density: {adj.nnz / (n * n):.2e})"
    )

    return adj, id_to_idx
