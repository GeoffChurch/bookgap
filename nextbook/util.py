import argparse
from dataclasses import dataclass
import enum
import logging
from functools import wraps
import inspect
import re
from typing import Any, Callable, Optional
import joblib
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

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


LOGGER = logging.getLogger("nextbook")

TIMER = Timer(print=LOGGER.info)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY = joblib.Memory(CACHE_DIR / "joblib", verbose=0)

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

class WorkType(enum.Enum):
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
    subfield: Optional[str] = None
    topic: Optional[str] = None
    work_type: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        ret = {}
        if self.topic is not None:
            ret["primary_topic.id"] = self.topic
        if self.subfield is not None:
            ret["primary_topic.subfield.id"] = self.subfield
        if self.work_type is not None:
            ret["type"] = self.work_type
        return ret
    
    def check_work(self, work: dict[str, Any]) -> bool:
        """Check if the work matches this filter."""
        if self.topic is not None and work["primary_topic"]["id"] != f"https://openalex.org/{self.topic}":
            return False
        if self.subfield is not None and work["primary_topic"]["subfield"]["id"] != f"https://openalex.org/subfields/{self.subfield}":
            return False
        if self.work_type is not None and work["type"] != self.work_type:
            return False
        return True
    
    def __str__(self) -> str:
        parts = []
        if self.topic is not None:
            parts.append(f"topic={self.topic}")
        if self.subfield is not None:
            parts.append(f"subfield={self.subfield}")
        if self.work_type is not None:
            parts.append(f"work_type={self.work_type}")
        return ", ".join(parts) if parts else "no filter"
    
    def __repr__(self) -> str:
        return str(self)

@dataclass(frozen=True)
class FilterPair:
    pre: Filter
    post: Filter

    def __post_init__(self):
        if self.pre.work_type is not None and self.post.work_type is not None:
            if self.pre.work_type != self.post.work_type:
                raise ValueError("Pre-filter and post-filter work types do not match")
        if self.pre.subfield is not None and self.post.subfield is not None:
            if self.pre.subfield != self.post.subfield:
                raise ValueError("Pre-filter and post-filter subfields do not match")
        if self.pre.topic is not None and self.post.topic is not None:
            if self.pre.topic != self.post.topic:
                raise ValueError("Pre-filter and post-filter topics do not match")
            
    def merged_name(self) -> str:
        """Return a merged name for the filter pair."""
        parts = {}
        if self.pre.topic is not None:
            parts["topic"] = self.pre.topic
        if self.pre.subfield is not None:
            parts["subfield"] = self.pre.subfield
        if self.pre.work_type is not None:
            parts["work_type"] = self.pre.work_type
        if self.post.topic is not None:
            parts["post_topic"] = self.post.topic
        if self.post.subfield is not None:
            parts["post_subfield"] = self.post.subfield
        if self.post.work_type is not None:
            parts["post_work_type"] = self.post.work_type
        return "(" + ", ".join(f"{k}={v}" for k, v in parts.items()) + ")"
            
class Filters(enum.Enum):
    # ROBOTICS = (TopicPreFilter(topic="T10191", subfield=None), lambda _: True)
    # NLP = (TopicFilter(topic=None, subfield="1702"), lambda work: work["primary_topic"]["id"] == "T10181")
    NLP = FilterPair(Filter(subfield=SubField.AI.value), Filter(topic=Topic.NLP.value))
    # AI = (TopicPreFilter(topic=None, subfield="1702"), lambda _: True)

    def work_type(self) -> str:
        """Return the work type for this filter."""
        if self.value.pre.work_type is not None:
            return self.value.pre.work_type
        if self.value.post.work_type is not None:
            return self.value.post.work_type
        raise ValueError("No work type defined for this filter")

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)

    def _validate(self):
        if self.value.pre.work_type is not None and self.value.post.work_type is not None:
            if self.value.pre.work_type != self.value.post.work_type:
                raise ValueError("Pre-filter and post-filter work types do not match")
        if self.value.pre.subfield is not None and self.value.post.subfield is not None:
            if self.value.pre.subfield != self.value.post.subfield:
                raise ValueError("Pre-filter and post-filter subfields do not match")
        if self.value.pre.topic is not None and self.value.post.topic is not None:
            if self.value.pre.topic != self.value.post.topic:
                raise ValueError("Pre-filter and post-filter topics do not match")

def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _.-=\(\),"\']', ' ', str(s)).rstrip()

def setup_logging(log_name: str = "", log_level: str = "INFO", include_time: bool = True):
    LOGGER.setLevel(log_level)
    
    parts = []
    if include_time:
        parts.append(time.strftime('%Y-%m-%d_%H-%M-%S'))
    if log_name:
        parts.append(log_name)
    if not parts:
        raise ValueError("At least one part must be provided for the log name")
    log_file = Path("logs") / ("_".join(parts) + ".log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-8s | %(funcName)-12s: %(message)s"))
    LOGGER.addHandler(file_handler)

    # Console handler
    console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)-8s | %(funcName)-12s: %(message)s'))
    LOGGER.addHandler(console)

    LOGGER.propagate = False  # Prevent log messages from being propagated to the root logger

def get_stats(a: NDArray):
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
