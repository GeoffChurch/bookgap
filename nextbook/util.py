from dataclasses import dataclass
import enum
import logging
from functools import wraps
import inspect
import re
from typing import Any, Callable, Optional
import joblib
from pathlib import Path
from time import time

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
                start_time = time()
                result = f(*args, **kw)
                elapsed_time = time() - start_time
                # self.print(f'func:{f.__name__}({filtered_s}) took {elapsed_time:.2f} seconds')
                self.print(f'Took {elapsed_time:.2f} seconds for func:{f.__name__}({filtered_s})')
                return result
            return wrap
        return decorator


LOGGER = logging.getLogger("nextbook")

TIMER = Timer(print=LOGGER.info)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY = joblib.Memory(CACHE_DIR / "joblib", verbose=1)

class SubField(enum.Enum):
    AI = "1702"
    def __repr__(self):
        return self.name

class Topic(enum.Enum):
    NLP = "T10181"
    ROBOTICS = "T10191"
    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class Subject:
    subfield: Optional[SubField] = None
    topic: Optional[Topic] = None
    def to_dict(self) -> dict[str, str]:
        ret = {}
        if self.topic is not None:
            ret["primary_topic.id"] = self.topic.value
        if self.subfield is not None:
            ret["primary_topic.subfield.id"] = self.subfield.value
        return ret
    def check_work(self, work: dict[str, Any]) -> bool:
        """Check if the work matches this filter."""
        if self.topic is not None and work["primary_topic"]["id"] != f"https://openalex.org/{self.topic.value}":
            return False
        if self.subfield is not None and work["primary_topic"]["subfield"]["id"] != f"https://openalex.org/subfields/{self.subfield.value}":
            return False
        return True

class Subjects(enum.Enum):
    # ROBOTICS = (TopicPreFilter(topic="T10191", subfield=None), lambda _: True)
    # NLP = (TopicFilter(topic=None, subfield="1702"), lambda work: work["primary_topic"]["id"] == "T10181")
    NLP = (Subject(subfield=SubField.AI), Subject(topic=Topic.NLP))
    # AI = (TopicPreFilter(topic=None, subfield="1702"), lambda _: True)

    def __init__(self, pre_filter: Subject, post_filter: Subject):
        self.pre_filter = pre_filter
        self.post_filter = post_filter

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)

class WorkType(enum.Enum):
    BOOK    = "book"
    ARTICLE = "article"
    def __str__(self):
        return self.value
    def __repr__(self):
        return str(self)

def create_unicode_safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9 _.-=\(\),"\']', ' ', str(s)).rstrip()