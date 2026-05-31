"""Stdlib-logging shim for the curriculum package.

The package is fully decoupled from bookgap, so it cannot use bookgap's
``LOGGER``/``TIMER`` singletons. Modules call ``get_logger(__name__)`` to
obtain a standard ``logging.Logger`` namespaced under ``curriculum``. The
library never configures handlers itself (no ``basicConfig``); the embedding
application owns log routing. A ``NullHandler`` on the root package logger
keeps "No handlers could be found" warnings quiet when the app does nothing.
"""

from __future__ import annotations

import logging

_ROOT = logging.getLogger("curriculum")
if not _ROOT.handlers:
    _ROOT.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``curriculum``.

    ``name`` is typically ``__name__`` (e.g. ``curriculum.core.rank``); the
    returned logger inherits the package ``NullHandler`` so the library stays
    silent until the host application attaches a handler.
    """
    if name == "__main__" or not name.startswith("curriculum"):
        name = f"curriculum.{name}"
    return logging.getLogger(name)
