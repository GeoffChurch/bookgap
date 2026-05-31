"""Optional curriculum layers for embedding-rich (or graph-spectral) domains.

None of these are needed by the minimal Lean-extractive core (rank -> select ->
order). Each is imported lazily by ``curriculum.build`` only when its config
toggle is on, so a missing heavy/optional dependency (e.g. ``ollama`` for the
judge) never breaks a run that doesn't use it.
"""
