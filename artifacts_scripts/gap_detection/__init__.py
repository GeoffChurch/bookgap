"""Plant-and-recover validation of density-ratio gap statistics.

Implements the cheapest validation from docs/backlog/curriculum-applications.md
("Gap detection, revisited"): plant synthetic holes (radial / concept / corridor)
in a foreground cloud and measure recovery by the proposed statistics

    gap(x) = g(x)^a * z( log g(x)/p(x) )

with g = background density (habitability), p = foreground density (occupancy),
z = Fightin'-Words variance-normalized log-ratio.

CPU-only by design: pure sklearn/scipy detectors; CUDA is masked in every module.
"""
