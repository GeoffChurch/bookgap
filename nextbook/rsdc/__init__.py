"""
Randomized Simultaneous Diagonalization by Congruence (RSDC) algorithms.

This package implements the RSDC algorithm from:
He, H., & Kressner, D. (2025). A randomized algorithm for simultaneously 
diagonalizing symmetric matrices by congruence. arXiv:2402.16557.
"""

from .rnojd import (
    rnojd,
    ffdiag_rnojd,
    manopt_rnojd,
    cdopt_rnojd,
    offdiag,
    offdiag_frobenius_square,
    random_combination,
    runif_in_simplex
)

from .other_jd_algorithms import ffdiag, qndiag

__all__ = [
    'rnojd',
    'ffdiag_rnojd', 
    'manopt_rnojd',
    'cdopt_rnojd',
    'offdiag',
    'offdiag_frobenius_square',
    'random_combination',
    'runif_in_simplex',
    'ffdiag',
    'qndiag'
]
