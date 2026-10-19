"""
Joint diagonalization algorithms.

`other_jd_algorithms` (qndiag, ffdiag) is third-party code under MIT and BSD-3
licences; see THIRD_PARTY_NOTICES.md. `rnojd`, the RSDC algorithm of He & Kressner
(arXiv:2402.16557) that the JAD denoiser uses, is not distributed: its upstream
carries no licence. See README.md for how to obtain it. Import it as
`bookgap.rsdc.rnojd` where needed, never from here, so that this package imports
without it.
"""

from .other_jd_algorithms import ffdiag, qndiag

__all__ = [
    'ffdiag',
    'qndiag'
]
