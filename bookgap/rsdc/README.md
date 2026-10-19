# Randomized Simultaneous Diagonalization by Congruence (RSDC)

This directory contains implementations of the Randomized Simultaneous Diagonalization by Congruence (RSDC) algorithm and related joint diagonalization methods.

## Citation

This implementation is based on the work:

**He, H., & Kressner, D. (2025).** *A randomized algorithm for simultaneously diagonalizing symmetric matrices by congruence.* arXiv preprint arXiv:2402.16557. https://doi.org/10.48550/arXiv.2402.16557

## Files

- `other_jd_algorithms.py`: Additional joint diagonalization algorithms for comparison
  (MIT and BSD-3 licensed; see `THIRD_PARTY_NOTICES.md`)
- `rnojd.py`: The RSDC algorithm and related optimization methods. **Not distributed**
  (see below); the JAD denoiser is the only code that needs it.

## Obtaining `rnojd.py`

`rnojd.py` comes from [haoze12345/rsdc](https://github.com/haoze12345/rsdc), which
carries no licence, so this repository cannot redistribute it. To run the JAD
denoiser (the JAD row of the paper's Appendix B), fetch it yourself, pinned to the
commit it was taken from, with two edits: a package-relative import, and `np.intp`
for `np.int0`, which NumPy 2 removed.

```sh
curl -sL https://raw.githubusercontent.com/haoze12345/rsdc/e3ddf5199ad30f9c05206e8813fdba02e2fffa18/rnojd.py \
  | sed -e 's/^from other_jd_algorithms import/from .other_jd_algorithms import/' \
        -e 's/np\.int0\b/np.intp/g' \
  > bookgap/rsdc/rnojd.py
```

The result gives bit-identical output to the copy the paper's runs used, for both
`rnojd` and `manopt_rnojd`. `.gitignore` keeps it out of commits.

## Abstract

A family of symmetric matrices A₁,...,Aₐ is SDC (simultaneous diagonalization by congruence, also called non-orthogonal joint diagonalization) if there is an invertible matrix X such that every XᵀAₖX is diagonal. The RSDC algorithm reduces SDC to a generalized eigenvalue problem by considering two (random) linear combinations of the family.

The algorithm achieves:
- **Exact recovery**: Diagonalization with probability 1 if the family is exactly SDC
- **Robust recovery**: For families ε-close to SDC, diagonalization up to an error of norm O(ε) with high probability
- **Improved efficiency**: Outperforms existing optimization-based methods while maintaining comparable accuracy

## Applications

The method has been verified for:
- Synthetic data analysis
- Image separation tasks
- EEG analysis
