# Randomized Simultaneous Diagonalization by Congruence (RSDC)

This directory contains implementations of the Randomized Simultaneous Diagonalization by Congruence (RSDC) algorithm and related joint diagonalization methods.

## Citation

This implementation is based on the work:

**He, H., & Kressner, D. (2025).** *A randomized algorithm for simultaneously diagonalizing symmetric matrices by congruence.* arXiv preprint arXiv:2402.16557. https://doi.org/10.48550/arXiv.2402.16557

## Files

- `rnojd.py`: Implementation of the RSDC algorithm and related optimization methods
- `other_jd_algorithms.py`: Additional joint diagonalization algorithms for comparison

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
