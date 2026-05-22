"""Alternative gap-size metrics for re-ranking K-ball candidates.

See ``docs/backlog/gap-detection-methods/kball-residual-mass.md`` for the
parent design doc tracking these variants. Each metric is a pure function
of ``(gap_member_indices, article_emb, book_emb)`` returning a float; the
larger the value, the more meaningful the gap. ``raw_kball`` is the
identity (number of members) and matches the historical behaviour.

These are gap-*sizing* metrics applied to re-rank an existing pool of
K-ball candidates (typically `gap.candidate_pool_size` of them). They are
NOT a replacement for the deflation strategy (which controls the
candidate generation order). They do not affect downstream proposal
generation other than by changing WHICH candidate is selected as the top
gap per topic.

None of these discriminate misclassified-topic debris (debris is
internally cohesive and geometrically distant from real books in the
topic, so book-claim signals don't shrink it). For debris filtering see
the binomial-p approach in ``bookgap/depletion_test.py``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


SizeMetricFn = Callable[[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]], float]


def raw_kball(
    members: NDArray[np.int64],
    article_emb: NDArray[np.float32],
    book_emb: NDArray[np.float32],
) -> float:
    """Historical: number of articles in the bookless ball."""
    del article_emb, book_emb
    return float(len(members))


def virtual_centroid(
    members: NDArray[np.int64],
    article_emb: NDArray[np.float32],
    book_emb: NDArray[np.float32],
) -> float:
    """Variant 4: count members preferring the gap centroid over any real book.

    Treat the gap's centroid as a virtual book. A member contributes 1 if
    its distance to the centroid is no greater than its distance to its
    nearest real book — i.e., the virtual book would 'win' the Voronoi
    cell of this article. Empty gap → 0. No books → all members prefer the
    centroid trivially → returns ``len(members)``.
    """
    if len(members) == 0:
        return 0.0
    member_embs = article_emb[members]
    centroid = member_embs.mean(axis=0, keepdims=True)
    d_to_centroid_sq = np.sum((member_embs - centroid) ** 2, axis=1)
    if book_emb.shape[0] == 0:
        return float(len(members))
    # Brute-force squared distance to all books: (M, B)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a2 = np.sum(member_embs ** 2, axis=1, keepdims=True)            # (M, 1)
    b2 = np.sum(book_emb ** 2, axis=1, keepdims=True).T             # (1, B)
    cross = member_embs @ book_emb.T                                 # (M, B)
    d_to_books_sq = a2 + b2 - 2.0 * cross
    np.clip(d_to_books_sq, 0.0, None, out=d_to_books_sq)            # numerical floor
    d_to_nearest_book_sq = d_to_books_sq.min(axis=1)
    return float(np.sum(d_to_centroid_sq <= d_to_nearest_book_sq))


def mutual_nn(
    members: NDArray[np.int64],
    article_emb: NDArray[np.float32],
    book_emb: NDArray[np.float32],
) -> float:
    """Variant 3: partition mutual-NN discount.

    Partition gap members by their nearest real book. For each partition
    ``p_b`` of size ``|p_b|``, query book b's top ``|p_b|`` nearest
    articles over the FULL article set (NOT excluding gap members — the
    book is allowed to "claim back" in-gap articles). The articles in the
    intersection of ``p_b`` and book b's claim set are 'really claimed by
    b' and are discounted from the gap.

    Returns ``len(members) - discount``. Empty members → 0. No books → no
    discount possible → ``len(members)``.
    """
    n = len(members)
    if n == 0:
        return 0.0
    n_books = book_emb.shape[0]
    if n_books == 0:
        return float(n)

    member_embs = article_emb[members]                                  # (M, D)
    # Brute-force nearest book per member.
    a2 = np.sum(member_embs ** 2, axis=1, keepdims=True)                # (M, 1)
    b2 = np.sum(book_emb ** 2, axis=1, keepdims=True).T                 # (1, B)
    d_mb_sq = a2 + b2 - 2.0 * (member_embs @ book_emb.T)                # (M, B)
    np.clip(d_mb_sq, 0.0, None, out=d_mb_sq)
    nearest_book = d_mb_sq.argmin(axis=1)                               # (M,)

    # Squared distances from each book to ALL articles (so we can pick
    # book b's top |p_b| nearest globally — no gap restriction).
    A = article_emb.shape[0]
    art2 = np.sum(article_emb ** 2, axis=1, keepdims=True)              # (A, 1)
    # We only need the rows of `d_book_articles_sq` for books that have a non-empty partition.
    used_books = np.unique(nearest_book)
    discount = 0
    for b in used_books:
        partition_mask = nearest_book == b
        p_b = members[partition_mask]
        k = int(p_b.size)
        if k == 0:
            continue
        # Compute book b's distances to all articles.
        bv = book_emb[b:b + 1]                                          # (1, D)
        d_b_all_sq = art2.ravel() + float(np.sum(bv * bv)) - 2.0 * (article_emb @ bv.T).ravel()
        # Top-k smallest -> book b's |p_b| nearest articles (global indices).
        # argpartition is O(A); cheaper than full argsort.
        nn_b = np.argpartition(d_b_all_sq, kth=min(k, A - 1))[:k]
        nn_b_set = set(int(x) for x in nn_b.tolist())
        for m in p_b.tolist():
            if int(m) in nn_b_set:
                discount += 1
    return float(n - discount)


def book_knn_discount(
    members: NDArray[np.int64],
    article_emb: NDArray[np.float32],
    book_emb: NDArray[np.float32],
    *,
    k_global: int = 50,
) -> float:
    """Variant 1: each book claims its top ``k_global`` nearest articles globally.

    A gap member is discounted iff its nearest book also has it in that
    book's top-``k_global`` neighborhood. Differs from variant 3 only in
    using a fixed global k instead of partition-size-tied k.
    """
    n = len(members)
    if n == 0:
        return 0.0
    n_books = book_emb.shape[0]
    if n_books == 0:
        return float(n)
    A = article_emb.shape[0]
    k = min(k_global, A)
    member_embs = article_emb[members]
    a2 = np.sum(member_embs ** 2, axis=1, keepdims=True)
    b2 = np.sum(book_emb ** 2, axis=1, keepdims=True).T
    d_mb_sq = a2 + b2 - 2.0 * (member_embs @ book_emb.T)
    np.clip(d_mb_sq, 0.0, None, out=d_mb_sq)
    nearest_book = d_mb_sq.argmin(axis=1)

    # For each unique book that has at least one member, compute its top-k nearest articles
    # and check membership.
    art2 = np.sum(article_emb ** 2, axis=1, keepdims=True).ravel()
    used_books = np.unique(nearest_book)
    book_nn: dict[int, set[int]] = {}
    for b in used_books:
        bv = book_emb[int(b):int(b) + 1]
        d_b_all_sq = art2 + float(np.sum(bv * bv)) - 2.0 * (article_emb @ bv.T).ravel()
        nn_b = np.argpartition(d_b_all_sq, kth=min(k, A - 1))[:k]
        book_nn[int(b)] = set(int(x) for x in nn_b.tolist())

    discount = sum(
        1 for m_idx, m in enumerate(members.tolist())
        if int(m) in book_nn.get(int(nearest_book[m_idx]), ())
    )
    return float(n - discount)


def kde_ratio(
    members: NDArray[np.int64],
    article_emb: NDArray[np.float32],
    book_emb: NDArray[np.float32],
) -> float:
    """Variant 2: soft KDE-on-KDE residual mass.

    Place a Gaussian on each book (bandwidth = global book pairwise
    distance median) and a single Gaussian on the gap (bandwidth = RMS
    of gap members to centroid). For each member, compute
    ``log(gap_density) - log(book_mixture_density)`` and sum the
    sigmoid. The constant ``-(D/2) log(2 pi sigma^2)`` cancels for the
    gap Gaussian but NOT across the book / gap comparison (different
    sigmas), so we include both normalization terms.

    Empty members → 0. No books → all members are in pure gap territory
    → sigmoid(+inf) ≈ 1 each → ``len(members)``. Degenerate gap (all
    members at the same point) → sigma_gap = 0 → gap_density is a
    delta → log_ratio = +inf → treat as raw count.
    """
    n = len(members)
    if n == 0:
        return 0.0
    n_books = book_emb.shape[0]
    if n_books == 0:
        return float(n)

    member_embs = article_emb[members]                                  # (M, D)
    D = member_embs.shape[1]

    # Gap Gaussian
    centroid = member_embs.mean(axis=0, keepdims=True)                  # (1, D)
    member_offsets = member_embs - centroid
    sigma_gap_sq = float(np.mean(np.sum(member_offsets ** 2, axis=1)))  # mean squared dist to centroid
    if sigma_gap_sq <= 1e-12:
        return float(n)  # collapsed gap

    # Book bandwidth from book pairwise distances (median squared distance
    # to nearest book; fallback to spread of book set if only 1-2 books).
    if n_books >= 2:
        # nearest-book distance for each book (excluding self)
        bb = book_emb @ book_emb.T
        b_sq = np.sum(book_emb ** 2, axis=1)
        d_bb_sq = b_sq[:, None] + b_sq[None, :] - 2.0 * bb
        np.fill_diagonal(d_bb_sq, np.inf)
        sigma_book_sq = float(np.median(d_bb_sq.min(axis=1)))
        if sigma_book_sq <= 1e-12:
            sigma_book_sq = sigma_gap_sq
    else:
        sigma_book_sq = sigma_gap_sq  # single book — match gap scale

    # log gap density at each member: -||m - centroid||^2 / (2 sigma_gap^2) - (D/2) log(2 pi sigma_gap^2)
    log_gap = -np.sum(member_offsets ** 2, axis=1) / (2.0 * sigma_gap_sq) \
              - 0.5 * D * np.log(2.0 * np.pi * sigma_gap_sq)            # (M,)

    # log mixture density at each member via logsumexp.
    # log book_b at m: -||m - book_b||^2 / (2 sigma_b^2) - (D/2) log(2 pi sigma_b^2)
    a2 = np.sum(member_embs ** 2, axis=1, keepdims=True)
    b2 = np.sum(book_emb ** 2, axis=1, keepdims=True).T
    d_mb_sq = a2 + b2 - 2.0 * (member_embs @ book_emb.T)
    np.clip(d_mb_sq, 0.0, None, out=d_mb_sq)
    log_per_book = -d_mb_sq / (2.0 * sigma_book_sq) \
                   - 0.5 * D * np.log(2.0 * np.pi * sigma_book_sq)      # (M, B)
    # logsumexp along books, then divide by n_books (-> subtract log)
    m_max = log_per_book.max(axis=1, keepdims=True)
    log_mix_unnorm = m_max.ravel() + np.log(np.exp(log_per_book - m_max).sum(axis=1))
    log_book_mixture = log_mix_unnorm - np.log(n_books)                 # (M,)

    log_ratio = log_gap - log_book_mixture                              # (M,)
    # Sigmoid (stable form)
    soft = np.where(
        log_ratio >= 0,
        1.0 / (1.0 + np.exp(-log_ratio)),
        np.exp(log_ratio) / (1.0 + np.exp(log_ratio)),
    )
    return float(np.sum(soft))


_REGISTRY: dict[str, SizeMetricFn] = {
    "raw_kball": raw_kball,
    "virtual_centroid": virtual_centroid,
    "mutual_nn": mutual_nn,
    "book_knn_discount": book_knn_discount,
    "kde_ratio": kde_ratio,
}


def get(name: str) -> SizeMetricFn:
    """Look up a size-metric function by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        valid = sorted(_REGISTRY)
        raise ValueError(
            f"unknown gap.size_metric={name!r}; valid options: {valid}"
        ) from exc
