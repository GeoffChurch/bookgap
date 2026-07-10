#!/usr/bin/env python3
"""Book-surprise bake-off: under which gap definition g are human-authored books
(and book chapters) MAXIMALLY EXCEPTIONAL vs a null?

The static discriminative question (where books ARE), not temporal prediction.
For each point-scorer g(x) we ask: how separable are book/chapter locations from
a null of field articles, AUC(g(positives) > g(null)) (Mann-Whitney)?

FRAME (point-scorer g(x)):  score an arbitrary embedding x for how "gappy /
exceptional" it is relative to the ARTICLE FIELD.
  * Foreground field      F = articles  (the cloud whose local density we read)
  * Background corpus      B = books     (defines habitability covariate r_B,k)
  * Positive class A       = in-corpus book locations
  * Positive class B       = ToC chapter locations (GIST-embedded title+description)
  * NULL (i)               = matched-control articles, matched to the books on
                             {log r_B,10, log r_F,10} (t2_outcomes.matched_controls)
  * NULL (ii)              = random field articles (robustness)
  * HABITABILITY = a hard GATE: score only within the book-dense hull
                   (r_B,10 <= q0.90 of books' r_B,10). NOT a soft g^a weight.

The seven definitions g (REUSE the exact validated kernels/formulas):
  A1 raw_kball : # articles within radius_fraction(0.75) x dist-to-nearest-book, k=1
                 (kball_algorithm.compute_gap_sizes_for_embeddings / chapter_gap.raw)
  A3 delta     : log r_F,10 - log r_B,10            (GapScorer 'delta_only', k=10)
  A4 full      : hab * (delta - lam*sigma)          (GapScorer 'full', k=10)
  A5 no_hab    : r_F,10  (distance-to-fg only)       (GapScorer 'no_hab' -- the
                 validated emptiness core; reported ALONGSIDE 'full' per the
                 banked lesson that the soft g^a HURTS in-scope)
  A6 fused     : rank by r_F,10, veto off-hull or delta>log3   (fusion.fused_scores)
  maxratio     : max over k nearest books of (#articles closer than book j)/j
                 (chapter_gap.py maxratio loop)
  kde_ratio    : single-point log-density-ratio  log p_F(x) - log p_B-mixture(x)
                 ADAPTED from the member-set gap_size_metrics.kde_ratio (see note).

POINT-SCORER REUSE: GapScorer.score() scores a FIXED candidate set (bg+probes);
to score ARBITRARY query locations (books/chapters/nulls), this module reproduces
the GapScorer / full_statistic formulas over arbitrary query rows using the EXACT
detector kernels knn_radii / count_within_radius. (book_audit.full_statistic is the
precedent: it reimplements the same 'full' formula over arbitrary F/B with these
kernels, and its identity is unit-tested against GapScorer.)

LEAKAGE GUARDS (spec #4):
  * book positives:   r_B,k is leave-one-out (the book is in B -> drop self).
                      hab-standardisation reference and kde book-mixture also LOO.
  * article positives & all null articles: r_F,k is leave-one-out (in F -> drop self).
  * chapters:         in neither corpus -> no self-exclusion.
  A1/A3/A4/A5 are otherwise clean: books are not in the article foreground F.

CALIBRATION (spec critical note #2): before trusting any g's AUC, the matcher is
calibrated on a RANDOM-location null FIRST -- random habitable article locations
treated as pseudo-positives, matched to controls; AUC(random vs matched) must be
~0.5 for every g ('random50' sanity). A biased matcher shows up here.

CLUSTERED BOOTSTRAP (spec critical note #3): positives are grouped by location
near-duplication (t2_outcomes.greedy_location_groups); the bootstrap resamples
whole groups (>=1000 resamples), so point near-duplication does not inflate the CI.

    PYTHONPATH=. uv run python -m artifacts_scripts.curriculum.book_surprise_bakeoff \
        --subfield 1702 \
        --toc artifacts/book_gaps/jurafsky_toc.json artifacts/book_gaps/koehn_toc.json \
              artifacts/book_gaps/W2898513995_toc.json artifacts/book_gaps/W4247813589_toc.json \
        --out artifacts/book_gaps/book_surprise_bakeoff.json
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only (banked: GPU may be owned by a sweep)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import time
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import mannwhitneyu

# REUSED exact validated kernels / machinery -------------------------------------
from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.gap_detection.detectors import (
    EPS,
    count_within_radius,
    knn_radii,
)
from artifacts_scripts.gap_detection.t2_outcomes import (
    DEFAULT_CALIPERS,
    greedy_location_groups,
    matched_controls,
)
from bookgap.subcloud import SubCloud
from bookgap.util import Filter, LOGGER, setup_logging

# ----- definition hyper-parameters (pinned to the validated values) -------------
K = 10                       # GapScorer / matcher k (A3/A4/A5/A6 + covariates)
K1 = 1                       # A1 raw k-ball uses the nearest book (k=1)
RADIUS_FRACTION = 0.75       # A1 raw k-ball radius fraction
KMAX = 20                    # maxratio: scan out to the 20th nearest book
HAB_EXP = 0.5                # 'a' in g^a (used only inside A4 'full')
LCB_LAMBDA = 0.5             # 'lam' in z = delta - lam*sigma
ALPHA_PRIOR = 1.0            # Dirichlet prior on local counts
HULL_Q = 0.90                # habitability hull quantile (hard gate + A6 veto)
DELTA_CEILING = float(np.log(3.0))  # A6 proximal-presence veto
# matcher calipers on the 2-D covariate {log r_B,10, log r_F,10}: take the
# DEFAULT_CALIPERS' first two widths (density, dist-to-fg) in sigma units. The
# 3rd default (growth) has no analog in this STATIC frame (no temporal axis).
CALIPERS = DEFAULT_CALIPERS[:2]


# ============================================================ radii (the substrate)
def query_radii(
    Q: np.ndarray, F: np.ndarray, B: np.ndarray, k: int,
    *, self_in_F: bool = False, self_in_B: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """r_F,k(q) (dist to k-th nearest article) and r_B,k(q) (k-th nearest book).

    `self_in_F` / `self_in_B`: query rows ARE members of F / B (leave-one-out:
    take the (k+1)-th neighbour so the self row at distance 0 is excluded). This
    is the detector's `ref_is_query` semantics, applied per side.
    """
    r_F = knn_radii(Q, F, k, ref_is_query=self_in_F)
    r_B = knn_radii(Q, B, k, ref_is_query=self_in_B)
    return r_F, r_B


# ============================================================ the seven definitions
def score_definitions(
    Q: np.ndarray, F: np.ndarray, B: np.ndarray,
    *, self_in_F: bool, self_in_B: bool,
    hab_mu: float, hab_sd: float,
    book_mix: "BookMixture",
) -> dict[str, np.ndarray]:
    """Score query locations Q under ALL definitions. Higher = more exceptional.

    F = article foreground, B = book background. hab_mu/hab_sd standardise
    log r_B,k against the BOOK corpus (hab is 'a property of the bg corpus only',
    per fusion.py). book_mix carries the precomputed book Gaussian mixture for
    the kde_ratio single-point density.
    """
    out: dict[str, np.ndarray] = {}
    nq = Q.shape[0]

    # --- shared radii (k=10) -----------------------------------------------------
    r_F = knn_radii(Q, F, K, ref_is_query=self_in_F)
    r_B = knn_radii(Q, B, K, ref_is_query=self_in_B)
    log_rF, log_rB = np.log(r_F), np.log(r_B)

    # A3 delta_only ---------------------------------------------------------------
    delta = log_rF - log_rB
    out["A3_delta"] = delta

    # A4 full = hab * (delta - lam*sigma) -----------------------------------------
    # evidence at the adaptive max-ball R = max(r_F, r_B); both clouds estimable there.
    R = np.maximum(r_F, r_B)
    n_FR = count_within_radius(Q, F, R, ref_is_query=self_in_F)
    n_BR = count_within_radius(Q, B, R, ref_is_query=self_in_B)
    sigma = np.sqrt(1.0 / (n_FR + ALPHA_PRIOR) + 1.0 / (np.maximum(n_BR, 0) + ALPHA_PRIOR))
    z = delta - LCB_LAMBDA * sigma
    hab = np.exp(HAB_EXP * (-(log_rB - hab_mu) / hab_sd))
    out["A4_full"] = hab * z

    # A5 no_hab = r_F,k  (the validated emptiness core) ---------------------------
    out["A5_no_hab"] = r_F

    # A6 fused: rank by r_F,k; veto off-hull (r_B above book hull) or delta>log3 --
    # Vetoed -> -inf (pushed to the bottom), exactly fusion.fused_scores semantics.
    veto = (r_B > book_mix.hull_thr) | (delta > DELTA_CEILING)
    fused = r_F.astype(np.float64).copy()
    fused[veto] = -np.inf
    out["A6_fused"] = fused

    # A1 raw k-ball: # articles within radius_fraction x dist-to-nearest-book -----
    # CANONICAL semantics (kball_algorithm / chapter_gap.py): radius_fraction scales
    # the SQUARED nearest-book distance (ball radius^2 = 0.75 * d_nn^2). count_within
    # _radius squares its radius arg, so pass sqrt(0.75)*d_nn -> ball radius^2 = 0.75*d_nn^2.
    r_B1 = knn_radii(Q, B, K1, ref_is_query=self_in_B)        # nearest book (Euclidean)
    radii_a1 = np.sqrt(RADIUS_FRACTION) * r_B1                 # -> radius^2 = 0.75 * d_nn^2
    out["A1_raw_kball"] = count_within_radius(
        Q, F, radii_a1, ref_is_query=self_in_F
    ).astype(np.float64)

    # maxratio + kde_ratio need the k nearest books per query and their distances.
    # Compute the KMAX nearest book distances once (LOO drops the self book col).
    out["maxratio"] = _maxratio(Q, F, B, self_in_F=self_in_F, self_in_B=self_in_B)
    out["kde_ratio"] = book_mix.single_point_log_ratio(
        Q, r_F=r_F, self_in_F=self_in_F
    )

    assert all(v.shape[0] == nq for v in out.values())
    return out


def _maxratio(
    Q: np.ndarray, F: np.ndarray, B: np.ndarray,
    *, self_in_F: bool, self_in_B: bool, kmax: int = KMAX,
) -> np.ndarray:
    """max over the kmax nearest books of (#articles closer than book j)/j.

    Direct port of chapter_gap.py's maxratio loop, vectorised over many queries.
    For each query: rank books by distance; for j=1..kmax let r_j = dist to the
    j-th nearest book and a_j = #articles within r_j; score = max_j a_j / j.
    """
    from sklearn.neighbors import NearestNeighbors

    # kmax nearest book DISTANCES per query (LOO: take an extra col, drop col 0).
    kk = kmax + 1 if self_in_B else kmax
    kk = min(kk, B.shape[0])
    nbB = NearestNeighbors(n_neighbors=kk, algorithm="brute").fit(B)
    dB, _ = nbB.kneighbors(Q)                                  # (nq, kk) ascending
    if self_in_B:
        dB = dB[:, 1:]                                         # drop self (dist 0)
    r_books = dB[:, :kmax]                                     # (nq, kmax) radii

    # a_j = # articles within r_j: count articles within the LARGEST radius once,
    # then the per-j counts are cumulative -> count within each r_j via searchsort
    # on the sorted article distances inside the max ball. Do it per query (kmax
    # is tiny) but reuse one chunked distance pass.
    nq = Q.shape[0]
    out = np.zeros(nq, dtype=np.float64)
    r_max = r_books[:, -1]                                     # per-query outer radius
    # count_within_radius gives only the count at one radius; we need counts at
    # kmax radii. Compute article sqdists to each query in chunks and, per query,
    # use np.searchsorted on the article distances <= r_max.
    Fc = F.astype(np.float32)
    F_sq = np.einsum("ij,ij->i", Fc, Fc)
    chunk = 256
    for s in range(0, nq, chunk):
        q = Q[s : s + chunk].astype(np.float32)
        d2 = np.maximum(
            np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ Fc.T) + F_sq[None, :], 0.0
        )
        d = np.sqrt(d2)                                       # (cs, nF)
        for ii in range(q.shape[0]):
            gi = s + ii
            rr = r_books[gi]                                  # (kmax,)
            di = d[ii]
            within = di[di <= rr[-1] * (1.0 + 1e-6)]
            if self_in_F:
                # one of these is the self article at distance 0; drop it.
                within = np.sort(within)[1:] if within.size else within
            else:
                within = np.sort(within)
            # counts at each r_j via searchsorted (right side, inclusive tol).
            a_j = np.searchsorted(within, rr * (1.0 + 1e-6), side="right").astype(np.float64)
            j = np.arange(1, rr.size + 1, dtype=np.float64)
            out[gi] = float(np.max(a_j / j))
    return out


# ============================================================ book Gaussian mixture
class BookMixture:
    """Book-corpus quantities shared across all query batches.

    * hull_thr     : the habitability gate / A6 veto threshold (q0.90 of books' r_B,10).
    * single_point_log_ratio: the ADAPTED kde_ratio -- a per-query
        log p_F(x) - log p_Bmixture(x), where p_F is a kNN density of the ARTICLE
        field at x and p_Bmixture is the book Gaussian mixture (same Gaussian-on-each-
        book construction as gap_size_metrics.kde_ratio: bandwidth = median nearest-
        book sqdist). See the module docstring 'kde_ratio' note.
    """

    def __init__(self, B: np.ndarray, hull_thr: float):
        self.B = B.astype(np.float32)
        self.D = B.shape[1]
        self.hull_thr = hull_thr
        self.b_sq = np.einsum("ij,ij->i", self.B, self.B)
        # book bandwidth: median nearest-OTHER-book squared distance (kde_ratio rule).
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(self.B)
        d, _ = nn.kneighbors(self.B)
        self.sigma_book_sq = float(np.median(d[:, 1] ** 2))
        if self.sigma_book_sq <= 1e-12:
            self.sigma_book_sq = 1.0
        self.n_books = self.B.shape[0]

    def _log_book_mixture(self, Q: np.ndarray, *, loo_self: bool) -> np.ndarray:
        """log of the book Gaussian-mixture density at each query point.

        Mixture = (1/n) sum_b N(x; book_b, sigma_book^2 I), logsumexp-stable.
        `loo_self`: query rows ARE books (a subset of self.B, NOT necessarily in
        order) -> leave-one-out by dropping the single nearest book whose squared
        distance is ~0 (the exact self match). Robust to subsetting/reordering.
        """
        Qf = Q.astype(np.float32)
        q_sq = np.einsum("ij,ij->i", Qf, Qf)
        nq = Qf.shape[0]
        sb = self.sigma_book_sq
        const = -0.5 * self.D * np.log(2.0 * np.pi * sb)
        out = np.empty(nq, dtype=np.float64)
        chunk = 512
        for s in range(0, nq, chunk):
            qf = Qf[s : s + chunk]
            d2 = np.maximum(
                q_sq[s : s + chunk, None] - 2.0 * (qf @ self.B.T) + self.b_sq[None, :], 0.0
            )                                                  # (cs, n_books)
            n_eff = float(self.n_books)
            if loo_self:
                # drop the self book = the (unique) column at squared-dist ~0.
                jmin = np.argmin(d2, axis=1)
                rows = np.arange(qf.shape[0])
                is_self = d2[rows, jmin] <= 1e-8               # exact self match
                d2[rows[is_self], jmin[is_self]] = np.inf      # exclude from mixture
                # per-row effective count (n-1 where a self book was dropped).
                n_row = self.n_books - is_self.astype(np.float64)
            log_per = -d2 / (2.0 * sb) + const                 # (cs, n_books)
            m = log_per.max(axis=1, keepdims=True)
            lse = m.ravel() + np.log(np.exp(log_per - m).sum(axis=1))
            out[s : s + chunk] = lse - (np.log(n_row) if loo_self else np.log(n_eff))
        return out

    def single_point_log_ratio(
        self, Q: np.ndarray, *, r_F: np.ndarray, self_in_F: bool
    ) -> np.ndarray:
        """log p_F(x) - log p_Bmixture(x). p_F via the kNN-balloon density at x:
        log p_F(x) = log k - log V_d(r_F,k) = log k - [ (D/2) log(pi) + (D/2) log r^2
        - log Gamma(D/2 + 1) ], the standard k-NN density estimator. Constants that
        do not vary across x cancel under AUC, but we keep them for interpretability.
        loo for the book mixture is taken from self_in_B via the caller; here we only
        need self_in_F to know whether r_F already excluded the self article (it does)."""
        del self_in_F  # r_F is passed already LOO-corrected by the caller
        from math import lgamma, log, pi

        D = self.D
        log_unit_vol = (D / 2.0) * log(pi) - lgamma(D / 2.0 + 1.0)
        log_pF = log(K) - log_unit_vol - D * np.log(np.maximum(r_F, EPS))
        # book mixture log-density; LOO handled in _log_book_mixture via the flag
        # set on this instance by the caller (see score loop).
        log_pB = self._log_book_mixture(Q, loo_self=self._loo_book)
        return log_pF - log_pB

    # the caller toggles this per positive class (book positives -> LOO mixture)
    _loo_book: bool = False


# ============================================================ AUC + clustered boot
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score(pos) > score(neg)).

    A6_fused encodes 'vetoed' as -inf (pushed to the bottom). Those points must
    COUNT as lowest-rank losers, not be dropped (dropping a positive's own vetoed
    points would inflate its AUC). Map -inf -> a sentinel strictly below every
    finite score on BOTH sides so the rank statistic is well defined; tie-break
    -inf vs -inf at 0.5 (Mann-Whitney's own tie convention)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    finite = np.concatenate([pos[np.isfinite(pos)], neg[np.isfinite(neg)]])
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    if finite.size == 0:
        return 0.5  # everything vetoed on both sides -> indistinguishable
    sentinel = float(finite.min()) - 1.0
    pos = np.where(np.isneginf(pos), sentinel, pos)
    neg = np.where(np.isneginf(neg), sentinel, neg)
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (pos.size * neg.size))


def clustered_auc_ci(
    pos_scores: np.ndarray, neg_scores: np.ndarray, groups: np.ndarray,
    *, n_boot: int = 1000, seed: int = 0,
) -> dict:
    """AUC point estimate + clustered bootstrap 95% CI. Resamples whole positive
    location groups (near-duplicate locations are not independent evidence; spec
    critical note #3). The null is the shared pool (matched controls / random)."""
    rng = np.random.default_rng(seed)
    point = auc(pos_scores, neg_scores)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        gs = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([members[g] for g in gs])
        boots[b] = auc(pos_scores[idx], neg_scores)
    boots = boots[np.isfinite(boots)]
    return {
        "auc": round(point, 4),
        "ci_lo": round(float(np.quantile(boots, 0.025)), 4) if boots.size else float("nan"),
        "ci_hi": round(float(np.quantile(boots, 0.975)), 4) if boots.size else float("nan"),
        "n_pos": int(pos_scores.size),
        "n_neg": int(neg_scores.size),
        "n_groups": int(uniq.size),
        "n_boot": int(boots.size),
    }


DEFS = ["A1_raw_kball", "A3_delta", "A4_full", "A5_no_hab", "A6_fused", "maxratio", "kde_ratio"]


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfield", default="1702")
    ap.add_argument("--field", default=None)
    ap.add_argument("--toc", nargs="+", required=True)
    ap.add_argument("--null-pool", type=int, default=60000,
                    help="# habitable articles sampled as the matched-control candidate "
                         "pool AND the random null (density radii still use the FULL field)")
    ap.add_argument("--m-controls", type=int, default=10,
                    help="matched controls per book location")
    ap.add_argument("--max-null", type=int, default=8000,
                    help="cap on the # null points actually SCORED (matched & random "
                         "each); 0 = no cap. Keeps scoring tractable; AUC stays tight.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-books", type=int, default=0,
                    help="if >0, subsample this many habitable book positives (speed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="book_surprise_bakeoff", log_level="INFO", include_time=False,
                  log_path=args.out.with_suffix(".log"))
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # --- load the AI subfield cloud ---------------------------------------------
    cfg = OmegaConf.structured(CurriculumConfig())
    if args.subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(args.subfield)
    if args.field:
        cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    LOGGER.info("Cloud: %s", cloud.name())
    books = SubCloud(parent=cloud, post_filter=Filter(work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(work_types=("article",)))
    B = np.asarray(books.embeddings(), dtype=np.float32)          # background corpus
    F = np.asarray(articles.embeddings(), dtype=np.float32)       # foreground field
    LOGGER.info("scope: %d books, %d articles (load %.1fs)", B.shape[0], F.shape[0], time.time() - t0)

    # --- book-corpus reference quantities (hull, hab standardisation, mixture) --
    # r_B,10 for books, LEAVE-ONE-OUT (each book's 10th nearest OTHER book).
    rB_books = knn_radii(B, B, K, ref_is_query=True)
    log_rB_books = np.log(rB_books)
    hab_mu, hab_sd = float(log_rB_books.mean()), float(log_rB_books.std() + EPS)
    hull_thr = float(np.quantile(rB_books, HULL_Q))               # habitability gate
    book_mix = BookMixture(B, hull_thr)
    LOGGER.info("hull_thr (r_B,10 q%.2f over books) = %.4f; sigma_book^2 = %.4f",
                HULL_Q, hull_thr, book_mix.sigma_book_sq)

    # ===================================================== POSITIVE CLASS A: books
    book_habitable = rB_books <= hull_thr                          # hard gate
    book_idx = np.where(book_habitable)[0]
    if args.max_books and book_idx.size > args.max_books:
        book_idx = np.sort(rng.choice(book_idx, size=args.max_books, replace=False))
    B_pos = B[book_idx]
    LOGGER.info("book positives (habitable): %d / %d", book_idx.size, B.shape[0])

    # ===================================================== POSITIVE CLASS B: chapters
    embedder = instantiate(cfg.experiment.cloud.embedder)
    chap_vecs, chap_book = [], []
    chap_meta = []
    with embedder as e:
        for ti, toc_path in enumerate(args.toc):
            toc = json.loads(Path(toc_path).read_text())
            chs = toc["chapters"]
            texts = [("", f"{c.get('title','')}. {c.get('description','')}") for c in chs]
            qv = np.asarray(e.embed_texts(texts), dtype=np.float32)
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
            chap_vecs.append(qv)
            chap_book.extend([ti] * len(chs))
            chap_meta.append({"toc": str(toc_path), "title": toc.get("title"), "n_chapters": len(chs)})
    C_all = np.vstack(chap_vecs).astype(np.float32)               # (n_chap, D)
    chap_book = np.asarray(chap_book)
    # habitability GATE on chapters: r_B,10 to books (NO self-exclusion; chapters
    # are not in B) must be within the book hull.
    _, rB_chap = query_radii(C_all, F, B, K, self_in_F=False, self_in_B=False)
    chap_habitable = rB_chap <= hull_thr
    C_pos = C_all[chap_habitable]
    chap_book_h = chap_book[chap_habitable]
    LOGGER.info("chapter positives (habitable): %d / %d", C_pos.shape[0], C_all.shape[0])
    for m in chap_meta:
        m["n_habitable"] = int((chap_book[chap_habitable] == chap_meta.index(m)).sum()) \
            if False else None  # (per-book habitable count filled below)
    for ti, m in enumerate(chap_meta):
        m["n_habitable"] = int((chap_book_h == ti).sum())

    # ===================================================== NULL candidate pool (articles)
    # The null draws from HABITABLE field articles. Computing r_B,10 for ALL 1.5M
    # articles is the heaviest step; do it once (chunked kNN to books) and keep the
    # habitable ones, then sample the pool. Density radii r_F,10 use the FULL field.
    LOGGER.info("computing r_B,10 for the full article field (habitability scope) ...")
    rB_art = _knn_radii_chunked(F, B, K, ref_is_query=False, qchunk=20000)
    art_habitable = np.where(rB_art <= hull_thr)[0]
    LOGGER.info("habitable articles: %d / %d (%.1f%%)", art_habitable.size, F.shape[0],
                100.0 * art_habitable.size / F.shape[0])
    pool_n = min(args.null_pool, art_habitable.size)
    pool_idx = np.sort(rng.choice(art_habitable, size=pool_n, replace=False))
    Pool = F[pool_idx]                                            # null candidate pool
    LOGGER.info("null candidate pool: %d habitable articles", pool_idx.size)

    # ===================================================== covariates for matching
    # {log r_B,10, log r_F,10}. Pool & books computed with the correct LOO per side.
    # books: r_F,10 (to articles, NO self -- books aren't articles), r_B,10 already have.
    rF_books, _ = query_radii(B_pos, F, B, K, self_in_F=False, self_in_B=True)
    C_books = np.column_stack([np.log(rB_books[book_idx]), np.log(rF_books)])
    # pool articles: r_F,10 LOO (in F), r_B,10 (to books).
    rF_pool = knn_radii(Pool, F, K, ref_is_query=True)            # LOO: in the field
    rB_pool = rB_art[pool_idx]
    C_pool = np.column_stack([np.log(rB_pool), np.log(rF_pool)])

    # ===================================================== matcher calibration FIRST
    # random50: random habitable article pseudo-positives, matched -> controls.
    # AUC(random vs their matched controls) must be ~0.5 for every g. (spec note #2)
    LOGGER.info("=== matcher calibration (random50 sanity) ===")
    calib = _matcher_calibration(
        F, B, Pool, pool_idx, C_pool, rF_pool, rB_pool, art_habitable,
        rB_art, hab_mu, hab_sd, book_mix, args.m_controls, rng,
    )
    for d in DEFS:
        LOGGER.info("  random50  %-12s AUC=%.3f", d, calib[d])

    # ===================================================== build matched-control null
    # loc radii = r_F,10 (fill neighbourhood); covariates = {log r_B,10, log r_F,10}.
    excluded = np.zeros(Pool.shape[0], dtype=bool)               # pool = articles only; no flagged docs
    ctrl_idx, info = matched_controls(
        loc_X=B_pos, loc_radii=rF_books, loc_C=C_books,
        pool_X=Pool, pool_radii=rF_pool, pool_C=C_pool,
        excluded=excluded, m_controls=args.m_controls,
        calipers=CALIPERS, seed=args.seed,
    )
    matched_pool_idx = np.unique(ctrl_idx.ravel())              # dedup the null set
    # cap the SCORED null for tractability (scoring a point = several 1.5M-field
    # kNN/count passes); a few thousand null points give a tight AUC. The matcher
    # still ran over ALL book locations -> the kept controls are a uniform sample
    # of a properly density/dist-matched set.
    if args.max_null and matched_pool_idx.size > args.max_null:
        matched_pool_idx = np.sort(rng.choice(matched_pool_idx, size=args.max_null, replace=False))
    LOGGER.info("matched-control null: %d unique articles scored (relax med=%.0f max=%d)",
                matched_pool_idx.size, info["relax_level_median"], info["relax_level_max"])
    Null_matched = Pool[matched_pool_idx]
    # random null (robustness): a fresh random habitable sample, same size as matched.
    rnd = rng.choice(art_habitable, size=min(matched_pool_idx.size, art_habitable.size), replace=False)
    Null_random = F[np.sort(rnd)]

    # ===================================================== SCORE everything
    LOGGER.info("=== scoring positives + nulls under all %d definitions ===", len(DEFS))
    book_mix._loo_book = True
    s_books = score_definitions(B_pos, F, B, self_in_F=False, self_in_B=True,
                                hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)
    book_mix._loo_book = False
    s_chap = score_definitions(C_pos, F, B, self_in_F=False, self_in_B=False,
                               hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)
    s_null_m = score_definitions(Null_matched, F, B, self_in_F=True, self_in_B=False,
                                 hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)
    s_null_r = score_definitions(Null_random, F, B, self_in_F=True, self_in_B=False,
                                 hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)

    # location near-duplication groups for the clustered bootstrap.
    book_groups = greedy_location_groups(B_pos, rF_books)
    # chapters: group by near-duplication too (book-id is a coarser grouping; use the
    # geometric rule for consistency with books).
    _, rB_chap_h = query_radii(C_pos, F, B, K, self_in_F=False, self_in_B=False)
    rF_chap_h, _ = query_radii(C_pos, F, B, K, self_in_F=False, self_in_B=False)
    chap_groups = greedy_location_groups(C_pos, rF_chap_h)
    LOGGER.info("book groups: %d (of %d locs); chapter groups: %d (of %d locs)",
                np.unique(book_groups).size, B_pos.shape[0],
                np.unique(chap_groups).size, C_pos.shape[0])

    # ===================================================== AUC table
    results: dict = {
        "meta": {
            "subfield": args.subfield, "field": args.field,
            "n_books_scope": int(B.shape[0]), "n_articles_scope": int(F.shape[0]),
            "n_book_pos_habitable": int(B_pos.shape[0]),
            "n_chap_pos_habitable": int(C_pos.shape[0]),
            "null_pool_size": int(pool_idx.size),
            "n_null_matched": int(Null_matched.shape[0]),
            "n_null_random": int(Null_random.shape[0]),
            "m_controls": int(args.m_controls), "n_boot": int(args.n_boot),
            "hull_q": HULL_Q, "k": K, "radius_fraction": RADIUS_FRACTION,
            "matcher_info": info, "tocs": chap_meta,
            "definitions": DEFS,
        },
        "matcher_calibration_random50": {d: round(calib[d], 4) for d in DEFS},
        "book_level": {"matched": {}, "random": {}},
        "chapter_level": {"matched": {}, "random": {}},
    }
    for d in DEFS:
        results["book_level"]["matched"][d] = clustered_auc_ci(
            s_books[d], s_null_m[d], book_groups, n_boot=args.n_boot, seed=args.seed)
        results["book_level"]["random"][d] = clustered_auc_ci(
            s_books[d], s_null_r[d], book_groups, n_boot=args.n_boot, seed=args.seed)
        results["chapter_level"]["matched"][d] = clustered_auc_ci(
            s_chap[d], s_null_m[d], chap_groups, n_boot=args.n_boot, seed=args.seed)
        results["chapter_level"]["random"][d] = clustered_auc_ci(
            s_chap[d], s_null_r[d], chap_groups, n_boot=args.n_boot, seed=args.seed)

    # ranked winners (by matched-null AUC).
    def ranked(block: dict) -> list:
        return sorted(block.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    results["winners"] = {
        "book_matched": [(d, r["auc"]) for d, r in ranked(results["book_level"]["matched"])],
        "chapter_matched": [(d, r["auc"]) for d, r in ranked(results["chapter_level"]["matched"])],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    _print_tables(results)
    LOGGER.info("wrote %s (total %.1fs)", args.out, time.time() - t0)


# ============================================================ helpers
def _knn_radii_chunked(Q: np.ndarray, R: np.ndarray, k: int, *, ref_is_query: bool,
                       qchunk: int = 20000) -> np.ndarray:
    """knn_radii over many query rows, chunked to bound memory (1.5M x 14k)."""
    out = np.empty(Q.shape[0], dtype=np.float64)
    for s in range(0, Q.shape[0], qchunk):
        out[s : s + qchunk] = knn_radii(Q[s : s + qchunk], R, k, ref_is_query=ref_is_query)
    return out


def _matcher_calibration(
    F, B, Pool, pool_idx, C_pool, rF_pool, rB_pool, art_habitable, rB_art,
    hab_mu, hab_sd, book_mix, m_controls, rng,
) -> dict:
    """random50: draw n_rand random habitable article pseudo-positives FROM the pool,
    match them to controls within the pool, and report AUC(rand vs matched) per g.
    A well-behaved matcher gives ~0.5 everywhere (no covariate leakage)."""
    n_rand = min(400, Pool.shape[0] // 4)
    loc_sel = rng.choice(Pool.shape[0], size=n_rand, replace=False)
    excluded = np.zeros(Pool.shape[0], dtype=bool)
    excluded[loc_sel] = True                                     # don't match a loc to itself
    ctrl_idx, _ = matched_controls(
        loc_X=Pool[loc_sel], loc_radii=rF_pool[loc_sel], loc_C=C_pool[loc_sel],
        pool_X=Pool, pool_radii=rF_pool, pool_C=C_pool,
        excluded=excluded, m_controls=m_controls, calipers=CALIPERS, seed=0,
    )
    ctrl_sel = np.unique(ctrl_idx.ravel())
    book_mix._loo_book = False
    s_loc = score_definitions(Pool[loc_sel], F, B, self_in_F=True, self_in_B=False,
                              hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)
    s_ctl = score_definitions(Pool[ctrl_sel], F, B, self_in_F=True, self_in_B=False,
                              hab_mu=hab_mu, hab_sd=hab_sd, book_mix=book_mix)
    return {d: auc(s_loc[d], s_ctl[d]) for d in DEFS}


def _print_tables(results: dict) -> None:
    m = results["meta"]
    print("\n" + "=" * 78)
    print(f"BOOK-SURPRISE BAKE-OFF  (subfield={m['subfield']}, k={m['k']}, hull_q={m['hull_q']})")
    print(f"  scope: {m['n_books_scope']} books, {m['n_articles_scope']} articles")
    print(f"  positives: {m['n_book_pos_habitable']} habitable books, "
          f"{m['n_chap_pos_habitable']} habitable chapters")
    print(f"  null: matched={m['n_null_matched']}, random={m['n_null_random']} "
          f"(pool {m['null_pool_size']}, m_controls={m['m_controls']})")
    print("=" * 78)
    print("\nMATCHER CALIBRATION (random50; should be ~0.50 for every g):")
    for d in DEFS:
        v = results["matcher_calibration_random50"][d]
        flag = "" if abs(v - 0.5) <= 0.07 else "  <-- off"
        print(f"  {d:<14} {v:.3f}{flag}")

    def show(level: str, null: str) -> None:
        block = results[level][null]
        order = sorted(block.items(), key=lambda kv: kv[1]["auc"], reverse=True)
        print(f"\n{level.upper().replace('_',' ')}  vs  {null.upper()} null   "
              f"(AUC books-exceptional > null, [95% clustered CI]):")
        print(f"  {'definition':<14} {'AUC':>6}  {'95% CI':>16}  {'n_pos':>6} {'n_grp':>6}")
        for d, r in order:
            ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            print(f"  {d:<14} {r['auc']:>6.3f}  {ci:>16}  {r['n_pos']:>6} {r['n_groups']:>6}")

    for level in ("book_level", "chapter_level"):
        for null in ("matched", "random"):
            show(level, null)

    print("\nWINNERS (by matched-null AUC):")
    print(f"  book-level   : {results['winners']['book_matched'][0]}")
    print(f"  chapter-level: {results['winners']['chapter_matched'][0]}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
