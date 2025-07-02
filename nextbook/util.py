import logging
from functools import wraps
import inspect
import math
import joblib
from math import gcd
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import scipy.linalg
from scipy.stats import distributions
from time import time
from fractions import Fraction

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
                self.print(f'func:{f.__name__}({filtered_s}) took {elapsed_time:.2f} seconds')
                return result
            return wrap
        return decorator


logger = logging.getLogger("nextbook")

TIMER = Timer(print=logger.info)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY = joblib.Memory(CACHE_DIR / "joblib", verbose=1)

class KstestResult:
    def __init__(self, statistic, pvalue: Fraction, statistic_location, statistic_sign):
        self.statistic = statistic
        self.pvalue = pvalue
        self.log_pvalue = math.log(pvalue.numerator) - math.log(pvalue.denominator)
        self.statistic_location = statistic_location
        self.statistic_sign = statistic_sign

    # def log_pvalue(self) -> float:
    #     return math.log(self.pvalue.numerator) - math.log(self.pvalue.denominator)

    def __repr__(self):
        return (f"KstestResult(statistic={self.statistic}, "
                f"pvalue={self.log_pvalue}, "
                f"statistic_location={self.statistic_location}, "
                f"statistic_sign={self.statistic_sign})")

@TIMER.time(formatters=dict(data1=lambda x: f"<shape={x.shape}>", data2=lambda x: f"<shape={x.shape}>"))
def ks_2samp(data1, data2, alternative='two-sided', method='auto'):
    """
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.

    This test compares the underlying continuous distributions F(x) and G(x)
    of two independent samples.  See Notes for a description of the available
    null and alternative hypotheses.

    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes below.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    res: KstestResult
        An object containing attributes:

        statistic : float
            KS test statistic.
        pvalue : float
            One-tailed or two-tailed p-value.
        statistic_location : float
            Value from `data1` or `data2` corresponding with the KS statistic;
            i.e., the distance between the empirical distribution functions is
            measured at this observation.
        statistic_sign : int
            +1 if the empirical distribution function of `data1` exceeds
            the empirical distribution function of `data2` at
            `statistic_location`, otherwise -1.

    See Also
    --------
    kstest, ks_1samp, epps_singleton_2samp, anderson_ksamp

    Notes
    -----
    There are three options for the null and corresponding alternative
    hypothesis that can be selected using the `alternative` parameter.

    - `less`: The null hypothesis is that F(x) >= G(x) for all x; the
      alternative is that F(x) < G(x) for at least one x. The statistic
      is the magnitude of the minimum (most negative) difference between the
      empirical distribution functions of the samples.

    - `greater`: The null hypothesis is that F(x) <= G(x) for all x; the
      alternative is that F(x) > G(x) for at least one x. The statistic
      is the maximum (most positive) difference between the empirical
      distribution functions of the samples.

    - `two-sided`: The null hypothesis is that the two distributions are
      identical, F(x)=G(x) for all x; the alternative is that they are not
      identical. The statistic is the maximum absolute difference between the
      empirical distribution functions of the samples.

    Note that the alternative hypotheses describe the *CDFs* of the
    underlying distributions, not the observed values of the data. For example,
    suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in
    x1 tend to be less than those in x2.

    If the KS statistic is large, then the p-value will be small, and this may
    be taken as evidence against the null hypothesis in favor of the
    alternative.

    If ``method='exact'``, `ks_2samp` attempts to compute an exact p-value,
    that is, the probability under the null hypothesis of obtaining a test
    statistic value as extreme as the value computed from the data.
    If ``method='asymp'``, the asymptotic Kolmogorov-Smirnov distribution is
    used to compute an approximate p-value.
    If ``method='auto'``, an exact p-value computation is attempted if both
    sample sizes are less than 10000; otherwise, the asymptotic method is used.
    In any case, if an exact p-value calculation is attempted and fails, a
    warning will be emitted, and the asymptotic p-value will be returned.

    The 'two-sided' 'exact' computation computes the complementary probability
    and then subtracts from 1.  As such, the minimum probability it can return
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.

    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.

    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-486.

    Examples
    --------
    Suppose we wish to test the null hypothesis that two samples were drawn
    from the same distribution.
    We choose a confidence level of 95%; that is, we will reject the null
    hypothesis in favor of the alternative if the p-value is less than 0.05.

    If the first sample were drawn from a uniform distribution and the second
    were drawn from the standard normal, we would expect the null hypothesis
    to be rejected.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> sample1 = stats.uniform.rvs(size=100, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=110, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.5454545454545454,
                 pvalue=7.37417839555191e-15,
                 statistic_location=-0.014071496412861274,
                 statistic_sign=-1)


    Indeed, the p-value is lower than our threshold of 0.05, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the data
    were *not* drawn from the same distribution.

    When both samples are drawn from the same distribution, we expect the data
    to be consistent with the null hypothesis most of the time.

    >>> sample1 = stats.norm.rvs(size=105, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=95, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.10927318295739348,
                 pvalue=0.5438289009927495,
                 statistic_location=-0.1670157701848795,
                 statistic_sign=-1)

    As expected, the p-value of 0.54 is not below our threshold of 0.05, so
    we cannot reject the null hypothesis.

    Suppose, however, that the first sample were drawn from
    a normal distribution shifted toward greater values. In this case,
    the cumulative density function (CDF) of the underlying distribution tends
    to be *less* than the CDF underlying the second sample. Therefore, we would
    expect the null hypothesis to be rejected with ``alternative='less'``:

    >>> sample1 = stats.norm.rvs(size=105, loc=0.5, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2, alternative='less')
    KstestResult(statistic=0.4055137844611529,
                 pvalue=3.5474563068855554e-08,
                 statistic_location=-0.13249370614972575,
                 statistic_sign=-1)

    and indeed, with p-value smaller than our threshold, we reject the null
    hypothesis in favor of the alternative.

    """
    mode = method

    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    if np.ma.is_masked(data1):
        data1 = data1.compressed()
    if np.ma.is_masked(data2):
        data2 = data2.compressed()
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2

    # Identify the location of the statistic
    argminS = np.argmin(cddiffs)
    argmaxS = np.argmax(cddiffs)
    loc_minS = data_all[argminS]
    loc_maxS = data_all[argmaxS]

    # Ensure sign of minS is not negative.
    minS = np.clip(-cddiffs[argminS], 0, 1)
    maxS = cddiffs[argmaxS]

    if alternative == 'less' or (alternative == 'two-sided' and minS > maxS):
        d = minS
        d_location = loc_minS
        d_sign = -1
    else:
        d = maxS
        d_location = loc_maxS
        d_sign = 1
    g = math.gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    # prob = -np.inf
    prob = Fraction(-1, 1)  # Use Fraction to handle precision issues
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int32).max / n2g:
            mode = 'asymp'
            logger.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning,
                stacklevel=3)

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            logger.warn(f"ks_2samp: Exact calculation unsuccessful. "
                          f"Switching to method={mode}.", RuntimeWarning,
                          stacklevel=3)

    if mode == 'asymp':
        raise NotImplementedError("TODO use mpmath here")
        # The product n1*n2 is large.  Use Smirnov's asymptotic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    # prob = np.clip(prob, 0, 1)
    # prob = max(Fraction(0, 1), min(Fraction(1, 1), prob))

    # Currently, `d` is a Python float. We want it to be a NumPy type, so
    # float64 is appropriate. An enhancement would be for `d` to respect the
    # dtype of the input.
    return KstestResult(np.float64(d), prob, statistic_location=d_location,
                        statistic_sign=np.int8(d_sign))


def _attempt_exact_2kssamp(n1: int, n2: int, g: int, d, alternative):
    """Attempts to compute the exact 2sample probability.

    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs

    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    # d = h * 1.0 / lcm
    d = Fraction(h, lcm)
    if h == 0:
        # return True, d, 1.0
        return True, d, Fraction(1, 1)
    # saw_fp_error, prob = False, np.nan
    saw_fp_error, prob = False, Fraction(-1, 1)
    # try:
    with np.errstate(invalid="raise", over="raise"):
        if alternative == 'two-sided':
            if n1 == n2:
                prob = _compute_prob_outside_square(n1, h)
            else:
                raise NotImplementedError(f"Need {n1=} == {n2=} for this method")
                prob = _compute_outer_prob_inside_method(n1, n2, g, h)
        else:
            raise NotImplementedError("Only 'two-sided' alternative for now")
            if n1 == n2:
                # prob = binom(2n, n-h) / binom(2n, n)
                # Evaluating in that form incurs roundoff errors
                # from special.binom. Instead calculate directly
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                with np.errstate(over='raise'):
                    num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)
                if num_paths > bin or np.isinf(bin):
                    saw_fp_error = True
                else:
                    prob = num_paths / bin

    # except (FloatingPointError, OverflowError):
    #     saw_fp_error = True

    if saw_fp_error:
        # return False, d, np.nan
        return False, d, Fraction(-1, 1)
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob




def _compute_prob_outside_square(n: int, h: int):
    """
    Compute the proportion of paths that pass outside the two diagonal lines.

    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n

    Returns
    -------
    p : Fraction
        The proportion of paths that pass outside the lines x-y = +/-h.

    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )
    # / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    # P = 0.0
    P = Fraction(0, 1)
    k = int(np.floor(n / h))
    while k >= 0:
        # p1 = 1.0
        p1 = Fraction(1, 1)
        # Each of the Ai terms has numerator and denominator with
        # h simple terms.
        for j in range(h):
            # p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
            p1 *= Fraction(n - k * h - j, n + k * h + j + 1)
        # P = p1 * (1.0 - P)
        P = p1 * (Fraction(1, 1) - P)
        k -= 1
    return 2 * P