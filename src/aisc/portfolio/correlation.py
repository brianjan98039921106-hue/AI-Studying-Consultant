"""C4.2 correlated-admission evaluator (Gaussian copula).

C4.1 (solvers.py) values a portfolio S assuming independent admissions
(A-C4-1). Real admissions are positively correlated through applicant
strength (B-C4-1). This module keeps each program's marginal P(admit) p_j
and couples the outcomes with a Gaussian copula (Sklar 1959):

    Z ~ N(0, R),   admitted_j  <=>  Z_j < Phi^{-1}(p_j)

so P(admitted_j) = p_j exactly for any correlation matrix R. The value is the
same objective as C4.1, v(S) = E[max_{j in S, admitted} u_j] (0 if none).

Correlation parameterisation (docs/THEORY.md, C4.2): one-factor applicant
strength, Z_j = lambda_j F + sqrt(1 - lambda_j^2) eps_j, rho_ij = lambda_i
lambda_j (Vasicek 2002; Li 2000). rho is the *residual* correlation left after
C3 has used the applicant's observed features (A-C4-6). A general correlation
matrix can be passed instead (`corr=`).

Evaluators, each a check on the others:
- `mc_portfolio_value`: Monte Carlo for any R; reports the standard error.
- `factor_portfolio_value`: one-factor model; given F the admissions are
  independent, so v(S) = E_F[v_C4.1(S; p(F))], a 1-D integral computed on a grid
  (deterministic, no sampling error).
- `comonotone_portfolio_value`: the rho = 1 limit in closed form,
  v(S) = integral_0^1 max{u_j : j in S, p_j > t} dt.
- `brute_force_correlated`: exhaustive search under the one-factor model
  (reference for small m; shows what the independence optimum gives up).

At rho = 0 all three reduce to C4.1's `portfolio_value`, which stays the
reference implementation (D8). This module evaluates portfolios; it does not
re-optimise them (B-C4-1 stays open for the choice step).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from aisc.contracts import AdmitEstimate
from aisc.portfolio.solvers import portfolio_value

ProbLike = float | AdmitEstimate

_GRID_HALF_WIDTH = 8.0
_DEFAULT_GRID = 4001
_CHUNK = 50_000


@dataclass(frozen=True)
class CorrelatedValue:
    """Portfolio value under correlated admissions.

    `*_se` are Monte Carlo standard errors (sample sd / sqrt(n_draws)); they are
    0.0 for the deterministic evaluators (`n_draws` = 0 there).
    """

    expected_utility: float
    expected_utility_se: float
    p_any_admit: float
    p_any_admit_se: float
    n_draws: int
    method: str


@dataclass(frozen=True)
class SweepRow:
    """One rho of `correlation_sweep`: how much independence overstates v(S)."""

    rho: float
    independent: float
    correlated: float
    p_any_independent: float
    p_any_correlated: float

    @property
    def overestimate(self) -> float:
        """Relative overstatement (v_indep - v_rho) / v_rho; 0 when v_rho = 0."""
        if self.correlated <= 0:
            return 0.0
        return (self.independent - self.correlated) / self.correlated


# --- public API -----------------------------------------------------------------


def one_factor_corr(m: int, rho: float | None = None, loadings: Sequence[float] | None = None):
    """Correlation matrix of the one-factor model: R_ij = lambda_i lambda_j, R_ii = 1.

    Give either a common `rho` in [0, 1] (lambda_j = sqrt(rho)) or per-program
    `loadings` lambda_j in [0, 1].
    """
    lam = _loadings(m, rho, loadings)
    r = np.outer(lam, lam)
    np.fill_diagonal(r, 1.0)
    return r


def mc_portfolio_value(
    p: Sequence[ProbLike],
    u: Sequence[float],
    portfolio: Sequence[int],
    *,
    rho: float | None = None,
    loadings: Sequence[float] | None = None,
    corr: np.ndarray | Sequence[Sequence[float]] | None = None,
    n_draws: int = 100_000,
    seed: int | np.random.Generator | None = 0,
) -> CorrelatedValue:
    """Monte Carlo estimate of v(S) and P(at least one admit) under a Gaussian copula.

    Correlation: `corr` (full m x m matrix over all programs), or the one-factor
    model via `rho` / `loadings`; nothing given means independence. Results are
    reproducible for a fixed integer `seed`.
    """
    probs, util = _prepare(p, u)
    sel = _portfolio(portfolio, len(probs))
    if n_draws < 2:
        raise ValueError("n_draws must be >= 2")
    r = _corr_matrix(len(probs), rho, loadings, corr)
    if not sel:
        return CorrelatedValue(0.0, 0.0, 0.0, 0.0, n_draws, "copula-mc")

    r_s = r[np.ix_(sel, sel)]
    root = _psd_root(r_s)
    thresh = np.array([_ndtri(probs[j]) for j in sel])
    u_s = np.array([util[j] for j in sel])
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    s_v = s_v2 = s_a = 0.0
    done = 0
    while done < n_draws:
        n = min(_CHUNK, n_draws - done)
        z = rng.standard_normal((n, len(sel))) @ root.T
        admitted = z < thresh
        best = np.where(admitted, u_s, 0.0).max(axis=1)
        s_v += float(best.sum())
        s_v2 += float((best * best).sum())
        s_a += float(admitted.any(axis=1).sum())
        done += n

    mean_v = s_v / n_draws
    var_v = max(s_v2 / n_draws - mean_v**2, 0.0) * n_draws / (n_draws - 1)
    p_any = s_a / n_draws
    se_any = math.sqrt(p_any * (1 - p_any) / (n_draws - 1))
    return CorrelatedValue(mean_v, math.sqrt(var_v / n_draws), p_any, se_any, n_draws, "copula-mc")


def factor_portfolio_value(
    p: Sequence[ProbLike],
    u: Sequence[float],
    portfolio: Sequence[int],
    *,
    rho: float | None = None,
    loadings: Sequence[float] | None = None,
    n_grid: int = _DEFAULT_GRID,
) -> CorrelatedValue:
    """v(S) and P(any admit) under the one-factor copula, by integrating over F.

    Given F, admissions are independent with
        p_j(F) = Phi((Phi^{-1}(p_j) - lambda_j F) / sqrt(1 - lambda_j^2)),
    so v(S) = E_F[portfolio_value(p(F), u, S)]. Deterministic (trapezoid rule on
    F in [-8, 8]). Loadings of exactly 1 are routed to the comonotone closed
    form when all loadings are 1; a mix of 1 and < 1 is integrated with the
    indicator 1[F < Phi^{-1}(p_j)] for the fully loaded programs.
    """
    probs, util = _prepare(p, u)
    sel = _portfolio(portfolio, len(probs))
    lam = _loadings(len(probs), rho, loadings)
    if not sel:
        return CorrelatedValue(0.0, 0.0, 0.0, 0.0, 0, "copula-factor")
    if all(lam[j] >= 1.0 for j in sel):
        return comonotone_portfolio_value(probs, util, sel)

    f, w = _grid(n_grid)
    cond = {j: _conditional_p(probs[j], lam[j], f) for j in sel}
    value, none_yet = _factor_value(cond, util, sel)
    return CorrelatedValue(
        float(w @ value), 0.0, float(1.0 - w @ none_yet), 0.0, 0, "copula-factor"
    )


def brute_force_correlated(
    p: Sequence[ProbLike],
    u: Sequence[float],
    *,
    rho: float | None = None,
    loadings: Sequence[float] | None = None,
    h: int | None = None,
    cost: Sequence[int] | None = None,
    budget: int | None = None,
    n_grid: int = _DEFAULT_GRID,
) -> tuple[list[int], float]:
    """Exhaustive search for the best portfolio under the one-factor copula.

    Reference implementation (like C4.1's `brute_force`), m <= ~12: it shows
    what the independence optimum gives up; it is not the production search.
    Limits as in `brute_force`: at most `h` programs and/or total cost <= budget.
    """
    probs, util = _prepare(p, u)
    m = len(probs)
    lam = _loadings(m, rho, loadings)
    if cost is not None and len(cost) != m:
        raise ValueError("cost must have the same length as p")
    if all(x >= 1.0 for x in lam):
        evaluate = lambda s: comonotone_portfolio_value(probs, util, s).expected_utility  # noqa: E731
    else:
        f, w = _grid(n_grid)
        cond = {j: _conditional_p(probs[j], lam[j], f) for j in range(m)}

        def evaluate(s):
            return float(w @ _factor_value(cond, util, s)[0])

    best_set: tuple[int, ...] = ()
    best_val = 0.0
    for size in range(0, (m if h is None else min(h, m)) + 1):
        for subset in combinations(range(m), size):
            if cost is not None and budget is not None and sum(cost[j] for j in subset) > budget:
                continue
            val = evaluate(subset) if subset else 0.0
            if val > best_val + 1e-12:
                best_set, best_val = subset, val
    return list(best_set), best_val


def comonotone_portfolio_value(
    p: Sequence[ProbLike], u: Sequence[float], portfolio: Sequence[int]
) -> CorrelatedValue:
    """rho = 1 limit: one common uniform U, admitted_j <=> U < p_j.

    The admitted set is always the programs with the largest p_j, so
    v(S) = sum_k (p_(k) - p_(k+1)) max_{i <= k} u_(i) with p sorted descending,
    and P(any admit) = max_j p_j.
    """
    probs, util = _prepare(p, u)
    sel = _portfolio(portfolio, len(probs))
    if not sel:
        return CorrelatedValue(0.0, 0.0, 0.0, 0.0, 0, "copula-comonotone")
    by_p = sorted(sel, key=lambda j: probs[j], reverse=True)
    value, best_u = 0.0, 0.0
    for k, j in enumerate(by_p):
        best_u = max(best_u, util[j])
        nxt = probs[by_p[k + 1]] if k + 1 < len(by_p) else 0.0
        value += (probs[j] - nxt) * best_u
    return CorrelatedValue(value, 0.0, probs[by_p[0]], 0.0, 0, "copula-comonotone")


def correlation_sweep(
    p: Sequence[ProbLike],
    u: Sequence[float],
    portfolio: Sequence[int],
    rhos: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
) -> list[SweepRow]:
    """Independence value vs. one-factor value over a range of rho (B-C4-8).

    rho is not observable yet, so report the range rather than one number.
    """
    probs, util = _prepare(p, u)
    sel = _portfolio(portfolio, len(probs))
    indep = portfolio_value(probs, util, sel)
    p_any_indep = 1.0 - math.prod(1.0 - probs[j] for j in sel)
    rows = []
    for rho in rhos:
        c = factor_portfolio_value(probs, util, sel, rho=rho)
        rows.append(SweepRow(float(rho), indep, c.expected_utility, p_any_indep, c.p_any_admit))
    return rows


# --- helpers --------------------------------------------------------------------


def _prepare(p: Sequence[ProbLike], u: Sequence[float]) -> tuple[list[float], list[float]]:
    probs = [x.mean if isinstance(x, AdmitEstimate) else float(x) for x in p]
    util = [float(x) for x in u]
    if len(probs) != len(util):
        raise ValueError("p and u must have the same length")
    if any(not 0 <= x <= 1 for x in probs):
        raise ValueError("Probabilities must lie in [0, 1]")
    if any(not x >= 0 for x in util):
        raise ValueError("Utilities must be non-negative")
    return probs, util


def _portfolio(portfolio: Sequence[int], m: int) -> list[int]:
    sel = [int(j) for j in portfolio]
    if any(j < 0 or j >= m for j in sel):
        raise ValueError("Portfolio index out of range")
    if len(set(sel)) != len(sel):
        raise ValueError("Portfolio indices must be distinct")
    return sel


def _loadings(m: int, rho: float | None, loadings: Sequence[float] | None) -> list[float]:
    if rho is not None and loadings is not None:
        raise ValueError("Give rho or loadings, not both")
    if loadings is not None:
        lam = [float(x) for x in loadings]
        if len(lam) != m:
            raise ValueError("loadings must have one entry per program")
        if any(not 0 <= x <= 1 for x in lam):
            raise ValueError("loadings must lie in [0, 1]")
        return lam
    rho = 0.0 if rho is None else float(rho)
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1] (one-factor model, A-C4-6)")
    return [math.sqrt(rho)] * m


def _corr_matrix(m, rho, loadings, corr) -> np.ndarray:
    if corr is None:
        return one_factor_corr(m, rho, loadings)
    if rho is not None or loadings is not None:
        raise ValueError("Give corr or rho/loadings, not both")
    r = np.asarray(corr, dtype=float)
    if r.shape != (m, m):
        raise ValueError("corr must be m x m")
    if not np.all(np.isfinite(r)) or not np.allclose(r, r.T, atol=1e-12):
        raise ValueError("corr must be finite and symmetric")
    if not np.allclose(np.diag(r), 1.0, atol=1e-12) or np.any(np.abs(r) > 1 + 1e-12):
        raise ValueError("corr must have unit diagonal and entries in [-1, 1]")
    if np.linalg.eigvalsh(r).min() < -1e-9:
        raise ValueError("corr must be positive semi-definite")
    return r


def _psd_root(r: np.ndarray) -> np.ndarray:
    """L with L L^T = r; symmetric eigen-decomposition so singular r (rho = 1) works."""
    vals, vecs = np.linalg.eigh(r)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def _grid(n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid nodes and normalised N(0, 1) weights on F in [-8, 8]."""
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3")
    f = np.linspace(-_GRID_HALF_WIDTH, _GRID_HALF_WIDTH, n_grid)
    w = np.exp(-0.5 * f * f)
    w[0] *= 0.5
    w[-1] *= 0.5
    return f, w / w.sum()


def _factor_value(cond, util, sel):
    """C4.1 recursion at every grid point F, given conditional P(admit | F)."""
    some = next(iter(cond.values()))
    none_yet = np.ones_like(some)
    value = np.zeros_like(some)
    for j in sorted(sel, key=lambda j: util[j], reverse=True):
        value += none_yet * cond[j] * util[j]
        none_yet = none_yet * (1.0 - cond[j])
    return value, none_yet


_erfc = np.frompyfunc(math.erfc, 1, 1)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0)).astype(float)


def _conditional_p(pj: float, lam: float, f: np.ndarray) -> np.ndarray:
    t = _ndtri(pj)
    if math.isinf(t):
        return np.full_like(f, 1.0 if t > 0 else 0.0)
    if lam >= 1.0:
        return (f < t).astype(float)
    return _norm_cdf((t - lam * f) / math.sqrt(1.0 - lam * lam))


def _ndtri(q: float) -> float:
    """Standard normal quantile Phi^{-1}(q) (Acklam's rational approximation
    refined by Halley steps on erfc; |error| < 1e-9 on [1e-9, 1 - 1e-9], checked
    against scipy in the tests). numpy-only on purpose: scipy is a dev dependency."""
    if q <= 0.0:
        return -math.inf
    if q >= 1.0:
        return math.inf
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)  # fmt: skip
    lo = 0.02425
    if q < lo:
        s = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / (
            (((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1
        )
    elif q > 1 - lo:
        s = math.sqrt(-2 * math.log(1 - q))
        x = -(((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / (
            (((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1
        )
    else:
        s = q - 0.5
        r = s * s
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * s
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    for _ in range(2):
        e = 0.5 * math.erfc(-x / math.sqrt(2)) - q
        g = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
        x -= g / (1 + x * g / 2)
    return x
