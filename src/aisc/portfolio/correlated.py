"""C4.2 correlated admissions (W-29, B-C4-1; THEORY "相關性（C4.2）", A-C4-5).

One-factor Gaussian copula: for an applicant, Z_j = sqrt(rho) F + sqrt(1 - rho) E_j
with F, E_j ~ N(0, 1) independent, and school j admits when Z_j < Phi^-1(p_j). Each
marginal P(admit j) stays p_j; rho is the latent correlation from the applicant's
overall strength. rho = 0 is the independent model of C4.1.

    v(S) = E[ max_{j in S, admitted} u_j ]   (0 if none), by Monte Carlo

Comparisons between portfolios use common random numbers (the same draws), so their
differences have much smaller error than each value. The C4.1 solvers stay the
verified reference (D8); `recommend_portfolio` starts from the C4.1 DP solution and
improves it by local search under the correlated model.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from statistics import NormalDist

import numpy as np

from aisc.portfolio.solvers import _check, dp_varied_cost


def admission_draws(
    p: Sequence[float], rho: float, n_sim: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_sim, m) boolean admissions under the one-factor Gaussian copula."""
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    if n_sim < 1:
        raise ValueError("n_sim must be positive")
    p = np.asarray(p, float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    nd = NormalDist()
    thr = np.array([-np.inf if q == 0 else np.inf if q == 1 else nd.inv_cdf(q) for q in p])
    f = rng.standard_normal((n_sim, 1))
    e = rng.standard_normal((n_sim, len(p)))
    z = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
    return z < thr


def portfolio_value_mc(u: Sequence[float], draws: np.ndarray, portfolio: Sequence[int]) -> float:
    """Monte Carlo v(S) on given admission draws (expected utility of the best admit)."""
    if not len(portfolio):
        return 0.0
    u = np.asarray(u, float)
    order = sorted(portfolio, key=lambda j: u[j], reverse=True)
    adm = draws[:, order]
    any_adm = adm.any(axis=1)
    first = np.argmax(adm, axis=1)  # index of the best admitted school in `order`
    return float(np.where(any_adm, u[np.asarray(order)][first], 0.0).mean())


def subsets_within_budget(cost: Sequence[int], budget: int) -> list[tuple[int, ...]]:
    """All portfolios whose total cost fits the budget (reference for small m)."""
    m = len(cost)
    return [s for k in range(m + 1) for s in combinations(range(m), k)
            if sum(cost[j] for j in s) <= budget]  # fmt: skip


def recommend_portfolio(
    p: Sequence[float],
    u: Sequence[float],
    cost: Sequence[int],
    budget: int,
    rho: float = 0.0,
    n_sim: int = 20_000,
    rng: np.random.Generator | None = None,
    draws: np.ndarray | None = None,
    exact_limit: int = 12,
) -> list[int]:
    """Portfolio maximising v(S) under correlated admissions.

    rho = 0 without `draws` returns the exact C4.1 DP solution. Otherwise (including
    rho = 0 with `draws`, which then gives the Monte Carlo optimum on those draws):
    exhaustive search over the feasible portfolios when there are at most
    `exact_limit` schools, else local search (add / drop / swap, first improvement)
    from the C4.1 solution; all candidates are evaluated on the same draws. `draws`
    must have been generated with the same rho (B-C4-8: rho is validated either way)."""
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    _check(p, u)
    if len(cost) != len(p):
        raise ValueError("cost must have the same length as p")
    start, _ = dp_varied_cost(p, u, cost, budget)
    if rho == 0 and draws is None:
        return sorted(start)
    if draws is None:
        draws = admission_draws(p, rho, n_sim, rng or np.random.default_rng(0))
    if len(p) <= exact_limit:
        cands = subsets_within_budget(cost, budget)
        return sorted(max(cands, key=lambda s: (portfolio_value_mc(u, draws, s), -len(s))))
    cur = set(start)
    cur_v = portfolio_value_mc(u, draws, sorted(cur))
    improved = True
    while improved:
        improved = False
        spent = sum(cost[j] for j in cur)
        moves = [cur | {j} for j in range(len(p)) if j not in cur and spent + cost[j] <= budget]
        moves += [cur - {j} for j in cur]
        moves += [(cur - {i}) | {j} for i in cur for j in range(len(p))
                  if j not in cur and spent - cost[i] + cost[j] <= budget]  # fmt: skip
        for cand in moves:
            v = portfolio_value_mc(u, draws, sorted(cand))
            if v > cur_v + 1e-12:
                cur, cur_v, improved = cand, v, True
                break
    return sorted(cur)
