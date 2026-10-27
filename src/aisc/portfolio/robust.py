"""C4.3 robust choice over probability intervals (W-2A, B-C4-2; THEORY "穩健選擇",
A-C4-6, A-C4-7).

Each school's admission probability is only known to lie in [low_j, high_j] (C3
interval). Three criteria, all under independent admissions (A-C4-1):
    E  expected value at the point estimate   -> C4.1 (dp_varied_cost)
    W  worst case: max_S min_p v(S, p). v is nondecreasing in every p_j, so the inner
       minimum is p = low and W is the C4.1 optimum at `low`.
    R  minimax regret: min_S max_p [v*(p) - v(S, p)], the maximum taken over a finite
       scenario set drawn from the box (A-C4-7): the all-low, all-high and point
       corners plus independent uniform draws per school.
For at most `exact_limit` schools R enumerates every feasible portfolio; otherwise
the candidates are the C4.1 optima of every scenario (the regret-minimiser among
them).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aisc.portfolio.correlated import subsets_within_budget
from aisc.portfolio.solvers import _check, dp_varied_cost


def worst_case_portfolio(
    low: Sequence[float], u: Sequence[float], cost: Sequence[int], budget: int
) -> list[int]:
    """Criterion W: the C4.1 optimum when every probability is at its lower bound."""
    chosen, _ = dp_varied_cost(low, u, cost, budget)
    return sorted(chosen)


def box_scenarios(
    low: Sequence[float],
    high: Sequence[float],
    point: Sequence[float] | None,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(k, m) probability scenarios: all-low, all-high, the point estimate (if given)
    and n uniform draws inside the box."""
    lo, hi = np.asarray(low, float), np.asarray(high, float)
    if lo.shape != hi.shape or np.any(lo > hi) or np.any(lo < 0) or np.any(hi > 1):
        raise ValueError("need 0 <= low <= high <= 1 elementwise")
    rows = [lo, hi] + ([np.asarray(point, float)] if point is not None else [])
    if point is not None and np.any((rows[2] < lo - 1e-12) | (rows[2] > hi + 1e-12)):
        raise ValueError("the point estimate must lie inside [low, high]")
    draws = lo + (hi - lo) * rng.random((n, len(lo)))
    return np.vstack([*rows, draws])


def values_matrix(
    u: Sequence[float], scenarios: np.ndarray, portfolios: Sequence[Sequence[int]]
) -> np.ndarray:
    """v(S, p) under independence for every portfolio (rows) and scenario (columns)."""
    u = np.asarray(u, float)
    out = np.zeros((len(portfolios), len(scenarios)))
    for r, s in enumerate(portfolios):
        if not len(s):
            continue
        order = sorted(s, key=lambda j: u[j], reverse=True)
        P = scenarios[:, order]  # (k, |S|)
        miss_before = np.cumprod(np.hstack([np.ones((len(P), 1)), 1 - P[:, :-1]]), axis=1)
        out[r] = (P * miss_before) @ u[order]
    return out


def _best_values(u, cost, budget, scenarios) -> np.ndarray:
    return np.array([dp_varied_cost(list(p), list(u), list(cost), budget)[1] for p in scenarios])


def max_regret(
    u: Sequence[float],
    cost: Sequence[int],
    budget: int,
    scenarios: np.ndarray,
    portfolio: Sequence[int],
) -> float:
    """Largest regret v*(p) - v(S, p) of `portfolio` over the scenarios."""
    best = _best_values(u, cost, budget, scenarios)
    return float(np.max(best - values_matrix(u, scenarios, [portfolio])[0]))


def minimax_regret_portfolio(
    low: Sequence[float],
    high: Sequence[float],
    point: Sequence[float] | None,
    u: Sequence[float],
    cost: Sequence[int],
    budget: int,
    n_scenarios: int = 200,
    rng: np.random.Generator | None = None,
    scenarios: np.ndarray | None = None,
    exact_limit: int = 12,
) -> list[int]:
    """Criterion R over the box [low, high] (see module docstring)."""
    _check(low, u)
    _check(high, u)
    if len(cost) != len(u):
        raise ValueError("cost must have the same length as u")
    if scenarios is None:
        scenarios = box_scenarios(low, high, point, n_scenarios, rng or np.random.default_rng(0))
    best = _best_values(u, cost, budget, scenarios)
    if len(u) <= exact_limit:
        cands = subsets_within_budget(cost, budget)
    else:
        cands = sorted({tuple(sorted(dp_varied_cost(list(p), list(u), list(cost), budget)[0]))
                        for p in scenarios})  # fmt: skip
    regret = (best[None, :] - values_matrix(u, scenarios, cands)).max(axis=1)
    k = min(range(len(cands)), key=lambda i: (round(float(regret[i]), 12), len(cands[i]), i))
    return sorted(cands[k])
