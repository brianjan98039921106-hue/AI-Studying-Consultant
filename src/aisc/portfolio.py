"""L4 application portfolio optimisation (the college application problem).

Choose a set S of programs to apply to, maximising the expected utility of
the best admission:

    v(S) = E[ max_{j in S, admitted} u_j ]     (0 if no admission)

with independent admissions, P(admit j) = p_j. Sorting S by utility in
descending order j_1, j_2, ...:

    v(S) = sum_i u_{j_i} p_{j_i} prod_{l<i} (1 - p_{j_l})

Reference: Kapur & Hong (2022), "The College Application Problem",
arXiv:2205.01869; Julia implementation maxkapur/OptimalApplication.jl.

- Same application cost, at most h schools: greedy is optimal and the
  optimal portfolios are nested in h (Kapur & Hong).
- Different costs, budget H: NP-complete in general; exact pseudo-polynomial
  DP over integer costs below.

Known limitation (see docs/ROADMAP.md, M4): admissions are assumed
independent. In reality they are positively correlated through applicant
quality, which makes the independent model over-optimistic about "reach"
portfolios.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations


def portfolio_value(p: Sequence[float], u: Sequence[float], portfolio: Sequence[int]) -> float:
    """Expected utility of the best admission in `portfolio` (indices)."""
    _check(p, u)
    value, none_yet = 0.0, 1.0
    for j in sorted(portfolio, key=lambda j: u[j], reverse=True):
        value += none_yet * p[j] * u[j]
        none_yet *= 1.0 - p[j]
    return value


def greedy_same_cost(
    p: Sequence[float], u: Sequence[float], h: int | None = None
) -> tuple[list[int], list[float]]:
    """Greedy application order for identical application costs.

    Returns (order, values): applying to order[:k] is optimal for a limit of
    k schools, and values[k-1] is its expected utility.
    """
    _check(p, u)
    m = len(p)
    h = m if h is None else min(h, m)
    chosen: list[int] = []
    values: list[float] = []
    remaining = set(range(m))
    for _ in range(h):
        best = max(remaining, key=lambda j: (portfolio_value(p, u, chosen + [j]), -j))
        chosen.append(best)
        remaining.remove(best)
        values.append(portfolio_value(p, u, chosen))
    return chosen, values


def dp_varied_cost(
    p: Sequence[float], u: Sequence[float], cost: Sequence[int], budget: int
) -> tuple[list[int], float]:
    """Exact DP for integer application costs and an integer budget.

    Process schools in ascending utility. If school j is added to a portfolio
    drawn from schools with lower utility, j is the new top choice, so
        V[j][b] = max(V[j-1][b], p_j u_j + (1 - p_j) V[j-1][b - c_j]).
    Time O(m * budget).
    """
    _check(p, u)
    if len(cost) != len(p):
        raise ValueError("cost must have the same length as p")
    if any(c < 0 or int(c) != c for c in cost) or budget < 0:
        raise ValueError("Costs and budget must be non-negative integers")

    order = sorted(range(len(p)), key=lambda j: u[j])
    prev = [0.0] * (budget + 1)
    take: list[list[bool]] = []
    for j in order:
        cur = prev[:]
        took = [False] * (budget + 1)
        for b in range(cost[j], budget + 1):
            cand = p[j] * u[j] + (1 - p[j]) * prev[b - cost[j]]
            if cand > cur[b] + 1e-15:
                cur[b], took[b] = cand, True
        take.append(took)
        prev = cur

    # Backtrack.
    chosen: list[int] = []
    b = budget
    for idx in range(len(order) - 1, -1, -1):
        j = order[idx]
        if take[idx][b]:
            chosen.append(j)
            b -= cost[j]
    return sorted(chosen), prev[budget]


def brute_force(
    p: Sequence[float],
    u: Sequence[float],
    cost: Sequence[int] | None = None,
    budget: int | None = None,
    h: int | None = None,
) -> tuple[list[int], float]:
    """Exhaustive search. Reference implementation for tests; m <= ~15."""
    _check(p, u)
    m = len(p)
    best_set: tuple[int, ...] = ()
    best_val = 0.0
    sizes = range(0, (h if h is not None else m) + 1)
    for size in sizes:
        for subset in combinations(range(m), size):
            if cost is not None and budget is not None and sum(cost[j] for j in subset) > budget:
                continue
            val = portfolio_value(p, u, subset)
            if val > best_val + 1e-12:
                best_set, best_val = subset, val
    return list(best_set), best_val


def _check(p: Sequence[float], u: Sequence[float]) -> None:
    if len(p) != len(u):
        raise ValueError("p and u must have the same length")
    if any(not 0 <= x <= 1 for x in p):
        raise ValueError("Probabilities must lie in [0, 1]")
    if any(x < 0 for x in u):
        raise ValueError("Utilities must be non-negative")
