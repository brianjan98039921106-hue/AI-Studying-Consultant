"""C4 — Application portfolio.

Sub-components (docs/components/C4-portfolio.md):
    C4.1 exact solvers under independence (greedy, DP, brute force) — done, verified
    C4.2 correlated-admission evaluator (Gaussian copula Monte Carlo) — W-29
    C4.3 robust choice under interval-valued P(admit)                — planned
    C4.4 utility elicitation (MAUT weights -> u_j)                   — planned
"""

from aisc.portfolio.correlated import (
    admission_draws,
    portfolio_value_mc,
    recommend_portfolio,
    subsets_within_budget,
)
from aisc.portfolio.solvers import brute_force, dp_varied_cost, greedy_same_cost, portfolio_value

__all__ = [
    "admission_draws",
    "brute_force",
    "dp_varied_cost",
    "greedy_same_cost",
    "portfolio_value",
    "portfolio_value_mc",
    "recommend_portfolio",
    "subsets_within_budget",
]
