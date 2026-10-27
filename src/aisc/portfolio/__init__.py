"""C4 — Application portfolio.

Sub-components (docs/components/C4-portfolio.md):
    C4.1 exact solvers under independence (greedy, DP, brute force) — done, verified
    C4.2 correlated-admission evaluator (Gaussian copula, one-factor)  — evaluator done (W-29)
    C4.3 robust choice under interval-valued P(admit)                — planned
    C4.4 utility elicitation (MAUT weights -> u_j)                   — planned
"""

from aisc.portfolio.correlation import (
    CorrelatedValue,
    SweepRow,
    brute_force_correlated,
    comonotone_portfolio_value,
    correlation_sweep,
    factor_portfolio_value,
    mc_portfolio_value,
    one_factor_corr,
)
from aisc.portfolio.solvers import brute_force, dp_varied_cost, greedy_same_cost, portfolio_value

__all__ = [
    "CorrelatedValue",
    "SweepRow",
    "brute_force",
    "brute_force_correlated",
    "comonotone_portfolio_value",
    "correlation_sweep",
    "dp_varied_cost",
    "factor_portfolio_value",
    "greedy_same_cost",
    "mc_portfolio_value",
    "one_factor_corr",
    "portfolio_value",
]
