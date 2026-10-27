"""C4 — Application portfolio.

Sub-components (docs/components/C4-portfolio.md):
    C4.1 exact solvers under independence (greedy, DP, brute force) — done, verified
    C4.2 correlated-admission evaluator (Gaussian copula Monte Carlo) — W-29
    C4.3 robust choice under interval-valued P(admit)                — W-2A
    C4.4 utility elicitation (MAUT weights -> u_j)                   — v0 (utility.py)
"""

from aisc.portfolio.correlated import (
    admission_draws,
    portfolio_value_mc,
    recommend_portfolio,
    subsets_within_budget,
)
from aisc.portfolio.robust import (
    box_scenarios,
    max_regret,
    minimax_regret_portfolio,
    values_matrix,
    worst_case_portfolio,
)
from aisc.portfolio.solvers import brute_force, dp_varied_cost, greedy_same_cost, portfolio_value
from aisc.portfolio.utility import ahp_weights, multi_attribute_utility, scale_linear

__all__ = [
    "admission_draws",
    "ahp_weights",
    "box_scenarios",
    "brute_force",
    "dp_varied_cost",
    "greedy_same_cost",
    "max_regret",
    "minimax_regret_portfolio",
    "multi_attribute_utility",
    "portfolio_value",
    "portfolio_value_mc",
    "recommend_portfolio",
    "scale_linear",
    "subsets_within_budget",
    "values_matrix",
    "worst_case_portfolio",
]
