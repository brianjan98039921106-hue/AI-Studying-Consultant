"""C4 — Application portfolio.

Sub-components (docs/components/C4-portfolio.md):
    C4.1 exact solvers under independence (greedy, DP, brute force) — done, verified
    C4.2 correlated-admission evaluator (Gaussian copula Monte Carlo) — planned
    C4.3 robust choice under interval-valued P(admit)                — planned
    C4.4 utility elicitation (MAUT weights -> u_j)                   — planned
"""

from aisc.portfolio.solvers import brute_force, dp_varied_cost, greedy_same_cost, portfolio_value

__all__ = ["brute_force", "dp_varied_cost", "greedy_same_cost", "portfolio_value"]
