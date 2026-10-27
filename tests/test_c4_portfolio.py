"""C4.1 verification (validation level L2).

- Reproduce the published OptimalApplication.jl example.
- Match exhaustive search on random small instances (same cost and varied cost).
- Check the nestedness property for identical costs.
"""

import random

import numpy as np
import pytest

from aisc.portfolio import brute_force, dp_varied_cost, greedy_same_cost, portfolio_value

TOL = 1e-9


def test_reproduces_optimalapplication_jl_readme_example():
    # From the README of github.com/maxkapur/OptimalApplication.jl (1-indexed there).
    p = [0.2, 0.5, 0.1, 0.6, 0.1]
    u = [1, 4, 9, 1, 8]
    order, values = greedy_same_cost(p, u)
    assert [j + 1 for j in order] == [2, 3, 5, 4, 1]
    assert values == pytest.approx([2.0, 2.7, 3.24, 3.483, 3.5154], abs=1e-9)


def test_portfolio_value_hand_computed():
    # Apply to school A (u=10, p=0.3) and B (u=5, p=0.8):
    # 0.3*10 + 0.7*0.8*5 = 3 + 2.8 = 5.8
    assert portfolio_value([0.3, 0.8], [10, 5], [0, 1]) == pytest.approx(5.8)
    assert portfolio_value([0.3, 0.8], [10, 5], []) == 0.0


def _random_instance(rng: random.Random, m: int):
    p = [round(rng.uniform(0.02, 0.95), 3) for _ in range(m)]
    u = [rng.randint(1, 100) for _ in range(m)]
    cost = [rng.randint(1, 6) for _ in range(m)]
    return p, u, cost


@pytest.mark.parametrize("seed", range(40))
def test_greedy_matches_brute_force_same_cost(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 10)
    p, u, _ = _random_instance(rng, m)
    order, values = greedy_same_cost(p, u)
    for h in range(1, m + 1):
        _, best = brute_force(p, u, h=h)
        assert values[h - 1] == pytest.approx(best, abs=TOL), f"h={h}"


@pytest.mark.parametrize("seed", range(40))
def test_dp_matches_brute_force_varied_cost(seed):
    rng = random.Random(1000 + seed)
    m = rng.randint(1, 10)
    p, u, cost = _random_instance(rng, m)
    budget = rng.randint(0, sum(cost))
    chosen, value = dp_varied_cost(p, u, cost, budget)
    _, best = brute_force(p, u, cost=cost, budget=budget)
    assert value == pytest.approx(best, abs=TOL)
    assert sum(cost[j] for j in chosen) <= budget
    assert portfolio_value(p, u, chosen) == pytest.approx(value, abs=TOL)


def test_dp_with_unit_costs_equals_greedy():
    rng = random.Random(7)
    p, u, _ = _random_instance(rng, 9)
    _, values = greedy_same_cost(p, u)
    for h in range(1, 10):
        _, v = dp_varied_cost(p, u, [1] * 9, h)
        assert v == pytest.approx(values[h - 1], abs=TOL)


def test_input_validation():
    with pytest.raises(ValueError):
        portfolio_value([1.2], [1], [0])
    with pytest.raises(ValueError):
        portfolio_value([0.5, 0.5], [1], [0])
    with pytest.raises(ValueError):
        dp_varied_cost([0.5], [1], [1.5], 3)


def test_recommendation_accounts_for_admission_correlation():
    """B-C4-1 (was a probe): with correlated admissions (rho = 0.7) the independence
    optimum is not the best portfolio; the recommendation matches brute force under
    the same Monte Carlo draws. Scenario found by random search over 6-school cases
    after the probe was written (the probe's first scenario had the same optimum under
    both models, so it could not separate them)."""
    import aisc.portfolio as pf

    p = [0.09, 0.53, 0.46, 0.11, 0.63, 0.82]
    u = [0.87, 0.8, 0.67, 0.61, 0.61, 0.41]
    cost, budget = [1] * 6, 3
    draws = pf.admission_draws(p, 0.7, 40_000, np.random.default_rng(0))
    vals = {s: pf.portfolio_value_mc(u, draws, s) for s in pf.subsets_within_budget(cost, budget)}
    best = max(vals.values())
    indep, _ = pf.dp_varied_cost(p, u, cost, budget)
    assert pf.portfolio_value_mc(u, draws, indep) < best - 0.02  # the blind spot is real
    chosen = pf.recommend_portfolio(p, u, cost, budget, rho=0.7, draws=draws)
    assert pf.portfolio_value_mc(u, draws, chosen) >= best - 1e-9


def test_mc_value_matches_analytic_when_independent():
    import aisc.portfolio as pf

    p = [0.2, 0.5, 0.8, 0.35]
    u = [1.0, 0.7, 0.4, 0.9]
    draws = pf.admission_draws(p, 0.0, 200_000, np.random.default_rng(1))
    for s in ([0, 1, 2], [3], [0, 3], [0, 1, 2, 3]):
        assert pf.portfolio_value_mc(u, draws, s) == pytest.approx(
            pf.portfolio_value(p, u, s), abs=0.006
        )


def test_mc_value_matches_comonotone_closed_form():
    """rho = 1: one common uniform U, school j admits iff U < p_j, so
    v(S) = integral_0^1 max{u_j : j in S, p_j > x} dx."""
    import aisc.portfolio as pf

    p = [0.2, 0.5, 0.8, 0.35]
    u = [1.0, 0.7, 0.4, 0.9]
    draws = pf.admission_draws(p, 1.0, 200_000, np.random.default_rng(2))
    for s in ([0, 1, 2], [0, 3], [1, 2]):
        cuts = sorted({0.0, 1.0, *(p[j] for j in s)})
        exact = sum(
            (b - a) * max([u[j] for j in s if p[j] > a] or [0.0])
            for a, b in zip(cuts, cuts[1:], strict=False)
        )
        assert pf.portfolio_value_mc(u, draws, s) == pytest.approx(exact, abs=0.006)


def test_marginals_preserved_and_rho_validated():
    import aisc.portfolio as pf

    p = [0.0, 0.1, 0.5, 0.9, 1.0]
    d = pf.admission_draws(p, 0.6, 100_000, np.random.default_rng(3))
    assert d.mean(axis=0) == pytest.approx(p, abs=0.005)
    with pytest.raises(ValueError):
        pf.admission_draws(p, 1.2, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        pf.admission_draws([1.1], 0.5, 10, np.random.default_rng(0))


def test_recommend_rho_zero_is_c41_dp_and_local_search_runs():
    import aisc.portfolio as pf

    p = [0.09, 0.53, 0.46, 0.11, 0.63, 0.82]
    u = [0.87, 0.8, 0.67, 0.61, 0.61, 0.41]
    cost = [1, 2, 1, 1, 2, 1]
    dp, _ = pf.dp_varied_cost(p, u, cost, 4)
    assert pf.recommend_portfolio(p, u, cost, 4) == sorted(dp)
    draws = pf.admission_draws(p, 0.7, 20_000, np.random.default_rng(0))
    exact = pf.recommend_portfolio(p, u, cost, 4, rho=0.7, draws=draws)
    local = pf.recommend_portfolio(p, u, cost, 4, rho=0.7, draws=draws, exact_limit=0)
    assert sum(cost[j] for j in local) <= 4
    v_dp = pf.portfolio_value_mc(u, draws, dp)
    assert pf.portfolio_value_mc(u, draws, local) >= v_dp - 1e-12
    assert pf.portfolio_value_mc(u, draws, exact) >= pf.portfolio_value_mc(u, draws, local) - 1e-12
