"""V1: solver correctness.

- Reproduce the published OptimalApplication.jl example.
- Match exhaustive search on random small instances (same cost and varied cost).
- Check the nestedness property for identical costs.
"""

import random

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
