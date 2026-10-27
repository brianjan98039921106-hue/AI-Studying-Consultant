"""C4.2 correlated-admission evaluator verification (validation level L2).

- rho = 0: Monte Carlo matches the C4.1 exact value within Monte Carlo error.
  Tolerance: |MC - exact| <= Z_TOL * SE, with SE = sample sd / sqrt(n) as
  reported by the evaluator. Z_TOL = 4 (two-sided normal tail 6e-5 per check;
  with fixed seeds the test is deterministic, the bound states what a fair MC
  estimate must satisfy).
- rho -> 1: comonotone closed form; the factor integral and MC converge to it.
- Monte Carlo and the one-factor integral agree for intermediate rho.
- Fixed seeds reproduce results exactly; reported SE matches the spread of
  independent replications.
- L4 probe for B-C4-1: the C4.1 choice ignores correlation.
"""

import math
import random

import numpy as np
import pytest

from aisc.contracts import AdmitEstimate
from aisc.portfolio import (
    brute_force,
    brute_force_correlated,
    comonotone_portfolio_value,
    correlation_sweep,
    factor_portfolio_value,
    greedy_same_cost,
    mc_portfolio_value,
    one_factor_corr,
    portfolio_value,
)
from aisc.portfolio.correlation import _ndtri

Z_TOL = 4.0
N_MC = 40_000


def _instance(seed: int, m: int | None = None):
    rng = random.Random(seed)
    m = m or rng.randint(1, 8)
    p = [round(rng.uniform(0.02, 0.95), 3) for _ in range(m)]
    u = [rng.randint(1, 100) for _ in range(m)]
    s = sorted(rng.sample(range(m), rng.randint(1, m)))
    return p, u, s


def _p_any_indep(p, s):
    return 1.0 - math.prod(1.0 - p[j] for j in s)


@pytest.mark.parametrize("seed", range(20))
def test_rho_zero_mc_matches_c41_exact_within_se(seed):
    p, u, s = _instance(seed)
    exact = portfolio_value(p, u, s)
    r = mc_portfolio_value(p, u, s, rho=0.0, n_draws=N_MC, seed=seed)
    assert r.expected_utility_se > 0
    assert abs(r.expected_utility - exact) <= Z_TOL * r.expected_utility_se
    assert abs(r.p_any_admit - _p_any_indep(p, s)) <= Z_TOL * max(r.p_any_admit_se, 1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_rho_zero_factor_equals_c41_exact(seed):
    p, u, s = _instance(seed)
    r = factor_portfolio_value(p, u, s, rho=0.0)
    assert r.expected_utility == pytest.approx(portfolio_value(p, u, s), abs=1e-9)
    assert r.p_any_admit == pytest.approx(_p_any_indep(p, s), abs=1e-12)


def test_comonotone_closed_form_hand_computed():
    # U ~ Uniform(0,1); A (u=10, p=0.3) admits iff U < 0.3, B (u=5, p=0.8) iff U < 0.8.
    # U < 0.3: both -> 10; 0.3 <= U < 0.8: only B -> 5; else none.
    r = comonotone_portfolio_value([0.3, 0.8], [10, 5], [0, 1])
    assert r.expected_utility == pytest.approx(0.3 * 10 + 0.5 * 5)
    assert r.p_any_admit == pytest.approx(0.8)
    # Safer school also the better one: the riskier one adds nothing.
    assert comonotone_portfolio_value([0.9, 0.3], [10, 50], [0, 1]).expected_utility == (
        pytest.approx(0.3 * 50 + 0.6 * 10)
    )


@pytest.mark.parametrize("seed", range(10))
def test_rho_one_limit_is_comonotone(seed):
    p, u, s = _instance(100 + seed)
    como = comonotone_portfolio_value(p, u, s).expected_utility
    assert factor_portfolio_value(p, u, s, rho=1.0).expected_utility == pytest.approx(como)
    # Continuity: the integral approaches the closed form as rho -> 1.
    near = factor_portfolio_value(p, u, s, rho=1 - 1e-8, n_grid=20001).expected_utility
    assert near == pytest.approx(como, abs=1e-2 * max(u))
    mc = mc_portfolio_value(p, u, s, rho=1.0, n_draws=N_MC, seed=seed)
    assert abs(mc.expected_utility - como) <= Z_TOL * mc.expected_utility_se + 1e-12
    assert abs(mc.p_any_admit - max(p[j] for j in s)) <= Z_TOL * mc.p_any_admit_se + 1e-12


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", range(5))
def test_mc_matches_factor_integral(rho, seed):
    p, u, s = _instance(200 + seed)
    exact = factor_portfolio_value(p, u, s, rho=rho)
    mc = mc_portfolio_value(p, u, s, rho=rho, n_draws=N_MC, seed=seed)
    assert abs(mc.expected_utility - exact.expected_utility) <= Z_TOL * mc.expected_utility_se
    assert abs(mc.p_any_admit - exact.p_any_admit) <= Z_TOL * mc.p_any_admit_se


def test_heterogeneous_loadings_mc_matches_factor():
    p, u, s = [0.3, 0.5, 0.7, 0.2], [90, 60, 30, 100], [0, 1, 2, 3]
    lam = [0.9, 0.6, 0.3, 0.0]
    exact = factor_portfolio_value(p, u, s, loadings=lam)
    mc = mc_portfolio_value(p, u, s, loadings=lam, n_draws=N_MC, seed=3)
    assert abs(mc.expected_utility - exact.expected_utility) <= Z_TOL * mc.expected_utility_se
    # Same model given as an explicit correlation matrix.
    mc2 = mc_portfolio_value(p, u, s, corr=one_factor_corr(4, loadings=lam), n_draws=N_MC, seed=3)
    assert abs(mc2.expected_utility - exact.expected_utility) <= Z_TOL * mc2.expected_utility_se


@pytest.mark.parametrize("seed", range(10))
def test_value_and_p_any_decrease_with_rho(seed):
    # max of admitted utilities is submodular in the admission indicators; higher
    # Gaussian correlation is higher in supermodular order, so v(S) cannot rise.
    p, u, s = _instance(300 + seed)
    rows = correlation_sweep(p, u, s, rhos=np.linspace(0, 1, 11))
    vals = [r.correlated for r in rows]
    anys = [r.p_any_correlated for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:], strict=False))
    assert all(b <= a + 1e-9 for a, b in zip(anys, anys[1:], strict=False))
    assert rows[0].overestimate == pytest.approx(0.0, abs=1e-9)
    assert all(r.overestimate >= -1e-9 for r in rows)
    # Bounds: comonotone value at rho = 1, independent value at rho = 0.
    assert vals[-1] == pytest.approx(comonotone_portfolio_value(p, u, s).expected_utility)


def test_marginals_are_preserved():
    # A one-program portfolio is p_j u_j at every rho (copula keeps marginals).
    for rho in (0.0, 0.3, 0.9, 1.0):
        assert factor_portfolio_value([0.37], [50], [0], rho=rho).expected_utility == (
            pytest.approx(0.37 * 50, abs=1e-9)
        )


def test_seed_reproducible_and_seed_sensitive():
    p, u, s = _instance(7, m=6)
    a = mc_portfolio_value(p, u, s, rho=0.4, n_draws=5_000, seed=11)
    b = mc_portfolio_value(p, u, s, rho=0.4, n_draws=5_000, seed=11)
    c = mc_portfolio_value(p, u, s, rho=0.4, n_draws=5_000, seed=12)
    assert a == b
    assert a.expected_utility != c.expected_utility


def test_reported_se_matches_replication_spread():
    p, u, s = [0.3, 0.5, 0.7], [90, 60, 30], [0, 1, 2]
    reps = [mc_portfolio_value(p, u, s, rho=0.5, n_draws=2_000, seed=1000 + i) for i in range(60)]
    sd = float(np.std([r.expected_utility for r in reps], ddof=1))
    se = float(np.mean([r.expected_utility_se for r in reps]))
    # sd of 60 replications has relative sampling error ~ 1/sqrt(2*59) ~ 9%.
    assert 0.7 < sd / se < 1.3


def test_admit_estimate_input_uses_mean():
    est = [
        AdmitEstimate("a", mean=0.3, low=0.2, high=0.4, method="assumed"),
        AdmitEstimate("b", mean=0.8, low=0.7, high=0.9, method="assumed"),
    ]
    r = factor_portfolio_value(est, [10, 5], [0, 1], rho=0.0)
    assert r.expected_utility == pytest.approx(5.8)


def test_brute_force_correlated_rho_zero_matches_c41():
    for seed in range(10):
        p, u, _ = _instance(400 + seed, m=7)
        for h in (1, 3, 7):
            _, best = brute_force(p, u, h=h)
            _, best_c = brute_force_correlated(p, u, rho=0.0, h=h)
            assert best_c == pytest.approx(best, abs=1e-9)


def test_ndtri_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    qs = np.concatenate([np.linspace(1e-9, 1 - 1e-9, 2001), [1e-12, 0.5, 1 - 1e-12]])
    for q in qs:
        assert _ndtri(float(q)) == pytest.approx(float(stats.norm.ppf(q)), abs=1e-8)
    assert _ndtri(0.0) == -math.inf
    assert _ndtri(1.0) == math.inf


def test_degenerate_probabilities():
    # p = 0 never admits, p = 1 always admits, at any correlation.
    for rho in (0.0, 0.5, 1.0):
        r = factor_portfolio_value([0.0, 1.0], [100, 10], [0, 1], rho=rho)
        assert r.expected_utility == pytest.approx(10.0)
        assert r.p_any_admit == pytest.approx(1.0)
    mc = mc_portfolio_value([0.0, 1.0], [100, 10], [0, 1], rho=0.5, n_draws=1000)
    assert mc.expected_utility == pytest.approx(10.0)


def test_empty_portfolio():
    assert mc_portfolio_value([0.5], [1], []).expected_utility == 0.0
    assert factor_portfolio_value([0.5], [1], [], rho=0.5).p_any_admit == 0.0
    assert comonotone_portfolio_value([0.5], [1], []).expected_utility == 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5], [1], [0], rho=1.2)
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5], [1], [0], rho=-0.1)
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5, 0.5], [1, 1], [0], rho=0.2, loadings=[0.1, 0.1])
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5, 0.5], [1, 1], [0, 1], corr=[[1, 0.9], [0.1, 1]])
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5, 0.5], [1, 1], [0, 1], corr=[[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5, 0.5, 0.5], [1, 1, 1], [0, 1, 2],
                           corr=[[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])  # fmt: skip
    with pytest.raises(ValueError):
        factor_portfolio_value([0.5], [1], [1], rho=0.0)
    with pytest.raises(ValueError):
        factor_portfolio_value([0.5, 0.5], [1, 1], [0, 0], rho=0.0)
    with pytest.raises(ValueError):
        factor_portfolio_value([0.5], [1], [0], loadings=[1.5])
    with pytest.raises(ValueError):
        mc_portfolio_value([0.5], [1], [0], n_draws=1)


# --- L4 probe ---------------------------------------------------------------------


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C4-1: C4.1 chooses the portfolio assuming independence; C4.2 only evaluates it",
)
def test_probe_independence_optimum_stays_optimal_under_correlation():
    # Two equal reaches and one safety, apply to two (docs/components/C4 L4 row).
    # Independence prefers both reaches; with rho = 0.6 the reaches fail together
    # and reach + safety is better (60.2 vs 54.3).
    p, u, rho = [0.4, 0.4, 0.9], [100, 100, 40], 0.6
    chosen, _ = greedy_same_cost(p, u, h=2)
    _, best = brute_force_correlated(p, u, rho=rho, h=2)
    got = factor_portfolio_value(p, u, chosen[:2], rho=rho).expected_utility
    assert got >= best - 1e-9
