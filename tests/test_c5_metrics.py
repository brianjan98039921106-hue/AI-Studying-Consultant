"""C5.2 metric verification (L2): every value below is computed by hand."""

import math

import numpy as np
import pytest

from aisc.simulation import metrics as m


def test_rmse():
    # errors 0.1, -0.2, 0.2 -> mean sq = (0.01+0.04+0.04)/3 = 0.03
    assert m.rmse([0.1, 0.3, 0.7], [0.0, 0.5, 0.5]) == pytest.approx(math.sqrt(0.03))


def test_kendall_tau_b_hand_cases():
    assert m.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert m.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # x=[1,2,3], y=[1,3,2]: pairs (12)+ (13)+ (23)- -> (2-1)/3
    assert m.kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    # ties: x=[1,1,2], y=[1,2,3]: pair(12) tie in x only; (13)+ (23)+
    # nc=2, nd=0, tx=1, ty=0 -> 2/sqrt(3*2)
    assert m.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(6))
    assert math.isnan(m.kendall_tau_b([1, 1, 1], [1, 2, 3]))


def test_kendall_tau_b_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    assert m.kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic)


def test_list_metrics():
    rec = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    familiar = {"a", "b"}
    assert m.recall_at_k(rec, relevant, 2) == pytest.approx(1 / 3)
    assert m.recall_at_k(rec, relevant, 4) == pytest.approx(2 / 3)
    q = m.list_quality(rec, relevant, familiar)
    assert q["relevant_rate"] == pytest.approx(0.5)  # a, c
    assert q["novel_relevant_rate"] == pytest.approx(0.25)  # only c
    assert q["unfamiliar_rate"] == pytest.approx(0.5)  # c, d
    assert m.best_fit_retained(rec, "c") == 1.0
    assert m.best_fit_retained(rec, "z") == 0.0
    assert math.isnan(m.relevant_rate([], relevant))


def test_catalog_coverage():
    assert m.catalog_coverage([["a", "b"], ["b", "c"]], 10) == pytest.approx(0.3)


def test_popularity_kl():
    catalog = [0.1, 0.3, 0.5, 0.7, 0.9]  # one per bin -> uniform
    assert m.popularity_kl(catalog, catalog) == pytest.approx(0.0, abs=1e-9)
    # all recommendations in the top bin: KL ≈ log(1/0.2) = log 5
    assert m.popularity_kl([0.95, 0.9], catalog) == pytest.approx(math.log(5), rel=1e-3)


def test_brier_and_coverage():
    # (0.8-1)^2=0.04, (0.3-0)^2=0.09 -> 0.065
    assert m.brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)
    assert m.interval_coverage([0, 0.2], [0.5, 0.4], [0.3, 0.5]) == pytest.approx(0.5)


def test_mean_se():
    mean, se = m.mean_se([1.0, 2.0, 3.0, float("nan")])
    assert mean == pytest.approx(2.0)
    assert se == pytest.approx(1.0 / math.sqrt(3))


@pytest.mark.parametrize(
    "fn,args",
    [
        (m.rmse, ([1, 2], [1])),
        (m.kendall_tau_b, ([1, 2], [1])),
        (m.brier_score, ([0.1, 0.2], [1])),
        (m.interval_coverage, ([0], [1, 2], [0])),
    ],
)
def test_shape_mismatch_rejected(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_empty_and_degenerate_inputs():
    assert math.isnan(m.recall_at_k(["a"], set(), 1))
    assert math.isnan(m.novel_relevant_rate([], {"a"}, set()))
    assert math.isnan(m.list_quality([], {"a"}, set())["unfamiliar_rate"])
    assert math.isnan(m.best_fit_retained(["a"], None))
    assert math.isnan(m.popularity_kl([], [0.5]))
    assert all(math.isnan(v) for v in m.mean_se([float("nan")]))
    assert math.isnan(m.mean_se([2.0])[1])
    with pytest.raises(ValueError):
        m.catalog_coverage([["a"]], 0)
