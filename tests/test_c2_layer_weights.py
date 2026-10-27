"""C2.1 calibration: layer-weight estimator (W-76), L2."""

import numpy as np
import pytest

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import LAYERS, NEED_KEYS, keys_in_layer
from aisc.discovery.layer_weights import (
    estimate_layer_weights,
    fit_layer_weights,
    layer_differences,
    pair_features,
)


def _simulate(beta, n_resp=300, k=20, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 0.05, (n_resp * k, len(beta)))
    p = 1 / (1 + np.exp(-(F @ beta)))
    y = (rng.random(len(p)) < p).astype(float)
    resp = [f"r{i // k}" for i in range(len(y))]
    return F, y, resp


def test_layer_differences_v1():
    w = np.full(len(NEED_KEYS), 0.5)
    x = w.copy()
    idx = NEED_KEYS.index(keys_in_layer("intensity")[0])
    x[idx] = 0.9
    d = layer_differences(w, x, "v1")[0]
    n_int = len(keys_in_layer("intensity"))
    expected = [0.16 / n_int if lay == "intensity" else 0.0 for lay in LAYERS]
    np.testing.assert_allclose(d, expected)


def test_pair_features_sign_and_dims():
    s = StudentProfile("s", {k: 0.8 for k in NEED_KEYS})
    near = Program("a", "U", "MS", "27.0501", {k: 0.8 for k in NEED_KEYS})
    far = Program("b", "U", "MS", "27.0501", {k: 0.1 for k in NEED_KEYS})
    assert (pair_features(s, near, far) > 0).all()  # positive favours A
    v2 = StudentProfile("t", {"riasec_i": 0.5}, dims="v2")
    with pytest.raises(ValueError):
        pair_features(v2, near, far)


def test_fit_recovers_known_weights_and_respects_zero():
    beta = np.array([20.0, 10.0, 60.0, 0.0])
    F, y, _ = _simulate(beta, n_resp=600)
    est = fit_layer_weights(F, y)
    assert (est >= 0).all()
    np.testing.assert_allclose(est / est.sum(), beta / beta.sum(), atol=0.05)


def test_estimate_interval_covers_truth():
    beta = np.array([25.0, 25.0, 25.0, 25.0])
    F, y, resp = _simulate(beta, n_resp=300, seed=1)
    e = estimate_layer_weights(F, y, resp, n_boot=100, rng=np.random.default_rng(2))
    assert e.n_respondents == 300 and e.n_pairs == 6000
    assert sum(e.shares.values()) == pytest.approx(1.0)
    for lay in LAYERS:
        lo, hi = e.ci[lay]
        assert lo <= 0.25 <= hi


def test_validation():
    with pytest.raises(ValueError):
        fit_layer_weights(np.zeros((3, 4)), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        estimate_layer_weights(np.zeros((2, 4)), np.array([0, 1]), ["a"])


def test_simex_reduces_measurement_error_bias():
    from aisc.dimensions import NEED_KEYS
    from aisc.discovery.layer_weights import estimate_layer_weights_simex

    rng = np.random.default_rng(5)
    d = len(NEED_KEYS)
    n_resp, k, sigma = 300, 20, 0.2
    true_w = rng.random((n_resp, d))
    X = rng.random((60, d))
    pairs = np.array([(r, a, (a + rng.integers(1, 60)) % 60)
                      for r in range(n_resp) for a in rng.integers(0, 60, k)])  # fmt: skip
    beta = np.array([30.0, 30.0, 30.0, 30.0])
    from aisc.discovery.layer_weights import _pair_rows

    G = _pair_rows(true_w, X, pairs, "v1")
    y = (rng.random(len(G)) < 1 / (1 + np.exp(-(G @ beta)))).astype(float)
    stated = np.clip(true_w + rng.normal(0, sigma, true_w.shape), 0, 1)
    est = estimate_layer_weights_simex(stated, X, pairs, y, sigma, "v1", n_sim=5, n_boot=20,
                                       rng=np.random.default_rng(6))  # fmt: skip
    err_naive = max(abs(est.naive.shares[lay] - 0.25) for lay in LAYERS)
    err_simex = max(abs(est.corrected.shares[lay] - 0.25) for lay in LAYERS)
    assert err_simex < err_naive
    assert sum(est.corrected.shares.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        estimate_layer_weights_simex(stated, X, pairs[:, :2], y, sigma, "v1")
