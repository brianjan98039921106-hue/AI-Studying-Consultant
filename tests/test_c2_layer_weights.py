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
