"""C0.4 v2 hierarchical-shrinkage fusion (L2)."""

import math

import numpy as np
import pytest

from aisc.catalog import estimate_tau, fuse_profile

PRIOR = {"a": 0.4, "b": 0.7, "c": 0.2}


def test_posterior_is_precision_weighted_and_keeps_unobserved_prior():
    f = fuse_profile(PRIOR, {"a": 0.8, "b": 0.7}, tau=0.1, sigma=0.2)
    w = (1 / 0.01) / (1 / 0.01 + 1 / 0.04)
    assert f.mean["a"] == pytest.approx(w * 0.4 + (1 - w) * 0.8)
    assert f.sd["a"] == pytest.approx(math.sqrt(1 / (1 / 0.01 + 1 / 0.04)))
    assert f.mean["c"] == 0.2 and f.sd["c"] == 0.1
    assert f.flags == ()


def test_limits_and_flags():
    exact = fuse_profile(PRIOR, {"a": 0.9}, tau=0.1, sigma=0.0)
    assert exact.mean["a"] == 0.9 and exact.sd["a"] == 0.0
    fixed = fuse_profile(PRIOR, {"a": 0.9}, tau=0.0, sigma=0.1)
    assert fixed.mean["a"] == 0.4
    flagged = fuse_profile(PRIOR, {"a": 0.9, "b": 0.75}, tau=0.1, sigma=0.1)
    assert flagged.flags == ("a",)  # |0.5| / sqrt(0.02) = 3.5 > 2; b: 0.35 < 2
    per_dim = fuse_profile(PRIOR, {"a": 0.9}, tau={"a": 0.5, "b": 0.1, "c": 0.1}, sigma={"a": 0.1})
    assert per_dim.mean["a"] > 0.85


def test_validation():
    with pytest.raises(ValueError):
        fuse_profile(PRIOR, {"z": 0.5}, tau=0.1, sigma=0.1)
    with pytest.raises(ValueError):
        fuse_profile(PRIOR, {"a": 0.5}, tau=-0.1, sigma=0.1)
    with pytest.raises(ValueError):
        fuse_profile(PRIOR, {"a": 0.5}, tau=0.0, sigma=0.0)
    with pytest.raises(ValueError):
        estimate_tau([PRIOR], [{"a": 0.5}, {"a": 0.4}], 0.1)


def test_estimate_tau_recovers_simulated_spread_and_fusion_beats_either_source():
    """Simulate programmes around their CIP prior (tau = 0.12) with evidence noise
    sigma = 0.10: the moment estimate is close, and the fused mean has lower error
    than both the prior alone (B-C0-1: all programmes of a CIP identical) and the raw
    evidence."""
    rng = np.random.default_rng(0)
    tau, sigma, n = 0.12, 0.10, 4000
    priors, evid, truth = [], [], []
    for _ in range(n):
        m = {k: float(rng.uniform(0.2, 0.8)) for k in PRIOR}
        x = {k: m[k] + float(rng.normal(0, tau)) for k in PRIOR}
        priors.append(m)
        truth.append(x)
        evid.append({k: x[k] + float(rng.normal(0, sigma)) for k in PRIOR})
    est = estimate_tau(priors, evid, sigma)
    assert all(abs(v - tau) < 0.01 for v in est.values())
    err = {"prior": 0.0, "evidence": 0.0, "fused": 0.0}
    for m, e, x in zip(priors, evid, truth, strict=True):
        f = fuse_profile(m, e, est, sigma)
        for k in PRIOR:
            err["prior"] += (m[k] - x[k]) ** 2
            err["evidence"] += (e[k] - x[k]) ** 2
            err["fused"] += (f.mean[k] - x[k]) ** 2
    assert err["fused"] < err["evidence"] < err["prior"]


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C0-17: a dimension without a tau estimate crashes fuse_profile with KeyError",
)
def test_probe_fusion_reports_missing_tau_clearly():
    priors = [PRIOR, PRIOR, PRIOR]
    evid = [{"a": 0.5, "b": 0.6}, {"a": 0.45}, {"a": 0.3}]
    tau = estimate_tau(priors, evid, 0.05)  # only "a" has >= 2 programmes
    try:
        fuse_profile(PRIOR, evid[0], tau, 0.05)
        ok = False
    except ValueError as e:
        ok = "b" in str(e)
    except KeyError:
        ok = False
    assert ok
