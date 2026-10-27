"""C3.2 v0 hierarchical admission model (L2, L4)."""

import numpy as np
import pytest


def _world(rng, n_prog=12, n_per=60, q1=1.0, q0=1.0, beta=(1.0, -0.5)):
    """Programmes in 3 families; applicants with 2 standardised features; reporting
    with probability q1 if admitted, q0 if rejected (outcome-dependent)."""
    fam = [f"f{i % 3}" for i in range(n_prog)]
    mu_f = {"f0": -1.5, "f1": -0.5, "f2": 0.5}
    a = {f"p{i}": mu_f[fam[i]] + rng.normal(0, 0.5) for i in range(n_prog)}
    recs, rates = [], {}
    for i in range(n_prog):
        pid = f"p{i}"
        X = rng.normal(0, 1, (4000, 2))
        p = 1 / (1 + np.exp(-(a[pid] + X @ np.array(beta))))
        y = rng.random(4000) < p
        rates[pid] = float(y.mean())
        keep = rng.random(4000) < np.where(y, q1, q0)
        idx = np.flatnonzero(keep)[:n_per]
        recs += [(pid, fam[i], X[j], int(y[j])) for j in idx]
    return recs, rates, a


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C3-1: self-reported outcomes (admits report more) inflate admission estimates",
)
def test_probe_self_report_bias_is_corrected_by_official_rates():
    """Admitted applicants report 4x as often as rejected ones. Fitted naively the
    average applicant's P(admit) is far too high; anchored to official admission
    rates the estimate must be close to the true rate."""
    import aisc.feasibility as fz

    assert hasattr(fz, "fit_admission_model")
    rng = np.random.default_rng(0)
    recs, rates, _ = _world(rng, q1=0.8, q0=0.2)
    data = [fz.AdmissionRecord(pid, fam, tuple(x), y, "self_report") for pid, fam, x, y in recs]
    fams = {pid: fam for pid, fam, _, _ in recs}
    naive = fz.fit_admission_model(data, families=fams)
    anchored = fz.fit_admission_model(data, families=fams, official_rates=rates)
    err_naive, err_anchored = [], []
    for pid, r in rates.items():
        err_naive.append(naive.predict(pid, (0.0, 0.0)).mean - r)
        err_anchored.append(anchored.predict(pid, (0.0, 0.0)).mean - r)
    assert np.mean(err_naive) > 0.15  # the bias is real
    assert abs(np.mean(err_anchored)) < 0.06
