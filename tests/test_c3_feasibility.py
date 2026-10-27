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


def test_self_report_bias_is_corrected_by_official_rates():
    """B-C3-1 (was a probe). Admitted applicants report 4x as often as rejected ones.
    Fitted naively the average applicant's P(admit) is far too high; anchored to
    official admission rates the estimate must be close to the true rate."""
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


def _records(recs, source):
    import aisc.feasibility as fz

    return [fz.AdmissionRecord(pid, fam, tuple(x), y, source) for pid, fam, x, y in recs]


def test_recovers_slopes_and_intervals_without_selection():
    import aisc.feasibility as fz

    rng = np.random.default_rng(1)
    recs, _, a = _world(rng, n_per=150)
    m = fz.fit_admission_model(_records(recs, "complete"))
    assert m.method == "hierarchical_v0.2" and not m.shift_estimated
    assert m.theta[:2] == pytest.approx([1.0, -0.5], abs=0.15)
    covered = 0
    for pid, ai in a.items():
        est = m.predict(pid, (0.0, 0.0))
        assert est.low <= est.mean <= est.high
        covered += est.low <= 1 / (1 + np.exp(-ai)) <= est.high
    assert covered >= 9  # 90% intervals over 12 programmes


def test_complete_records_identify_the_self_report_shift():
    import aisc.feasibility as fz

    rng = np.random.default_rng(2)
    biased, _, _ = _world(rng, n_per=150, q1=0.8, q0=0.2)
    rng = np.random.default_rng(3)
    complete, _, _ = _world(rng, n_per=60)
    m = fz.fit_admission_model(_records(biased, "self_report") + _records(complete, "complete"))
    assert m.method == "hierarchical_v0.2_corrected"
    assert m.shift == pytest.approx(np.log(4), abs=0.35)


def test_new_programme_uses_family_and_is_less_certain():
    import aisc.feasibility as fz

    rng = np.random.default_rng(4)
    recs, _, _ = _world(rng)
    m = fz.fit_admission_model(_records(recs, "complete"))
    known = m.predict("p0", (0.0, 0.0))
    new = m.predict("p-new", (0.0, 0.0), family="f0")
    assert (new.high - new.low) > (known.high - known.low)
    with pytest.raises(ValueError):
        m.predict("p-new", (0.0, 0.0))
    with pytest.raises(ValueError):
        m.predict("p0", (0.0,))


def test_record_and_fit_validation():
    import aisc.feasibility as fz

    with pytest.raises(ValueError):
        fz.AdmissionRecord("p", "f", (0.0,), 2)
    with pytest.raises(ValueError):
        fz.AdmissionRecord("p", "f", (0.0,), 1, "gradcafe")
    with pytest.raises(ValueError):
        fz.AdmissionRecord("p", "f", (float("nan"),), 1)
    with pytest.raises(ValueError):
        fz.fit_admission_model([])
    r = [fz.AdmissionRecord("p", "f", (0.0,), 1), fz.AdmissionRecord("p", "g", (0.0,), 0)]
    with pytest.raises(ValueError):
        fz.fit_admission_model(r)


def test_intervals_cover_when_stronger_applicants_report_more():
    """B-C3-8 (was a probe; fixed by the pool-integrated anchor, F139).
    Self-reports from admitted applicants 4x as likely, and stronger applicants
    (feature 1) also report more; the official pool is shifted on feature 1 (A-C3-3
    and A-C3-5 both violated, as in the V-C3-1 stress generator). 90% intervals for
    new applicants should still cover the true probability at least 80% of the time
    (v0 in V-C3-1 stress: 61-66%, F130)."""
    import aisc.feasibility as fz

    rng = np.random.default_rng(5)
    beta = np.array([0.8, -0.4])
    recs, rates, tests, fams = [], {}, {}, {}
    for i in range(24):
        pid, fam = f"p{i}", f"f{i % 3}"
        fams[pid] = fam
        a = (-2.0, -1.0, 0.0)[i % 3] + rng.normal(0, 0.6)
        pool = rng.normal(0, 1, (4000, 2)) + np.array([0.3, 0.0])
        y = rng.random(4000) < 1 / (1 + np.exp(-(a + pool @ beta)))
        rates[pid] = float(y.mean())
        rep = np.minimum(1.0, np.where(y, 0.8, 0.2) * 2 / (1 + np.exp(-0.7 * pool[:, 0])))
        idx = np.flatnonzero(rng.random(4000) < rep)[:60]
        recs += [fz.AdmissionRecord(pid, fam, tuple(pool[j]), int(y[j]), "self_report")
                 for j in idx]  # fmt: skip
        Xt = rng.normal(0, 1, (100, 2)) + np.array([0.3, 0.0])
        tests[pid] = (Xt, 1 / (1 + np.exp(-(a + Xt @ beta))))
    m = fz.fit_admission_model(recs, families=fams, official_rates=rates)
    hits = [m.predict(pid, tuple(x)).low <= p <= m.predict(pid, tuple(x)).high
            for pid, (Xt, pt) in tests.items() for x, p in zip(Xt, pt, strict=True)]  # fmt: skip
    assert np.mean(hits) >= 0.8


def test_intervals_cover_nominally_when_well_specified():
    """B-C3-9 (was a probe; fixed by tau model averaging, F136). Complete records
    from the model's own family of worlds: 90% intervals for the known programmes
    should cover the true P(admit) at an average applicant at least 0.90 - 3 SE
    (about 0.87 over 960 checks); plug-in tau gave 0.85."""
    import aisc.feasibility as fz

    hits = []
    for r in range(80):
        rng = np.random.default_rng([7, r])
        recs, _, a = _world(rng, n_per=40)
        m = fz.fit_admission_model(_records(recs, "complete"))
        for pid, ai in a.items():
            e = m.predict(pid, (0.0, 0.0))
            hits.append(e.low <= 1 / (1 + np.exp(-ai)) <= e.high)
    assert np.mean(hits) >= 0.90 - 3 * np.sqrt(0.09 / len(hits))


def test_v01_options_validate_and_behave():
    """v0.1 candidate options: the pool anchor matches the official rate as an average
    over the programme's (reweighted) records rather than at x = 0; tau averaging
    returns mixture weights that sum to 1."""
    import aisc.feasibility as fz

    rng = np.random.default_rng(6)
    recs, rates, _ = _world(rng, q1=0.8, q0=0.2)
    data = _records(recs, "self_report")
    with pytest.raises(ValueError):
        fz.fit_admission_model(data, anchor="median")
    with pytest.raises(ValueError):
        fz.fit_admission_model(data, tau=0.5, tau_average=True)
    m = fz.fit_admission_model(data, official_rates=rates, anchor="pool", tau_average=True)
    assert m.method == "hierarchical_v0.2_corrected"
    assert sum(w for w, *_ in m.mixture) == pytest.approx(1.0)
    gaps = {"pool": [], "mean": []}
    mean_model = fz.fit_admission_model(data, official_rates=rates, anchor="mean")
    for name, model in (("pool", m), ("mean", mean_model)):
        for pid, r in rates.items():
            rows = [d for d in data if d.program_id == pid]
            w = np.array([np.exp(-model.shift * d.admitted) for d in rows])
            p = np.array([model.predict(pid, d.features).mean for d in rows])
            gaps[name].append(abs(float((w * p).sum() / w.sum()) - r))
    assert np.mean(gaps["pool"]) < np.mean(gaps["mean"])


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C3-6: one model for all admission processes; programme type cannot enter",
)
def test_probe_programme_type_shifts_new_programme_predictions():
    """Interview-based programmes admit fewer applicants (logit -1.2) in every family.
    Told the types, the model must predict an unseen interview programme lower than
    an unseen standard one of the same family, close to the truth."""
    import inspect

    import aisc.feasibility as fz

    assert "types" in inspect.signature(fz.fit_admission_model).parameters
    rng = np.random.default_rng(8)
    recs, types, fams = [], {}, {}
    for i in range(24):
        pid, fam = f"p{i}", f"f{i % 3}"
        t = "interview" if i % 2 else "standard"
        types[pid], fams[pid] = t, fam
        a = (-1.0, 0.0, 0.8)[i % 3] + (-1.2 if t == "interview" else 0.0) + rng.normal(0, 0.3)
        X = rng.normal(0, 1, (80, 2))
        y = rng.random(80) < 1 / (1 + np.exp(-(a + X @ np.array([0.8, -0.4]))))
        recs += [fz.AdmissionRecord(pid, fam, tuple(X[j]), int(y[j]), "complete")
                 for j in range(80)]  # fmt: skip
    m = fz.fit_admission_model(recs, families=fams, types=types)
    hi = m.predict("new-std", (0.0, 0.0), family="f1", ptype="standard").mean
    lo = m.predict("new-int", (0.0, 0.0), family="f1", ptype="interview").mean
    assert hi - lo > 0.15
    assert lo == pytest.approx(1 / (1 + np.exp(1.2)), abs=0.1)
