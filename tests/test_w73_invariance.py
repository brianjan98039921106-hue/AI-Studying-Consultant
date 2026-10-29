"""W-73 pre-registered invariance / DIF analysis (L2): the ML CFA engine, the ordinal
DIF test and a dry run on synthetic files. All data are synthetic."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_w73_invariance.py"
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("analyze_w73_invariance", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["analyze_w73_invariance"] = mod
spec.loader.exec_module(mod)

FAC = [0, 0, 1, 1, 2, 2]
LAM = np.array([1, 0.8, 1, 1.2, 1, 0.7])


def _groups(n, rng, tau_shift=0.0):
    Phi = 0.04 * (0.4 + 0.6 * np.eye(3))
    L = np.zeros((6, 3))
    L[np.arange(6), FAC] = LAM
    out = []
    for g in range(2):
        tau = np.full(6, 0.5)
        if g == 1:
            tau[1] += tau_shift
        f = rng.multivariate_normal(np.zeros(3), Phi, n)
        out.append(tau + f @ L.T + rng.normal(0, 0.1, (n, 6)))
    return out


def _cfa(X):
    return mod.MultiGroupCFA([x.mean(0) for x in X], [np.cov(x.T, ddof=0) for x in X],
                             [len(x) for x in X], FAC)  # fmt: skip


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    cfa = _cfa(_groups(200, rng))
    for shared in (False, True):
        idx, k = cfa.layout([shared] * 3, [shared] * 6, [shared] * 3)
        x = cfa.start(idx, k) + rng.normal(0, 0.05, k)
        _, g = cfa.objective(x, idx)
        num = np.array([(cfa.objective(x + e, idx)[0] - cfa.objective(x - e, idx)[0]) / 2e-6
                        for e in np.eye(k) * 1e-6])  # fmt: skip
        assert np.abs(num - g).max() < 1e-6


def test_degrees_of_freedom_and_recovery():
    rng = np.random.default_rng(1)
    cfa = _cfa(_groups(2000, rng))
    res = mod.invariance(cfa)
    # 2 groups x (21 + 6) moments; configural 2 x (3 + 6 + 6 + 6) = 42 parameters
    assert [res[m].df for m in ("configural", "metric", "scalar")] == [12, 15, 18]
    idx = cfa.layout([True] * 3, [False] * 6, [False] * 3)[0]
    lam = [res["metric"].x[idx[("lam", i, None)][0]] for i in cfa.free_items]
    assert np.allclose(lam, LAM[cfa.free_items], atol=0.08)
    assert all(res[m].converged for m in ("configural", "metric", "scalar"))
    assert res["configural"].cfi - res["metric"].cfi <= mod.DELTA_CFI


def test_intercept_violation_shows_at_the_scalar_step_only():
    rng = np.random.default_rng(2)
    res = mod.invariance(_cfa(_groups(1000, rng, tau_shift=0.15)))
    assert res["configural"].cfi - res["metric"].cfi <= mod.DELTA_CFI
    assert res["metric"].cfi - res["scalar"].cfi > mod.DELTA_CFI
    per = res["per_dim"]
    assert max(per, key=lambda f: per[f][1]) == 0  # the shifted item's dimension


def test_ordinal_loglik_and_bh():
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    ll = mod.ordinal_loglik(y, np.empty((10, 0)))
    n_k = np.bincount(y)
    assert math.isclose(ll, float(np.sum(n_k * np.log(n_k / 10))), rel_tol=1e-6)
    assert np.allclose(
        mod.bh_adjust([0.01, 0.04, 0.03, 0.5]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5]
    )


def test_dif_item_detects_strong_uniform_dif_only_when_present():
    rng = np.random.default_rng(3)
    n = 400
    theta = rng.normal(size=n)
    g = np.repeat(["a", "b"], n // 2)
    match = theta + rng.normal(0, 0.3, n)
    cuts = [-1.5, -0.5, 0.5, 1.5]
    clean = np.digitize(theta + rng.logistic(0, 0.5, n), cuts) + 1
    biased = np.digitize(theta + 1.2 * (g == "b") + rng.logistic(0, 0.5, n), cuts) + 1
    r_clean, r_bias = mod.dif_item(clean, match, g), mod.dif_item(biased, match, g)
    assert r_clean["p"] > 0.01 and r_clean["dr2"] < mod.R2_MODERATE
    assert r_bias["p"] < 1e-6 and r_bias["dr2"] >= mod.R2_MODERATE
    assert r_bias["p_uniform"] < 1e-6


def test_dry_run_on_synthetic_files(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}
    folder = tmp_path / "study"
    make = [sys.executable, str(SCRIPT), "--make-fake", str(folder), "--n", "80"]
    assert subprocess.run(make, capture_output=True, text=True, env=env).returncode == 0
    run = subprocess.run([sys.executable, str(SCRIPT), "--data", str(folder)],
                         capture_output=True, text=True, env=env)  # fmt: skip
    assert run.returncode == 0, run.stderr[-800:]
    text = (folder / "invariance.md").read_text(encoding="utf-8")
    assert "含合成資料" in text and "不具約束力" in text
    assert "## A." in text and "## B." in text and "兩組" in text  # 20 per stratum -> near/far
