"""W-73 pre-registered invariance / DIF analysis (L2): statistics against reference
formulas, decision rules, planted-DIF and no-DIF dry runs on synthetic data, and probes
for the blind spots the plan registers (B-C1-40, B-C1-41, B-C1-42)."""

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_w73_invariance.py"
spec = importlib.util.spec_from_file_location("analyze_w73_invariance", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _probe(reason):
    """Probe marker: strict xfail that only accepts an AssertionError."""

    def mark(fn):
        return pytest.mark.probe(pytest.mark.xfail(strict=True, raises=AssertionError,
                                                   reason=reason)(fn))  # fmt: skip

    return mark


# --- statistics ------------------------------------------------------------------------


def test_mantel_equals_classic_mh_chi2_for_binary_items():
    rng = np.random.default_rng(0)
    n = 400
    match = rng.integers(0, 5, n)
    focal = rng.random(n) < 0.4
    y = (rng.random(n) < 0.3 + 0.1 * match - 0.1 * focal).astype(int)
    num = var = 0.0
    for k in range(5):
        s = match == k
        a = np.sum(y[s] & focal[s])  # focal and correct
        n1, m1, nk = focal[s].sum(), y[s].sum(), s.sum()
        num += a - n1 * m1 / nk
        var += n1 * (nk - n1) * m1 * (nk - m1) / (nk**2 * (nk - 1))
    r = mod.mantel(y, focal, match)
    assert abs(r["chi2"] - num**2 / var) < 1e-9


def test_two_item_sum_matching_mirrors_the_items():
    """F160: matching on the sum of two items gives both items the same Mantel
    statistic and exactly opposite SMDs, so item-level DIF is not identified."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        y1, y2 = rng.integers(1, 6, 300), rng.integers(1, 6, 300)
        y1[:60] = np.minimum(y1[:60] + 1, 5)  # some DIF in item 1 for the first rows
        f = np.arange(300) < 100
        a, b = mod.mantel(y1, f, y1 + y2), mod.mantel(y2, f, y1 + y2)
        assert abs(a["chi2"] - b["chi2"]) < 1e-9 and abs(a["smd"] + b["smd"]) < 1e-12


def test_proportional_odds_reduces_to_logistic_regression():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.3 + 1.2 * x)))).astype(int)
    ll, ok, beta = mod.po_fit(y, x[:, None])
    X = np.column_stack([np.ones(500), x])
    w = np.zeros(2)
    for _ in range(30):  # IRLS reference
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ w))
    assert ok and abs(beta[0] - w[1]) < 1e-4
    assert abs(ll - np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))) < 1e-6


def test_proportional_odds_null_model_matches_closed_form():
    y = np.repeat([0, 1, 2, 3], [10, 30, 25, 5])
    ll, ok, _ = mod.po_fit(y, np.zeros((len(y), 0)))
    cnt = np.bincount(y)
    assert ok and abs(ll - np.sum(cnt * np.log(cnt / len(y)))) < 1e-6


def test_lr_dif_detects_group_shift_and_reports_effect_size():
    rng = np.random.default_rng(3)
    n = 1200
    st = np.repeat([0, 1, 2, 3], n // 4)
    th = rng.normal(size=n)
    y1 = mod.grm_sample(th, 1.8, np.array([-1.5, -0.5, 0.5, 1.5]), rng)
    y2 = mod.answers_with_dif(th, st, 1.8, np.array([-1.2, -0.3, 0.6, 1.4]),
                              {"group": 2, "shift": 1.0}, rng)  # fmt: skip
    r = mod.lr_dif(y2, y1 + y2, st)
    assert r["ok"] and r["df"] == 6 and r["p"] < 1e-6 and r["dr2"] > 0


def test_benjamini_hochberg_reference_values():
    adj = mod.bh([0.01, 0.04, 0.03, 0.005, math.nan])
    assert np.allclose(adj[:4], [0.02, 0.04, 0.04, 0.02]) and math.isnan(adj[4])


def test_classification_boundaries():
    assert mod.ets_class(0.01, 0.17) == "AA" and mod.ets_class(0.01, 0.20) == "BB"
    assert mod.ets_class(0.01, -0.26) == "CC" and mod.ets_class(0.06, 0.9) == "AA"
    assert mod.lr_class(0.01, 0.034) == "A" and mod.lr_class(0.01, 0.035) == "B"
    assert mod.lr_class(0.01, 0.070) == "C" and mod.lr_class(0.2, 0.5) == "A"
    assert mod.lr_class(0.01, 0.20, mod.ZT) == "B"


# --- tiers, CFA export and decision rules --------------------------------------------


def test_matching_criteria_follow_the_preregistered_tiers():
    dims, match = mod.dim_items(), mod.matching_items()
    t1, t2 = mod.tiers()
    assert len(t1) == 20 and len(t2) == 53
    for it in t1:
        assert it in match[it] and len(match[it]) == 2
    riasec_items = {i for t in "riasec" for i in dims[f"riasec_{t}"]}
    for it in t2:
        assert match[it] and set(match[it]) <= riasec_items and it not in match[it]
    assert match["bi_law#1"] == dims["riasec_e"]  # Law links to Enterprising in O*NET


def test_collapse_categories_merges_toward_midpoint_in_every_stratum():
    st = np.repeat([0, 1], 6)
    x = np.array([1, 2, 3, 4, 5, 5, 2, 2, 3, 4, 5, 4])  # category 1 empty in stratum 1
    c = mod.collapse_categories(x, st)
    assert list(c) == [1, 1, 2, 3, 4, 4, 1, 1, 2, 3, 4, 3]
    assert all(set(c[st == g]) == set(c) for g in (0, 1))


def test_cfa_decision_rules():
    def s(cfi, rmsea, ok=True):
        return {"ok": ok, "cfi": cfi, "rmsea": rmsea}

    good = {"configural": s(0.97, 0.05), "thresholds": s(0.965, 0.055), "scalar": s(0.958, 0.06)}
    v = mod.cfa_verdict(good)
    assert v["configural"] and v["thresholds"] and v["scalar"] and not v["scalar_small"]
    bad = good | {"scalar": s(0.94, 0.08)}
    assert mod.cfa_verdict(bad)["scalar"] is False
    assert mod.cfa_verdict({"configural": s(0.93, 0.05)})["configural"] is False
    assert mod.cfa_verdict({"configural": s(0.97, 0.05, ok=False)})["configural"] is False
    part = bad | {"partial": s(0.962, 0.058)}
    assert mod.cfa_verdict(part)["partial"] is True


def test_dimension_verdicts():
    ok = {"configural": True, "scalar": True}
    assert mod.dimension_verdict(2, 0, False, ok, False) == "可比較"
    assert "只依 DIF" in mod.dimension_verdict(2, 0, False, None, False)
    assert "檢定力不足" in mod.dimension_verdict(2, 0, False, None, True)
    assert "維度層 DIF" in mod.dimension_verdict(2, 1, False, None, False)
    assert "形態" in mod.dimension_verdict(2, 0, False, {"configural": False}, False)
    assert "純量未通過" in mod.dimension_verdict(2, 0, False, {"configural": True,
                                                              "scalar": False}, False)  # fmt: skip
    # partial invariance needs >= 3 items with >= 2 clean ones (unreachable today)
    assert "部分可比較" in mod.dimension_verdict(3, 1, True, None, False)
    failed_partial = {"configural": True, "partial": False}
    assert "部分純量未通過" in mod.dimension_verdict(3, 1, True, failed_partial, False)


def test_grm_conversion_from_theta_parameterisation():
    rows = [{"lhs": "F", "op": "=~", "rhs": "x1", "group": "1", "est": "2.0"},
            {"lhs": "x1", "op": "|", "rhs": "t2", "group": "1", "est": "1.0"},
            {"lhs": "x1", "op": "|", "rhs": "t1", "group": "1", "est": "-1.0"}]  # fmt: skip
    assert mod.grm_from_lavaan(rows) == {("x1", "1"): (2.0, [-0.5, 0.5])}


def test_r_script_carries_the_preregistered_choices():
    dims = mod.dim_items()
    partial = {"riasec": ["riasec_i_1", "riasec_i_2"], "core": []}
    r = mod.r_script(dims, partial)
    for s in ('estimator = "WLSMV"', 'parameterization = "theta"', "Wu.Estabrook.2016",
              "F_riasec_i =~ riasec_i_1", '"riasec_i_1 | t4"', "w73_cfa_fit.csv"):  # fmt: skip
        assert s in r, s
    assert r.count("partial = c(") == 1  # only the model with a C dimension frees items


# --- synthetic dry runs --------------------------------------------------------------


def _tier1(folder, n_per, seed, planted, impact):
    mod.make_fake(folder, n_per, seed, planted, impact=impact, missing=0.0)
    bank = [it["item_id"] for it in mod.item_bank()]
    Y, st, syn = mod.read_s1(folder / "s1.csv", bank)
    cols = {i: j for j, i in enumerate(bank)}
    t1, t2 = mod.tiers()
    return Y, st, syn, cols, t1, t2


def test_planted_dif_is_flagged_at_dimension_level(tmp_path):
    Y, st, syn, cols, t1, _ = _tier1(tmp_path, 200, 11, None, (0.0, -0.3, -0.6, -0.3))
    dims = mod.dim_items()
    dd = mod.dimension_dif(mod.dif_family(Y, st, cols, t1, mod.matching_items()), dims)
    assert syn
    assert dd["riasec_i"]["cls"] == "C" and dd["riasec_i"]["focal"] == [2]
    assert dd["quant_intensity"]["cls"] == "C" and dd["quant_intensity"]["focal"] == [1]
    assert all(v["cls"] == "A" for k, v in dd.items() if k not in ("riasec_i", "quant_intensity"))


def test_no_dif_data_raise_no_tier1_flag(tmp_path):
    Y, st, _, cols, t1, _ = _tier1(tmp_path, 200, 12, {}, (0.0, 0.0, 0.0, 0.0))
    dif = mod.dif_family(Y, st, cols, t1, mod.matching_items())
    assert all(r["cls"] == "A" for r in dif.values())


def test_dry_run_writes_report_cfa_files_and_reads_fit(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}
    folder = tmp_path / "study"
    make = [sys.executable, str(SCRIPT), "--make-fake", str(folder), "--n-per", "30"]
    assert subprocess.run(make, capture_output=True, text=True, env=env).returncode == 0
    run = [sys.executable, str(SCRIPT), "--data", str(folder)]
    res = subprocess.run(run, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-800:]
    text = (folder / "w73_analysis.md").read_text(encoding="utf-8")
    assert "含合成資料" in text and "不具約束力" in text and "檢定力不足" in text
    for head in ("## 1.", "## 2.", "## 3.", "## 4."):
        assert head in text, head
    with open(folder / "w73_cfa_items.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "stratum" and len(rows[0]) == 21
    assert (folder / "w73_cfa.R").exists()
    with open(folder / "w73_cfa_fit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "step", "converged", "cfi", "rmsea", "chisq", "df"])
        for m in ("riasec", "core"):
            w.writerow([m, "configural", "TRUE", 0.97, 0.05, 50, 40])
            w.writerow([m, "thresholds", "TRUE", 0.968, 0.05, 60, 70])
            w.writerow([m, "scalar", "TRUE", 0.95, 0.07, 90, 85])
    res = subprocess.run(run, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-800:]
    text = (folder / "w73_analysis.md").read_text(encoding="utf-8")
    assert "| riasec | ✅ | ✅ | ❌ |" in text


# --- probes (blind spots registered by the plan) -------------------------------------


@_probe("B-C1-40: a two-item dimension cannot localise DIF; the partner item is "
        "flagged too")  # fmt: skip
def test_probe_two_item_dimension_localises_dif(tmp_path):
    Y, st, _, cols, t1, _ = _tier1(tmp_path, 300, 13, {"riasec_i#1": {"group": 2, "shift": 1.0}},
                                   (0.0, 0.0, 0.0, 0.0))  # fmt: skip
    dif = mod.dif_family(Y, st, cols, t1, mod.matching_items())
    assert dif["riasec_i#1"]["cls"] == "C"
    assert dif["riasec_i#2"]["cls"] == "A"


@_probe("B-C1-41: a short external anchor (Tier 2) flags items without DIF when strata "
        "differ in mean by about 1 SD")  # fmt: skip
def test_probe_tier2_quiet_without_dif_under_impact():
    bank = [it["item_id"] for it in mod.item_bank()]
    pars = dict(zip(bank, mod.item_params(len(bank), np.random.default_rng(mod.SEED)),
                    strict=True))  # fmt: skip
    cols, (_, t2) = {i: j for j, i in enumerate(bank)}, mod.tiers()
    flags = 0
    for seed in range(3):
        Y, st = mod.sim_items(200, (0.0, -0.5, -1.0, -0.5), None, pars,
                              np.random.default_rng(seed))  # fmt: skip
        dif = mod.dif_family(Y, st, cols, t2, mod.matching_items(), lr=False)
        flags += sum(dif[i]["cls"] == "C" for i in t2)
    assert flags / (3 * len(t2)) <= 0.05


@_probe("B-C1-42: at the planned 40 per stratum a large DIF (delta = 1.0) is usually missed")
def test_probe_planned_n_detects_large_dif():
    bank = [it["item_id"] for it in mod.item_bank()]
    pars = dict(zip(bank, mod.item_params(len(bank), np.random.default_rng(mod.SEED)),
                    strict=True))  # fmt: skip
    t1 = mod.dim_items()["riasec_i"]
    cols = {i: bank.index(i) for i in t1}
    rng = np.random.default_rng(15)
    hits = 0
    for _ in range(20):
        Y, st = mod.sim_items(40, mod.POWER_IMPACT, {"group": 2, "shift": 1.0}, pars, rng)
        dif = mod.dif_family(Y, st, cols, t1, mod.matching_items())
        hits += any(dif[i]["cls"] == "C" for i in t1)
    assert hits / 20 >= 0.8
