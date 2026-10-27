"""W-41 pre-registered analysis script (L2): statistics helpers and a dry run on
synthetic files written to pytest's temporary directory."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_w41_study.py"
spec = importlib.util.spec_from_file_location("analyze_w41_study", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_cronbach_alpha_matches_the_formula():
    X = np.array([[1, 2, 2], [2, 2, 3], [3, 4, 3], [4, 4, 5], [5, 5, 4]], dtype=float)
    k, item_var, tot_var = 3, X.var(axis=0, ddof=1).sum(), X.sum(axis=1).var(ddof=1)
    assert abs(mod.cronbach_alpha(X) - k / (k - 1) * (1 - item_var / tot_var)) < 1e-12
    assert mod.cronbach_alpha(np.hstack([X[:, :1]] * 3)) == 1.0


def test_top5_recall_breaks_ties_in_expectation():
    v = np.zeros(10)
    v[:5] = 1.0
    assert mod.top5_recall(v, v) == 1.0
    tie = np.zeros(10)
    tie[:4], tie[4:8] = 1.0, 0.5  # 4 sure + 1 of 4 tied entries
    p = mod.top5_probs(tie)
    assert abs(p.sum() - 5) < 1e-12 and np.allclose(p[4:8], 0.25)
    assert abs(mod.top5_recall(v, tie) - (4 + 0.25) / 5) < 1e-12


def test_wilson_interval_contains_the_rate():
    lo, hi = mod.wilson(30, 40)
    assert lo < 0.75 < hi and lo > 0.55 and hi < 0.9


def test_templates_match_the_item_bank():
    items = [it["item_id"] for it in json.loads(mod.BANK.read_text(encoding="utf-8"))["items"]]
    for name, lead in (("study_s1.csv", ["respondent_id", "stratum"]),
                       ("study_s2.csv", ["respondent_id"])):  # fmt: skip
        with open(ROOT / "data" / "templates" / name, encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == [*lead, *items]


def test_dry_run_on_synthetic_files(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}
    folder = tmp_path / "study"
    make = [sys.executable, str(SCRIPT), "--make-fake", str(folder), "--n", "60",
            "--n-retest", "30"]  # fmt: skip
    assert subprocess.run(make, capture_output=True, text=True, env=env).returncode == 0
    run = subprocess.run([sys.executable, str(SCRIPT), "--data", str(folder)],
                         capture_output=True, text=True, env=env)  # fmt: skip
    assert run.returncode == 0, run.stderr[-800:]
    text = (folder / "analysis.md").read_text(encoding="utf-8")
    assert "含合成資料" in text and "不具約束力" in text
    for head in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "F115"):
        assert head in text, head
