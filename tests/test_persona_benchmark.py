"""W-80 persona benchmark harness (L2). Synthetic personas and fake expectations; the
real expectations are written by the raters (docs/persona-benchmark.md)."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_persona_benchmark.py"
spec = importlib.util.spec_from_file_location("run_persona_benchmark", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_persona_benchmark"] = mod
spec.loader.exec_module(mod)


class _T:
    def __init__(self, pid, cip):
        self.program_id, self.cip = pid, cip


def test_personas_cover_all_backgrounds_and_give_73_valid_answers():
    from aisc.profiling.hierarchy import AREA_GROUPS
    from aisc.profiling.questionnaire import load_questions

    personas = mod.load_personas()
    assert len(personas) == 16
    assert {p["background"] for p in personas} == set(mod.BACKGROUNDS)
    qs = load_questions(ROOT / "data" / "items" / "item_bank_v2.json")
    a = mod.persona_answers(personas[0], qs)
    assert len(a) == 73 and set(a.values()) <= {1, 2, 3, 4, 5}
    p = personas[0]  # P01: finance 5 -> group 1 answer 5
    g1 = next(q for q in qs if q.stage == "group" and q.group == 1)
    assert a[g1.item_id] == max(v for k, v in p["areas"].items() if AREA_GROUPS[k][0] == 1)


def test_longest_prefix_wins():
    labels = [("27", "acceptable"), ("27.05", "core"), ("27.0599", "exclude")]
    assert mod.label_of("27.0501", labels) == "core"
    assert mod.label_of("27.0101", labels) == "acceptable"
    assert mod.label_of("27.0599", labels) == "exclude"
    assert mod.label_of("52.0801", labels) is None


def test_metrics_perfect_and_violation():
    types = [_T(f"t{i}", f"52.08{i:02d}") for i in range(10)] + [_T("x", "01.0000")]
    labels = [("52.08", "core"), ("01", "exclude")]
    perfect = mod.metrics(types[:10], labels, types)
    assert perfect["ndcg"] == pytest.approx(1.0) and perfect["hit"] and perfect["precision"] == 1
    bad = mod.metrics([types[-1], *types[:9]], labels, types)
    assert bad["violations"] == ["01.0000"] and bad["ndcg"] < 1


def test_agreement_identical_and_opposite():
    types = [_T(f"t{i}", c) for i, c in enumerate(["52.0801", "27.0501", "01.0000", "11.0701"])]
    a = [("52.08", "core"), ("27", "acceptable"), ("01", "exclude")]
    same = mod.agreement(a, a, types)
    assert same["kappa"] == pytest.approx(1.0) and same["core_jaccard"] == 1.0
    flip = mod.agreement(a, [("01", "core"), ("52.08", "exclude")], types)
    assert flip["kappa"] < 0 and flip["core_jaccard"] == 0


def test_guard_refuses_uncommitted_expectations():
    with pytest.raises(SystemExit, match="committed"):
        mod.require_committed([ROOT / "data" / "benchmark" / "no_such_file.csv"])
    mod.require_committed([ROOT / "data" / "items" / "item_bank_v2.json"])


def test_dry_run_with_fake_expectations(tmp_path):
    shutil.copy(ROOT / "data" / "benchmark" / "personas.json", tmp_path / "personas.json")
    rows = "persona_id,cip_prefix,label\nP01,52.08,core\nP01,01,exclude\nP02,11,core\n"
    (tmp_path / "expectations_fake.csv").write_text(rows, encoding="utf-8")
    (tmp_path / "expectations_other.csv").write_text(rows, encoding="utf-8")
    out = tmp_path / "res.json"
    cmd = [sys.executable, str(SCRIPT), "--rater", "fake", "--second", "other",
           "--allow-uncommitted", "--bench-dir", str(tmp_path), "--json-out", str(out)]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-800:]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res["rows"]) == {"P01", "P02"} and set(res["agreement"]) == {"P01", "P02"}
    assert all(len(r["MODEL"]["top"]) == 10 for r in res["rows"].values())
    assert {"vs_b1", "vs_b2", "hit_rate", "violations"} == set(res["summary"]["checks"])
    assert not (ROOT / "results" / "persona-benchmark.md").exists()


def test_regression_check_runs():
    run = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
