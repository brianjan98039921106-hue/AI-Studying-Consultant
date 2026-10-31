"""W-80 persona benchmark with blind pooled judging (L2). Synthetic personas and fake
judgments; the real judgments come from the raters (docs/persona-benchmark.md)."""

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


@pytest.fixture(scope="module")
def tops():
    return mod.method_tops(mod.load_personas())


def _block(personas: dict, rater="fake") -> str:
    """The text web/judge.html copies (payload() in the page)."""
    obj = {"schema": "aisc-judgments/1", "rater": rater, "date": "2026-10-31",
           "pool_version": 1, "personas": personas}  # fmt: skip
    return f"chat text before\nAISC-JUDGMENTS v1\n{json.dumps(obj)}\nEND\nafter\n"


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


def test_committed_pool_contains_every_current_top10(tops):
    """If this fails the model changed after the pool was built: extend the pool and judge
    the new types (or rebuild it if nobody has judged yet)."""
    pool = json.loads(mod.POOL.read_text(encoding="utf-8"))
    assert pool["seed"] == mod.SEED and set(pool["personas"]) == set(tops)
    for pid, by_method in tops.items():
        items = pool["personas"][pid]
        assert len(items) == len(set(items))
        listed = {i for m in mod.METHODS for i in by_method[m]}
        assert listed <= set(items), pid
        assert len(set(items) - listed) == mod.DECOYS
        assert items != sorted(items)  # shuffled, so position does not reveal the source


def test_metrics_perfect_unjudged_and_must_include():
    top = [f"t{i}" for i in range(10)]
    series = {f"t{i}": f"52.0{i % 3}" for i in range(12)}
    judged = {i: 2 for i in top} | {"t10": 0, "t11": 1}
    m = mod.metrics(top, judged, ["t0", "t10"], series)
    assert m["ndcg"] == pytest.approx(1.0) and m["hit"] and m["precision"] == 1
    assert m["mi_recall"] == 0.5 and m["coverage"] == 1 and m["series"] == 3
    worse = mod.metrics(["t10", *top[:9]], judged, [], series)
    assert worse["grade0"] == 1 and worse["ndcg"] < 1 and np_isnan(worse["mi_recall"])
    gap = mod.metrics(["new", *top[:9]], judged, [], series | {"new": "01.00"})
    assert gap["coverage"] == 0.9 and gap["grade0"] == 0  # unjudged counts 0 gain, not grade 0


def np_isnan(x):
    return x != x


def test_kappa_and_agreement():
    assert mod.kappa_quadratic([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert mod.kappa_quadratic([0, 0, 2, 2], [2, 2, 0, 0]) < 0
    a = {"P01": {"must": ["x", "y"], "grades": {"x": 2, "y": 1, "z": 0}}}
    b = {"P01": {"must": ["x"], "grades": {"x": 2, "y": 1, "z": 0, "w": 1}}}
    ag = mod.agreement(a, b)["P01"]
    assert ag["n"] == 3 and ag["exact"] == 1 and ag["must_jaccard"] == 0.5


def test_load_judgments_reads_page_format_and_later_blocks_win(tmp_path):
    f = tmp_path / "j.txt"
    first = {"P01": {"must_include": ["a"], "grades": {"a": 2, "b": 0}}}
    second = {"P01": {"must_include": ["b"], "grades": {"a": 1, "b": 1}}}
    f.write_text(_block(first) + _block(second), encoding="utf-8")
    assert mod.load_judgments(f) == {"P01": {"must": ["b"], "grades": {"a": 1, "b": 1}}}
    f.write_text(_block({"P01": {"must_include": [], "grades": {"a": 3}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="outside 0-2"):
        mod.load_judgments(f)
    f.write_text("no block here", encoding="utf-8")
    with pytest.raises(SystemExit, match="no AISC-JUDGMENTS"):
        mod.load_judgments(f)


def test_guard_refuses_uncommitted_judgments():
    with pytest.raises(SystemExit, match="committed"):
        mod.require_committed([ROOT / "data" / "benchmark" / "no_such_file.txt"])
    mod.require_committed([ROOT / "data" / "items" / "item_bank_v2.json"])


def test_build_pool_refuses_when_judgments_exist(monkeypatch, tmp_path):
    (tmp_path / "pool.json").write_text("{}", encoding="utf-8")
    (tmp_path / "judgments_brian.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(mod, "BENCH", tmp_path)
    monkeypatch.setattr(mod, "POOL", tmp_path / "pool.json")
    with pytest.raises(SystemExit, match="judgments exist"):
        mod.build_pool()


def test_judging_page_embeds_data_safely(tmp_path):
    out = tmp_path / "judge.html"
    mod.build_page(out)
    page = out.read_text(encoding="utf-8")
    assert "window.JUDGE_DATA = null" not in page and "<script src" not in page
    start = page.index("window.JUDGE_DATA = ") + len("window.JUDGE_DATA = ")
    blob = page[start : page.index(";</script>", start)]
    assert "<" not in blob
    data = json.loads(blob)
    assert data["consultant"] == ["P02", "P06", "P09", "P13"]
    assert all(i in data["types"] for ids in data["pool"].values() for i in ids)
    assert len(data["types"]) == 863 and len({t["cip"] for t in data["types"].values()}) == 863
    fams = data["families"]  # the picker groups by CIP family: every family needs a label
    assert {t["fam"] for t in data["types"].values()} == set(fams)
    assert all(f["zh"] and f["en"] for f in fams.values())
    assert sum(f["n"] for f in fams.values()) == 863
    assert fams["27"]["en"] == "Mathematics and Statistics"


def test_dry_run_with_fake_judgments(tmp_path, tops):
    shutil.copy(ROOT / "data" / "benchmark" / "personas.json", tmp_path / "personas.json")
    pool = json.loads(mod.POOL.read_text(encoding="utf-8"))["personas"]
    model = {pid: tops[pid]["MODEL"] for pid in ("P01", "P02")}
    # the fake rater likes exactly MODEL's list, so MODEL must beat both baselines
    fake = {pid: {"must_include": model[pid][:2],
                  "grades": {i: (2 if i in model[pid] else 0) for i in pool[pid]}}
            for pid in model}  # fmt: skip
    (tmp_path / "judgments_fake.txt").write_text(_block(fake), encoding="utf-8")
    (tmp_path / "judgments_other.txt").write_text(_block(fake, "other"), encoding="utf-8")
    out = tmp_path / "res.json"
    cmd = [sys.executable, str(SCRIPT), "--score", "--rater", "fake", "--second", "other",
           "--allow-uncommitted", "--bench-dir", str(tmp_path), "--json-out", str(out)]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-800:]
    res = json.loads(out.read_text(encoding="utf-8"))
    s = res["summary"]
    assert set(res["rows"]) == {"P01", "P02"} and set(res["agreement"]) == {"P01", "P02"}
    assert s["MODEL"]["ndcg"] == pytest.approx(1.0) and s["MODEL"]["mi_recall"] == 1
    assert set(s["checks"]) == {"vs_b1", "vs_b2", "hit_rate", "grade0", "mi_recall"}
    assert s["decoy_mean"] == 0 and res["agreement"]["P01"]["kappa"] == pytest.approx(1.0)
    assert not (ROOT / "results" / "persona-benchmark.md").exists()
    assert "評分者一致性" in mod.report(res)


def test_regression_check_runs():
    run = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
