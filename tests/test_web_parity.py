"""W-53: the web page's JavaScript engine must give the same results as the Python
pipeline (A-C2-60), and web/data.json must be current. Needs Node.js (skipped if
absent; GitHub's runners have it)."""

import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from aisc.catalog.real import load_real_catalog_v0
from aisc.discovery import discover, shortlist
from aisc.pipeline import merge_identical_programs, prepare_profile
from aisc.profiling import apply_familiarity
from aisc.profiling.questionnaire import Questionnaire, load_questions
from aisc.report import reason_keys

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

NODE_SCRIPT = r"""
const fs = require("fs");
const { makeEngine } = require(process.argv[2]);
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const cases = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const eng = makeEngine(data);
const out = cases.map((c) => {
  const { needs, groupScores } = eng.scoreAnswers(c.answers);
  const { vector } = eng.complete(needs, groupScores);
  const res = eng.discover(vector, c.familiarity);
  return {
    needs, groupScores, vector,
    areaOrder: eng.areaOrder(groupScores),
    shortlist: eng.shortlist(vector),
    discoveries: res.discoveries.map((r) => [r.id, r.relevance]),
    reasons: res.discoveries.map((r) => eng.reasons(r, vector).map((x) => x.key)),
    knownGood: res.knownGood.map((r) => [r.id, r.relevance]),
  };
});
process.stdout.write(JSON.stringify(out));
"""


def test_web_data_is_current():
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web_data.py"), "--check"],
                         capture_output=True, text=True)  # fmt: skip
    assert run.returncode == 0, run.stdout + run.stderr


def _cases(types, n=24, seed=53):
    rng = np.random.default_rng(seed)
    questions = load_questions(ROOT / "data" / "items" / "item_bank_v2.json")
    out = []
    for c in range(n):
        qs = Questionnaire(questions)
        skip = (0.0, 0.3, 0.9)[c % 3]
        base = rng.uniform(1, 5, 60)
        while qs.next_page():
            for q in qs.next_page():
                if (q.stage == "group" and c % 4 == 3) or rng.random() < skip:
                    qs.answer(q.item_id, None)
                else:
                    v = base[zlib.crc32((q.dimension or q.item_id).encode()) % 60] + rng.normal(
                        0, 1
                    )
                    qs.answer(q.item_id, int(np.clip(round(v), 1, 5)))
        prof = prepare_profile(qs.profile(f"c{c}"), types)
        short = shortlist(prof, types, n=25)
        fam = {pid: ("heard" if i % 3 == 1 else "can_describe" if i % 5 == 2 else "unknown")
               for i, pid in enumerate(short)}  # fmt: skip
        res = discover(apply_familiarity(prof, fam), types, k_new=10, k_known=5, asked=short)
        out.append({
            "answers": dict(qs.answers),
            "familiarity": fam,
            "expected": {
                "needs": qs.needs(),
                "groupScores": {str(k): v for k, v in qs.group_scores().items()},
                "vector": [float(v) for v in prof.vector],
                "areaOrder": [q.dimension for q in qs.page("area")],
                "shortlist": short,
                "discoveries": [[r.program.program_id, r.relevance] for r in res.discoveries],
                "reasons": [reason_keys(r, prof.needs) for r in res.discoveries],
                "knownGood": [[r.program.program_id, r.relevance] for r in res.known_good],
            },
        })  # fmt: skip
    return out


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_javascript_engine_matches_the_python_pipeline(tmp_path):
    types, _ = merge_identical_programs(load_real_catalog_v0(ROOT / "data" / "reference").programs)
    cases = _cases(types)
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "run.js").write_text(NODE_SCRIPT, encoding="utf-8")
    run = subprocess.run([NODE, str(tmp_path / "run.js"), str(ROOT / "web" / "engine.js"),
                          str(ROOT / "web" / "data.json"), str(tmp_path / "cases.json")],
                         capture_output=True, text=True)  # fmt: skip
    assert run.returncode == 0, run.stderr[-800:]
    got = json.loads(run.stdout)
    assert len(got) == len(cases)
    for i, (c, g) in enumerate(zip(cases, got, strict=True)):
        e = c["expected"]
        assert g["needs"].keys() == e["needs"].keys(), i
        assert all(abs(g["needs"][k] - v) < 1e-12 for k, v in e["needs"].items()), i
        assert g["groupScores"] == pytest.approx(e["groupScores"]), i
        assert np.allclose(g["vector"], e["vector"], atol=1e-12), i
        assert g["areaOrder"] == e["areaOrder"], i
        assert g["shortlist"] == e["shortlist"], i
        assert g["reasons"] == e["reasons"], i
        for key in ("discoveries", "knownGood"):
            assert [x[0] for x in g[key]] == [x[0] for x in e[key]], (i, key)
            assert np.allclose([x[1] for x in g[key]], [x[1] for x in e[key]], atol=1e-9)


def test_bundle_inlines_engine_and_data(tmp_path):
    out = tmp_path / "page.html"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web_data.py"), "--bundle",
                          str(out)], capture_output=True, text=True)  # fmt: skip
    assert run.returncode == 0, run.stderr
    page = out.read_text(encoding="utf-8")
    assert '<script src="engine.js"></script>' not in page
    assert "function makeEngine" in page and "window.AISC_DATA = {" in page
    assert "</script>" not in page.split("window.AISC_DATA = ", 1)[1].split(";</script>", 1)[0]


# --- independent review of W-53 (2026-10-29): B-C2-34, B-C1-42 (started as probes) ----


def test_probe_reasons_only_cite_needs_the_student_has():
    from aisc.pipeline import interactive_discovery
    from aisc.report import render_profile_report

    types, _ = merge_identical_programs(load_real_catalog_v0(ROOT / "data" / "reference").programs)
    qs = Questionnaire(load_questions(ROOT / "data" / "items" / "item_bank_v2.json"))
    for q in qs.page("core"):
        qs.answer(q.item_id, 5 if q.dimension == "riasec_i" else 2)
    student = qs.profile("probe")
    res = interactive_discovery(student, types, lambda ids: dict.fromkeys(ids, "unknown"))
    md = render_profile_report(student, prepare_profile(student, types), res)
    low = ("實用型", "藝術型", "社會型", "企業型", "事務型", "數理強度", "程式強度")
    cited = [lab for line in md.splitlines() if "（+" in line for lab in low if f"{lab}（+" in line]
    assert not cited, cited[:5]


def test_probe_web_page_structure_privacy_and_accessibility(tmp_path):
    page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in page, "third-party font request"
    assert "documentElement.lang" in page
    assert "aria-describedby" in page
    assert "超過一半" in page, "no warning when most items are skipped"
    out = tmp_path / "standalone.html"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web_data.py"), "--bundle",
                          str(out)], capture_output=True, text=True)  # fmt: skip
    assert run.returncode == 0, run.stderr
    head = out.read_text(encoding="utf-8")[:600].lower()
    assert head.startswith("<!doctype html>") and 'name="viewport"' in head and "zh-hant" in head
