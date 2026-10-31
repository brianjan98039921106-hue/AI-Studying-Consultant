"""W-81/W-82: feedback built by the web engine is parsed and scored by
scripts/score_feedback.py (docs/evaluation-plan.md §3). Synthetic feedback only."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_feedback.py"
spec = importlib.util.spec_from_file_location("score_feedback", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
NODE = shutil.which("node")


def _fb(ratings, considering=(), fit=4, bg="adjacent"):
    recs = [{"section": "new" if i < 10 else "known", "rank": i + 1 if i < 10 else i - 9,
             "id": f"t{i}", "cip": "27.0501", "name": f"Degree {i}", "rating": r,
             "already_considering": i in considering, "want_to_explore": r == 5}
            for i, r in enumerate(ratings)]  # fmt: skip
    return {"schema": "aisc-feedback/1", "data_version": 1, "scorer": "x", "date": "2026-10-31",
            "background": bg, "duration_sec": 720, "answered": 70, "total_items": 73,
            "profile_fit": fit, "profile": {}, "prior_list": "", "unclear_items": "",
            "recs": recs}  # fmt: skip


def test_score_counts_new_relevant_and_far_false_positives():
    r = mod.score(_fb([5, 4, 4, 1, 1, 3, None, 2, 5, 4, 5, 1], considering={1}))
    assert r["nr10"] == 4  # ranks 1, 3, 9, 10 (rank 2 already considered)
    assert r["far_fp"] == 2 and r["explore"] == 3 and r["rated"] == 11
    assert r["r15"] == pytest.approx(6 / 11)


def test_diagnosis_follows_the_plan_table():
    assert "問卷" in mod.diagnose({"fit": 3, "nr10": 5, "explore": 2})
    assert "C0、C2" in mod.diagnose({"fit": 4, "nr10": 1, "explore": 2})
    assert "呈現" in mod.diagnose({"fit": 5, "nr10": 4, "explore": 0})


def test_extract_ignores_chat_text_and_duplicates(tmp_path):
    block = "AISC-FEEDBACK v1\n" + json.dumps(_fb([5] * 15)) + "\nEND"
    (tmp_path / "chat.txt").write_text(f"hi, here it is:\n{block}\nthanks\n{block}\n", "utf-8")
    (tmp_path / "other.txt").write_text(
        "AISC-FEEDBACK v1\n" + json.dumps(_fb([1] * 15)) + "\nEND", "utf-8"
    )
    fbs = mod.extract([tmp_path])
    assert len(fbs) == 2
    md = mod.summary(fbs)
    assert "回饋份數 2" in md and "評 1–2 分的學門" in md


def test_bad_schema_is_refused(tmp_path):
    bad = _fb([5] * 15)
    bad["schema"] = "other/9"
    (tmp_path / "x.txt").write_text("AISC-FEEDBACK v1\n" + json.dumps(bad) + "\nEND", "utf-8")
    with pytest.raises(SystemExit, match="schema"):
        mod.extract([tmp_path])


NODE_SCRIPT = r"""
const { makeEngine } = require(process.argv[2]);
const data = require(process.argv[3]);
const e = makeEngine(data);
const { needs, groupScores } = e.scoreAnswers({ "riasec_i#1": 5, "group_7": 5 });
const { vector } = e.complete(needs, groupScores);
const res = e.discover(vector, { [e.shortlist(vector)[3]]: "heard" });
const ratings = Object.fromEntries(res.discoveries.map((r, i) => [r.id, (i % 5) + 1]));
const fb = e.makeFeedback({ result: res, ratings, considering: { [res.discoveries[3].id]: true },
  explore: {}, profileFit: 4, background: "distant", durationSec: 900, answered: 2, vector,
  prior: "", unclear: "第 5 題", date: "2026-10-31" });
let bad = 0;
try {
  e.makeFeedback({ result: res, ratings: { [res.discoveries[0].id]: 7 }, considering: {},
    explore: {}, profileFit: 4, background: "", durationSec: 1, answered: 1, vector });
} catch (err) { bad = 1; }
process.stdout.write("chat\n" + e.feedbackText(fb) + "\nbye\nBAD=" + bad);
"""


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_feedback_from_the_web_engine_is_scored(tmp_path):
    (tmp_path / "fb.js").write_text(NODE_SCRIPT, encoding="utf-8")
    args = [
        str(tmp_path / "fb.js"),
        str(ROOT / "web" / "engine.js"),
        str(ROOT / "web" / "data.json"),
    ]
    run = subprocess.run([NODE, *args], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-500:]
    assert run.stdout.endswith("BAD=1")  # out-of-range rating refused
    (tmp_path / "msg.txt").write_text(run.stdout, encoding="utf-8")
    out = tmp_path / "summary.md"
    cmd = [sys.executable, str(SCRIPT), str(tmp_path / "msg.txt"), "--out", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    fb = mod.extract([tmp_path / "msg.txt"])[0]
    assert fb["background"] == "distant" and len(fb["profile"]) == 51
    assert "name" not in json.dumps(list(fb.keys()))  # no personal fields at the top level
    r = mod.score(fb)
    # ratings 1..5 cycling over 10 discoveries: ranks rated 4 or 5 = 4, minus rank 4 (considered)
    assert r["nr10"] == 3 and r["far_fp"] == 2 and r["fit"] == 4
    assert "第 5 題" in out.read_text(encoding="utf-8")
