"""W-52: the student self-report questionnaire on the v2 item bank (L2), and the
self-service terminal flow on the real catalogue v0."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aisc.dimensions import NEED_KEYS_V2
from aisc.pipeline import prepare_profile
from aisc.profiling.hierarchy import GROUP_LABELS
from aisc.profiling.questionnaire import Questionnaire, load_questions

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "items" / "item_bank_v2.json"


def _run_all(qs, value_of):
    while qs.next_page():
        for q in qs.next_page():
            qs.answer(q.item_id, value_of(q))


def test_bank_covers_every_v2_dimension_and_group():
    qs = load_questions(BANK)
    assert len(qs) == 73
    assert {q.dimension for q in qs if q.dimension} == set(NEED_KEYS_V2)
    assert {q.group for q in qs if q.stage == "group"} == set(GROUP_LABELS)
    assert all(len(q.labels_zh) == 5 for q in qs)


def test_pages_run_core_group_area_and_finish():
    qs = Questionnaire(load_questions(BANK))
    seen = []
    while qs.next_page():
        page = qs.next_page()
        seen.append(page[0].stage)
        for q in page:
            qs.answer(q.item_id, 3)
    assert seen == ["core", "group", "area"] and qs.stage() is None
    assert qs.progress() == (73, 73)


def test_scoring_is_the_mean_of_rescaled_answers():
    qs = Questionnaire(load_questions(BANK))
    qs.answer("riasec_r#1", 5)
    qs.answer("riasec_r#2", 2)
    qs.answer("group_7", 4)
    qs.answer("riasec_i#1", None)  # skipped
    needs = qs.needs()
    assert needs["riasec_r"] == pytest.approx((1.0 + 0.25) / 2)
    assert "riasec_i" not in needs and qs.skipped() == ["riasec_i#1"]
    assert qs.group_scores() == {7: 0.75}
    with pytest.raises(ValueError):
        qs.answer("riasec_r#1", 6)
    with pytest.raises(ValueError):
        qs.answer("no_such_item", 3)


def test_area_items_follow_the_group_answers():
    qs = Questionnaire(load_questions(BANK))
    for q in qs.page("group"):
        qs.answer(q.item_id, 5 if q.group == 7 else 1)
    first = qs.page("area")[0]
    from aisc.profiling.hierarchy import AREA_GROUPS

    assert 7 in AREA_GROUPS[first.dimension]
    assert len(qs.page("area")) == 41


def test_profile_completes_and_keeps_answers():
    from aisc.catalog.real import load_real_catalog_v0

    qs = Questionnaire(load_questions(BANK))
    _run_all(qs, lambda q: None if q.stage == "area" else 4)
    p = qs.profile("s1")
    assert p.dims == "v2" and len(p.group_scores) == 12
    assert not any(k.startswith("bi_") for k in p.needs)
    catalog = load_real_catalog_v0(ROOT / "data" / "reference").programs
    done = prepare_profile(p, catalog)
    assert not done.missing_dimensions and done.needs["riasec_r"] == 0.75


def _quant_answers(tmp_path):
    items = json.loads(BANK.read_text(encoding="utf-8"))["items"]
    hi = {"riasec_i", "riasec_c", "quant_intensity", "programming_intensity",
          "bi_mathematics_statistics", "bi_information_technology", "bi_finance",
          "bi_management_administration"}  # fmt: skip
    a = {it["item_id"]: 5 if it["dimension"] in hi or it.get("group") in (1, 7) else 1
         for it in items}  # fmt: skip
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"consent": True, "items": a}), encoding="utf-8")
    return path


def _self_service(tmp_path, *extra):
    cmd = [sys.executable, str(ROOT / "examples" / "self_service.py"), "--answers",
           str(_quant_answers(tmp_path)), "--out", str(tmp_path / "r.md"), *extra]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-800:]
    section = run.stdout.split("== 你可能沒想過 ==")[1].split("完整報告")[0]
    return [ln.strip() for ln in section.splitlines() if ln.strip()]


def test_self_service_runs_without_consent_records_nothing(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"consent": False, "items": {}}), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "examples" / "self_service.py"), "--answers", str(path),
           "--out", str(tmp_path / "r.md")]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0 and not (tmp_path / "r.md").exists()


@pytest.mark.probe
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="B-C2-30: one degree type fills the self-service list")  # fmt: skip
def test_probe_self_service_list_shows_distinct_degree_types(tmp_path):
    lines = _self_service(tmp_path)
    degrees = {ln.split("｜")[-1] for ln in lines}
    assert len(lines) >= 5 and len(degrees) == len(lines), lines[:3]
