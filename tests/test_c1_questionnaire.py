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


def test_probe_self_service_list_shows_distinct_degree_types(tmp_path):
    """B-C2-30 (fixed): the list shows distinct degree types, not one degree at
    many schools."""
    lines = _self_service(tmp_path)
    degrees = {next(seg for seg in ln.split("｜") if "(Master's)" in seg) for ln in lines}
    assert len(lines) >= 5 and len(degrees) == len(lines), lines[:3]


def test_merge_identical_programs_keeps_members_and_singletons():
    from dataclasses import replace

    from aisc.catalog.real import load_real_catalog_v0
    from aisc.pipeline import merge_identical_programs

    real = load_real_catalog_v0(ROOT / "data" / "reference").programs
    odd = replace(real[0], program_id="odd", attributes={**real[0].attributes, "riasec_r": 0.0})
    merged, members = merge_identical_programs([*real, odd])
    assert sum(len(v) for v in members.values()) == len(real) + 1
    assert len({p.program_id for p in merged}) == len(merged) == len(members)
    assert members["odd"] == [odd]  # a programme with no twin is kept unchanged
    big = max(members.values(), key=len)
    rep = next(p for p in merged if members[p.program_id] is big)
    assert rep.program_id.startswith("type:") and f"等 {len(big)} 所" in rep.school
    assert {p.cip for p in big} == {rep.cip} and rep.attributes == big[0].attributes


# --- independent review of W-52 (2026-10-29): B-C2-31..33, B-C1-41 -------------------


def _probe(reason):
    def mark(fn):
        return pytest.mark.probe(pytest.mark.xfail(strict=True, raises=AssertionError,
                                                   reason=reason)(fn))  # fmt: skip

    return mark


def _real():
    from aisc.catalog.real import load_real_catalog_v0

    return load_real_catalog_v0(ROOT / "data" / "reference").programs


@_probe("B-C2-31: merging hides fields that hard constraints read")
def test_probe_merge_keeps_programmes_with_different_constraint_fields_apart():
    from dataclasses import replace

    from aisc.pipeline import merge_identical_programs

    base = next(p for p in _real() if p.cip == "27.0501")
    a = replace(base, program_id="a", school="A", tuition_usd=80_000)
    b = replace(base, program_id="b", school="B", tuition_usd=20_000)
    c = replace(base, program_id="c", school="C", tuition_usd=20_000, synthetic=True)
    merged, members = merge_identical_programs([a, b, c])
    assert len(merged) == 3, [len(v) for v in members.values()]


@_probe("B-C2-32: anchors given as real programme ids are lost after merging")
def test_probe_anchor_ids_can_be_translated_to_degree_types():
    from aisc import pipeline

    real = _real()
    merged, members = pipeline.merge_identical_programs(real)
    translate = getattr(pipeline, "to_type_ids", None)
    assert translate is not None, "no helper to translate programme ids to type ids"
    anchor = next(p for p in real if p.cip == "27.0501").program_id
    (tid,) = translate([anchor], members)
    assert any(m.program_id == anchor for m in members[tid])


@_probe("B-C2-33: degree-type ids depend on input order")
def test_probe_type_ids_are_stable_under_input_order():
    import random

    from aisc.pipeline import merge_identical_programs

    real = _real()
    ids1 = {p.cip: p.program_id for p in merge_identical_programs(real)[0]}
    shuffled = real[:]
    random.Random(0).shuffle(shuffled)
    ids2 = {p.cip: p.program_id for p in merge_identical_programs(shuffled)[0]}
    assert ids1 == ids2


def _run_script(tmp_path, payload):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "r.md"
    cmd = [sys.executable, str(ROOT / "examples" / "self_service.py"), "--answers", str(path),
           "--out", str(out)]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT), out


@_probe("B-C1-41: the self-service flow accepts bad consent and answers silently")
def test_probe_self_service_validates_consent_and_answers(tmp_path):
    run, out = _run_script(tmp_path, {"consent": "false", "items": {}})
    assert not out.exists(), "string 'false' was taken as consent"
    run, out = _run_script(tmp_path, {"consent": True, "items": {"riasec_r#1": True}})
    assert run.returncode != 0 and "1-5" in run.stderr + run.stdout
    run, out = _run_script(tmp_path, {"consent": True, "items": {"riasec_r_1": 3}})
    assert run.returncode != 0 and "riasec_r_1" in run.stderr + run.stdout
    run, out = _run_script(tmp_path, {"consent": True, "items": {}})
    assert run.returncode == 0 and not out.exists(), "a report built on no answers"
