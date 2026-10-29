"""W-61 student-record intake (C1 / C6.4): L2 verification, probes and the D9 guard.

All records here are FICTIONAL; requirement texts and URLs are fixtures.
"""

import json
import re
from pathlib import Path

import pytest

from aisc.contracts import RequirementItem, StudentRecord
from aisc.profiling import (
    RecordIntakeError,
    intake_student_record,
    load_course_map,
    load_student_record,
    normalize_name,
)
from aisc.requirements import compute_gap, load_taxonomy

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "data" / "reference" / "requirement_taxonomy.json"
COURSE_MAP = ROOT / "data" / "reference" / "course_name_map_tw.json"
EXAMPLE = ROOT / "data" / "synthetic" / "student_record_example.json"
URL, DATE = "https://example.edu/fixture-requirements", "2026-10-29"


@pytest.fixture(scope="module")
def cmap():
    return load_course_map()


def _intake(cmap, **kw):
    data = {"schema": "student_record_intake/1", "student_id": "fx", "synthetic": True, **kw}
    return intake_student_record(data, cmap)


def _req(category, kind="course", level="required", **kw):
    return RequirementItem("p", kind, category, level, f"Fixture text for {category}", URL, DATE,
                           **kw)  # fmt: skip


# --- mapping table ---------------------------------------------------------------


def test_course_map_is_versioned_draft_and_consistent_with_taxonomy(cmap):
    spec = json.loads(COURSE_MAP.read_text(encoding="utf-8"))
    tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    assert spec["version"] == "draft-1" and spec["taxonomy_version"] == tax["version"]
    assert "NOT CONFIRMED" in spec["status"]  # the user confirms the Taiwan examples (W-61)
    courses = set(tax["categories"]["course"])
    mapped = {c for e in spec["courses"] for c in e["categories"]}
    assert mapped <= courses
    # every course category of the taxonomy has at least one Taiwanese example
    assert courses - mapped == set(), courses - mapped
    zh = [n for e in spec["courses"] for n in e["names"] if re.search(r"[一-鿿]", n)]
    assert len(zh) >= 100


@pytest.mark.parametrize(
    "name, expected",
    [
        ("微積分（一）", ("calculus_1",)),
        ("微積分(二)", ("calculus_2",)),
        ("微積分甲上", ("calculus_1",)),
        ("微積分", ("calculus_1",)),  # no ordinal: conservative
        ("Calculus II", ("calculus_2",)),
        ("線性代數", ("linear_algebra",)),
        ("統計學（英文授課）", ("statistics",)),
        ("商用統計學", ("statistics",)),  # not probability (B-C6-2)
        ("機率與統計", ("probability", "statistics")),
        ("程式設計一", ("programming_intro",)),
        ("Ｐｙｔｈｏｎ程式設計", ("programming_intro",)),  # full-width letters (NFKC)
        ("會計學（上）", ("accounting",)),
        ("經濟學（一）", ("microeconomics",)),
        ("經濟學（二）", ("macroeconomics",)),
        ("經濟學原理", ("microeconomics", "macroeconomics")),
        ("個體經濟學", ("microeconomics",)),
    ],
)
def test_taiwanese_course_names_map_by_explicit_rules(cmap, name, expected):
    source, cats, _ = cmap.map_course(name)
    assert source.startswith("table:") and cats == expected


@pytest.mark.parametrize("name", ["計算機概論", "商用微積分", "工程數學", "高等微積分"])
def test_ambiguous_names_are_listed_for_review_not_mapped(cmap, name):
    assert cmap.map_course(name)[:2] == ("review", ())


@pytest.mark.parametrize("name", ["作業研究", "體育", "微積分（五）", "統計"])
def test_unknown_names_are_unmatched_never_guessed(cmap, name):
    assert cmap.map_course(name)[:2] == ("unmatched", ())


def test_normalisation():
    assert normalize_name("微積分（一）") == ("微積分", 1)
    assert normalize_name(" Calculus  I ") == ("calculus", 1)
    assert normalize_name("Linear Algebra") == ("linearalgebra", None)
    assert normalize_name("TOEFL iBT", ordinals=False) == ("toeflibt", None)
    assert normalize_name("統計學（二）（全英語授課）") == ("統計學", 2)


def test_course_map_load_rejects_inconsistent_tables(tmp_path):
    spec = json.loads(COURSE_MAP.read_text(encoding="utf-8"))

    def write(mut):
        s = json.loads(json.dumps(spec))
        mut(s)
        p = tmp_path / "m.json"
        p.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        return p

    p = write(lambda s: s["courses"][0]["names"].append("線性代數"))
    with pytest.raises(ValueError, match="claimed by"):
        load_course_map(p)
    p = write(lambda s: s["courses"][0].__setitem__("categories", ["gre"]))
    with pytest.raises(ValueError, match="not a course category"):
        load_course_map(p)
    p = write(lambda s: s["courses"][0]["names"].append("微積分（三）"))
    with pytest.raises(ValueError, match="by_ordinal"):
        load_course_map(p)
    p = write(lambda s: s.__setitem__("taxonomy_version", "draft-0"))
    with pytest.raises(ValueError, match="taxonomy"):
        load_course_map(p)


# --- intake --------------------------------------------------------------------------


def test_example_file_is_synthetic_and_builds_a_record():
    res = load_student_record(EXAMPLE)
    assert res.synthetic and res.map_version == "draft-1"
    rec = res.record
    assert isinstance(rec, StudentRecord) and rec.target_apply_by is None
    assert {"calculus_1", "calculus_2", "statistics", "microeconomics", "accounting",
            "programming_intro", "english_proficiency", "work_experience",
            "bachelor_degree"} == set(rec.completed)  # fmt: skip
    assert set(rec.planned) == {"linear_algebra", "gre"}
    assert "2027 春季" in rec.planned["linear_algebra"]
    assert set(rec.unmapped) == {"計算機概論", "作業研究"}
    assert rec.online_only == {"programming_intro"}
    assert "會計學（下）" in rec.completed["accounting"]
    assert "會計學（上）" not in rec.completed["accounting"]  # failed (58) does not count
    assert "TOEFL iBT（95" in rec.completed["english_proficiency"]  # score kept as evidence
    assert "bachelor_equivalency" not in rec.completed  # never inferred (B-C6-5)


def test_multiple_degrees_and_degree_rules(cmap):
    rec = _intake(cmap, degrees=[
        {"level": "bachelor", "field": "Materials Science", "cip": "40.1001", "year": 2018},
        {"level": "professional", "field": "Dentistry (DDS)"},
    ]).record  # fmt: skip
    assert [d.level for d in rec.degrees] == ["bachelor", "professional"]
    assert "Materials Science" in rec.completed["bachelor_degree"]
    only_prof = _intake(cmap, degrees=[{"level": "other", "field": "五專 化工"}]).record
    assert "bachelor_degree" not in only_prof.completed  # A-C6-12: not inferred


def test_grades(cmap):
    def done(grade):
        return (
            "linear_algebra"
            in _intake(cmap, courses=[{"name": "線性代數", "grade": grade}]).record.completed
        )

    for g in (60, "60", "A+", "C-", "P", "通過", "抵免", None, ""):
        assert done(g), g
    for g in (59, "F", "不及格", "W", "停修"):
        assert not done(g), g
    for g in ("D", "3.7", "A++", 120, True):
        with pytest.raises(RecordIntakeError):
            done(g)


def test_manual_mapping_and_explicit_no_category(cmap):
    res = _intake(cmap, courses=[
        {"name": "作業研究", "categories": ["linear_algebra"]},
        {"name": "計算機概論", "categories": ["programming_intro"]},
        {"name": "體育", "categories": []},
    ])  # fmt: skip
    assert res.record.unmapped == {}
    assert {"linear_algebra", "programming_intro"} <= set(res.record.completed)
    assert [e.source for e in res.entries] == ["manual", "manual", "none"]


def test_exams_and_experience(cmap):
    rec = _intake(cmap,
        exams=[{"name": "托福", "score": 100}, {"name": "IELTS Academic", "score": "7.5"},
               {"name": "多益", "score": 950}, {"name": "GMAT Focus", "status": "planned",
                                                "expected": "2027-01"},
               {"name": "某某檢定"}, {"name": "某某檢定 II", "category": "subject_test"}],
        experience=[{"category": "research_experience", "description": "大學專題", "months": 12}],
        other=[{"category": "design_portfolio", "evidence": "作品集 12 件"}],
    ).record  # fmt: skip
    assert "托福（100）" in rec.completed["english_proficiency"]
    assert "IELTS" in rec.completed["english_proficiency"]
    assert set(rec.unmapped) == {"多益", "某某檢定"}  # review + unmatched
    assert "gmat" in rec.planned and "subject_test" in rec.completed
    assert "12 個月" in rec.completed["research_experience"]
    assert rec.completed["design_portfolio"] == "作品集 12 件"


def test_validation_collects_every_error_with_its_path(cmap):
    bad = {
        "schema": "student_record_intake/0",
        "student_id": "",
        "nickname": "x",
        "target_apply_by": "next year",
        "degrees": [{"level": "associate", "field": "x", "cip": "52"}],
        "courses": [
            {"name": "微積分", "credits": 0},
            {"name": "線性代數", "status": "planned"},
            {"name": "統計學", "categories": ["statistcs"]},
            {"grade": "A"},
            {"name": "機率", "colour": "red"},
        ],
        "experience": [{"category": "gre", "description": "x"}],
    }
    with pytest.raises(RecordIntakeError) as exc:
        intake_student_record(bad, cmap)
    errs = exc.value.errors
    for frag in ("schema:", "student_id:", "nickname: unknown field", "target_apply_by:",
                 "degrees[0].level:", "degrees[0].cip:", "courses[0].credits:",
                 "courses[1].expected:", "did you mean statistics", "courses[3]: missing",
                 "courses[4]: unknown field", "experience[0].category:"):  # fmt: skip
        assert any(frag in e for e in errs), (frag, errs)


def test_planned_course_cannot_have_a_grade_and_completed_wins(cmap):
    with pytest.raises(RecordIntakeError, match="planned course has no grade"):
        _intake(cmap, courses=[{"name": "線性代數", "status": "planned", "expected": "x",
                                "grade": "A"}])  # fmt: skip
    rec = _intake(cmap, courses=[{"name": "統計學"},
                                 {"name": "統計學（二）", "status": "planned",
                                  "expected": "2027"}]).record  # fmt: skip
    assert "statistics" in rec.completed and "statistics" not in rec.planned


def test_online_only_needs_every_evidence_online(cmap):
    rec = _intake(cmap, courses=[{"name": "統計學", "online": True},
                                 {"name": "統計學（二）"},
                                 {"name": "線性代數", "online": True}]).record  # fmt: skip
    assert rec.online_only == {"linear_algebra"}


# --- into C6 -------------------------------------------------------------------------


def test_intake_record_feeds_the_existing_gap_computation():
    rec = load_student_record(EXAMPLE).record
    items = [_req("calculus_1"), _req("linear_algebra"), _req("probability"),
             _req("gre", kind="test"), _req("bachelor_degree", kind="credential")]  # fmt: skip
    gap = compute_gap("p", items, rec, load_taxonomy(TAXONOMY))
    assert {i.category for i in gap.met} == {"calculus_1", "bachelor_degree"}
    assert {i.category for i in gap.planned} == {"linear_algebra", "gre"}
    assert {i.category for i in gap.missing} == {"probability"}  # 商用統計 ≠ probability


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C6-11: online-only evidence still meets accepts_online=False",
)
def test_probe_online_only_course_does_not_meet_offline_requirement(cmap):
    rec = _intake(cmap, courses=[{"name": "Python程式設計", "online": True}]).record
    strict = _req("programming_intro", accepts_online=False)
    gap = compute_gap("p", [strict], rec, load_taxonomy(TAXONOMY))
    assert gap.met == () and gap.missing == (strict,)


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C6-12: C6 checks taken/not taken, not published minimum scores",
)
def test_probe_score_below_published_minimum_is_not_met(cmap):
    rec = _intake(cmap, exams=[{"name": "TOEFL iBT", "score": 70}]).record
    item = RequirementItem("p", "language", "english_proficiency", "required",
                           "Minimum TOEFL iBT score of 90", URL, DATE)  # fmt: skip
    gap = compute_gap("p", [item], rec, load_taxonomy(TAXONOMY))
    assert gap.met == ()


# --- D9 guard --------------------------------------------------------------------------


def test_c2_never_reads_the_intake():
    """D9: the record intake feeds C6 only. Discovery must not import or mention it,
    and the discovery entry points take no record argument."""
    import inspect

    from aisc import pipeline

    banned = ("record_intake", "intake_student_record", "load_student_record", "StudentRecord",
              "online_only", "unmapped")  # fmt: skip
    offenders = [
        f"{p.name}: {w}"
        for p in (ROOT / "src" / "aisc" / "discovery").glob("*.py")
        for w in banned
        if w in p.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
    for fn in (pipeline.interactive_discovery, pipeline.prepare_profile):
        params = inspect.signature(fn).parameters
        assert not any("record" in n or "gap" in n for n in params), (fn.__name__, list(params))
