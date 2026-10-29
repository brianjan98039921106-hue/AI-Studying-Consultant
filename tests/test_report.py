"""W-50 v0 profile report (system layer)."""

import copy
from datetime import date

import pytest

from aisc.contracts import AdmitEstimate, StudentProfile
from aisc.pipeline import interactive_discovery, plan_applications, prepare_profile
from aisc.report import render_profile_report

NEEDS = {"quant_intensity": 0.8, "domain_finance": 0.9, "riasec_e": 0.6}


def _run(catalog):
    profile = StudentProfile("demo", NEEDS)
    result = interactive_discovery(
        profile, catalog, lambda ids: {i: ("heard" if i == ids[0] else "unknown") for i in ids},
        k_new=4, k_known=2,
    )  # fmt: skip
    return profile, prepare_profile(profile, catalog), result


def test_report_labels_synthetic_data_completion_and_assumptions(catalog):
    profile, completed, result = _run(catalog)
    admit = {r.program.program_id: AdmitEstimate(r.program.program_id, 0.3, 0.2, 0.4, "assumed")
             for r in result.discoveries}  # fmt: skip
    plan = plan_applications(result.discoveries, admit, budget=2)
    before = copy.deepcopy(result)
    md = render_profile_report(profile, completed, result, plan=plan, today=date(2026, 10, 28))
    assert result == before  # formatting only
    assert "合成的示範資料" in md and "假設值" in md and "補全" in md
    assert "2026-10-28" in md
    for r in result.discoveries:
        assert r.program.degree_name in md
    assert md.count("| 興趣 |") == 6  # every RIASEC dimension listed
    assert "ρ = 0.4" in md


def test_report_validates_inputs(catalog):
    profile, completed, result = _run(catalog)
    with pytest.raises(ValueError):
        render_profile_report(StudentProfile("other", NEEDS), completed, result)
    with pytest.raises(ValueError):
        render_profile_report(profile, profile, result)  # not completed


def test_report_shows_gaps_without_changing_order(catalog):
    from aisc.contracts import RequirementGap, RequirementItem

    profile, completed, result = _run(catalog)
    pid = result.discoveries[0].program.program_id

    def item(level, cat="calculus"):
        return RequirementItem(
            pid, "course", cat, level, "quote", "https://example.edu", "2026-10-28"
        )

    gap = RequirementGap(
        pid,
        missing=(item("required"), item("recommended")),
        planned=(item("required"),),
        unknown=(item("required", None),),
        none_published=True,
        before_target=(item("required"),),
    )
    gaps = {pid: gap}
    md = render_profile_report(profile, completed, result, gaps=gaps)
    assert "缺必修 1" in md and "缺建議 1" in md and "規劃中 1" in md
    assert "無法判讀 1" in md and "官網未列條件" in md and "申請前須完成 1" in md
    assert "不會**影響上面的排序" in md
    names = [r.program.degree_name for r in result.discoveries]
    order = [md.index(f"| {n} |") for n in names]
    assert order == sorted(order)  # same order as the discoveries
    other = result.discoveries[1].program.program_id
    md2 = render_profile_report(profile, completed, result,
                                gaps={other: RequirementGap(other)})  # fmt: skip
    assert "條件皆已符合" in md2


def test_report_gap_cards_from_intake_record(catalog):
    """W-61: intake -> C6 gaps -> gap cards with the student's evidence; the order of
    recommendations is the same with or without the record (D9)."""
    from pathlib import Path

    from aisc.contracts import RequirementItem
    from aisc.profiling import load_student_record
    from aisc.requirements import compute_gaps

    root = Path(__file__).resolve().parents[1]
    rec = load_student_record(root / "data" / "synthetic" / "student_record_example.json").record
    profile, completed, result = _run(catalog)
    profile = StudentProfile(rec.student_id, NEEDS)
    completed = prepare_profile(profile, catalog)
    ids = [r.program.program_id for r in (*result.discoveries, *result.known_good)]
    src = ("https://example.edu/fixture", "2026-10-29")
    items = [
        RequirementItem(ids[0], "course", "calculus_1", "required", "One term of calculus", *src),
        RequirementItem(ids[0], "course", "probability", "required", "Probability theory", *src),
        RequirementItem(ids[0], "course", "programming_intro", "recommended", "Python", *src),
        RequirementItem(ids[0], "test", "gre", "required", "GRE General Test", *src),
    ]
    gaps = compute_gaps(ids, items, rec)
    plain = render_profile_report(profile, completed, result, gaps=gaps, today=date(2026, 10, 29))
    md = render_profile_report(profile, completed, result, gaps=gaps, record=rec,
                               today=date(2026, 10, 29))  # fmt: skip
    assert "## 4. 條件差距卡" in md and "「Probability theory」" in md
    assert "你的紀錄：微積分（一）" in md and "你的規劃：GRE：預計 2027-03" in md
    assert "僅線上修課" in md  # Python programming was online, page says nothing
    assert "待人工對應" in md and "| 作業研究 |" in md and "| 計算機概論 |" in md
    assert "不比對分數" in md  # B-C6-12 warning
    head = md.split("## 4.")[0]
    assert head == plain.split("## 4.")[0]  # tables and order unchanged by the record
    with pytest.raises(ValueError, match="another student"):
        render_profile_report(profile, completed, result, gaps=gaps,
                              record=type(rec)("someone-else"))  # fmt: skip
