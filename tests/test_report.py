"""W-50 v0 profile report (system layer)."""

import copy
from datetime import date

import pytest

from aisc.contracts import StudentProfile
from aisc.pipeline import interactive_discovery, prepare_profile
from aisc.report import render_profile_report

NEEDS = {"quant_intensity": 0.8, "domain_finance": 0.9, "riasec_e": 0.6}


def _run(catalog):
    profile = StudentProfile("demo", NEEDS)
    result = interactive_discovery(
        profile, catalog, lambda ids: {i: ("heard" if i == ids[0] else "unknown") for i in ids},
        k_new=4, k_known=2,
    )  # fmt: skip
    return profile, prepare_profile(profile, catalog), result


def test_report_labels_synthetic_data_and_completion(catalog):
    profile, completed, result = _run(catalog)
    before = copy.deepcopy(result)
    md = render_profile_report(profile, completed, result, today=date(2026, 10, 28))
    assert result == before  # formatting only
    assert "合成的示範資料" in md and "補全" in md
    assert "2026-10-28" in md
    for r in result.discoveries:
        assert r.program.degree_name in md
    assert md.count("| 興趣 |") == 6  # every RIASEC dimension listed
    assert "## 附錄：申請組合" not in md  # core report only (D13)
    assert "不估計錄取機率" in md


def test_report_has_no_plan_parameter(catalog):
    profile, completed, result = _run(catalog)
    with pytest.raises(TypeError):
        render_profile_report(profile, completed, result, plan=object())


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
