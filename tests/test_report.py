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
