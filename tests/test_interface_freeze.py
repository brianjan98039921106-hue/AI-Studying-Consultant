"""D13: admission feasibility (C3) and application portfolios (C4) are paused, their
interface kept. These tests fix that interface so the rest of the system can change
without silently breaking the way back in, and keep the main flow independent of
the paused components."""

import dataclasses
import inspect
import re
from datetime import date
from pathlib import Path

from aisc.catalog import load_programs
from aisc.contracts import AdmitEstimate, PortfolioResult, StudentProfile
from aisc.pipeline import ApplicationPlan, interactive_discovery, plan_applications
from aisc.report import render_profile_report

ROOT = Path(__file__).resolve().parents[1]
PAUSED = re.compile(r"aisc\.(feasibility|portfolio)|from aisc import .*(feasibility|portfolio)")


def _fields(cls):
    return [f.name for f in dataclasses.fields(cls)]


def test_admit_estimate_contract_is_frozen():
    assert _fields(AdmitEstimate) == ["program_id", "mean", "low", "high", "method"]
    assert AdmitEstimate.__dataclass_params__.frozen
    assert _fields(PortfolioResult) == ["chosen", "expected_utility", "method"]
    assert _fields(ApplicationPlan) == ["program_ids", "result", "value_if_correlated",
                                        "assumed"]  # fmt: skip


def test_plan_applications_signature_is_frozen():
    sig = inspect.signature(plan_applications)
    assert [(p.name, p.default) for p in sig.parameters.values()] == [
        ("recommendations", inspect.Parameter.empty),
        ("admit", inspect.Parameter.empty),
        ("budget", inspect.Parameter.empty),
        ("cost", None),
        ("utility", None),
        ("criterion", "point"),
        ("rhos", (0.0, 0.4, 0.8)),
        ("n_sim", 20_000),
        ("rng", None),
    ]
    assert inspect.signature(render_profile_report).parameters["plan"].default is None


def test_any_admit_source_plugs_into_the_paused_planner():
    """An AdmitEstimate from any source (a future C3 model, a consultant) still
    reaches C4 and the report through the frozen interface."""
    catalog = load_programs(ROOT / "data" / "synthetic" / "programs.jsonl")
    student = StudentProfile("freeze", needs={"quant_intensity": 0.8}, anchors=["syn-01"])
    result = interactive_discovery(student, catalog, lambda ids: dict.fromkeys(ids, "unknown"),
                                   n_shortlist=6, k_new=3, k_known=2)  # fmt: skip
    recs = result.discoveries + result.known_good
    admit = {r.program.program_id: AdmitEstimate(r.program.program_id, 0.4, 0.3, 0.5,
                                                 "consultant") for r in recs}  # fmt: skip
    plan = plan_applications(recs, admit, budget=2)
    assert len(plan.chosen_ids) == 2 and not plan.assumed
    from aisc.pipeline import prepare_profile

    md = render_profile_report(student, prepare_profile(student, catalog), result, plan=plan,
                               today=date(2026, 10, 29))  # fmt: skip
    assert "建議申請" in md


def test_main_flow_does_not_import_paused_components():
    """C0, C1, C2, C6 and C5 must not depend on C3 or C4 (D5, D13)."""
    offenders = []
    for pkg in ("catalog", "profiling", "discovery", "requirements", "simulation"):
        for f in (ROOT / "src" / "aisc" / pkg).rglob("*.py"):
            if PAUSED.search(f.read_text(encoding="utf-8")):
                offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, offenders
