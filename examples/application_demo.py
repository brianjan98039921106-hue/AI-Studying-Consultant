"""Future extension demo (not part of the current product scope; ROADMAP Phase 7):
core discovery -> assumed admission probabilities -> application portfolio (C4),
with the correlated-admission sensitivity (C4.2) and the plan section for the report.

Run:  python examples/application_demo.py
"""

from pathlib import Path

from aisc.application import plan_applications, render_plan_section
from aisc.catalog import load_programs
from aisc.contracts import AdmitEstimate, StudentProfile
from aisc.pipeline import interactive_discovery

ROOT = Path(__file__).resolve().parents[1]
catalog = load_programs(ROOT / "data" / "synthetic" / "programs.jsonl")
student = StudentProfile(
    student_id="demo",
    needs={"quant_intensity": 0.8, "domain_finance": 0.7, "riasec_e": 0.6,
           "programming_intensity": 0.5},
    anchors=["syn-01"],
)  # fmt: skip
known = {"syn-12", "syn-04"}
result = interactive_discovery(
    student, catalog, lambda ids: {i: ("heard" if i in known else "unknown") for i in ids},
    n_shortlist=8, k_new=5, k_known=3,
)  # fmt: skip

# No C3 estimate is used here: admission probabilities are explicit assumptions.
recs = result.discoveries + result.known_good
means = [0.35, 0.25, 0.45, 0.30, 0.50, 0.40, 0.55, 0.20]
assumed = {
    r.program.program_id: AdmitEstimate(
        r.program.program_id, mean=m, low=max(0.0, m - 0.15), high=m + 0.15, method="assumed"
    )
    for r, m in zip(recs, means, strict=False)
}
recs = [r for r in recs if r.program.program_id in assumed]
plan = plan_applications(recs, assumed, budget=3)
print(render_plan_section(plan, [r.program for r in recs]))
