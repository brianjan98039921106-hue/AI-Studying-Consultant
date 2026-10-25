"""End-to-end demo on synthetic data: profile -> discover -> portfolio.

Run:  python examples/quickstart.py
"""

from pathlib import Path

from aisc.catalog import load_programs
from aisc.contracts import StudentProfile
from aisc.discovery import recommend
from aisc.portfolio import dp_varied_cost
from aisc.profiling import blend_profile

ROOT = Path(__file__).resolve().parents[1]
catalog = load_programs(ROOT / "data" / "synthetic" / "programs.jsonl")
by_id = {p.program_id: p for p in catalog}

# A student who only knows "MS Finance" but says they like math and some coding.
student = StudentProfile(
    student_id="demo",
    needs={
        "quant_intensity": 0.8,
        "domain_finance": 0.7,
        "riasec_e": 0.6,
        "programming_intensity": 0.5,
    },
    anchors=["syn-01"],
)
profile = blend_profile(student, by_id)


def show(title: str, recs) -> None:
    print(f"== {title} ==")
    for r in recs:
        p = r.program
        print(f"{p.degree_name:<42} CIP {p.cip}  fit={r.relevance:.2f}  novelty={r.novelty:.2f}")
    print()


# C2 v0 default (additive novelty + MMR). Rejected at L3 (results/synthetic-l3-v0-findings.md,
# F1): it lowers both relevance and novel-relevant rate. Kept as the reference until W-28.
show(
    "C2 v0: relevance x novelty, MMR (REJECTED at L3, reference only)",
    recommend(profile, catalog, k=6, novelty_weight=0.4),
)
recs = recommend(profile, catalog, k=6, novelty_weight=0.0, diversity_weight=0.0)
show("Fit-only (best performer at L3 so far)", recs)

# Placeholder admission probabilities: C3 does not exist yet, so these are
# assumptions (AdmitEstimate(method="assumed") in the real pipeline).
p_admit = [0.35, 0.25, 0.45, 0.55, 0.30, 0.50][: len(recs)]
utility = [round(100 * r.relevance) for r in recs]
fee = [1] * len(recs)  # one unit per application

chosen, value = dp_varied_cost(p_admit, utility, fee, budget=3)
print("== Portfolio for 3 applications (fit-only list, assumed probabilities) ==")
for j in chosen:
    print(f"  apply: {recs[j].program.degree_name}")
print(f"  expected utility of best admission: {value:.1f}")
