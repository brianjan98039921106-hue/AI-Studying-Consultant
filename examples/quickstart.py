"""End-to-end demo on synthetic data: profile -> discover -> portfolio.

Run:  python examples/quickstart.py
"""

from pathlib import Path

from aisc.discovery import recommend
from aisc.portfolio import dp_varied_cost
from aisc.profiling import blend_profile
from aisc.schema import StudentProfile, load_programs

ROOT = Path(__file__).resolve().parents[1]
catalog = load_programs(ROOT / "data" / "synthetic" / "programs.jsonl")
by_id = {p.program_id: p for p in catalog}

# A student who only knows "MS Finance" but says they like math and some coding.
student = StudentProfile(
    student_id="demo",
    needs={"quant": 0.8, "finance": 0.7, "business": 0.6, "programming": 0.5},
    anchors=["syn-01"],
)
profile = blend_profile(student, by_id)

print("== Discovery (relevance x novelty, MMR-diversified) ==")
recs = recommend(profile, catalog, k=6, novelty_weight=0.4)
for r in recs:
    p = r.program
    print(f"{p.degree_name:<42} CIP {p.cip}  fit={r.relevance:.2f}  novelty={r.novelty:.2f}")

# Placeholder admission probabilities and utilities: in the real pipeline
# p comes from L3 and u from the student's own weighting.
p_admit = [0.35, 0.25, 0.45, 0.55, 0.30, 0.50][: len(recs)]
utility = [round(100 * r.relevance) for r in recs]
fee = [1] * len(recs)  # one unit per application

chosen, value = dp_varied_cost(p_admit, utility, fee, budget=3)
print("\n== Portfolio for 3 applications (illustrative probabilities) ==")
for j in chosen:
    print(f"  apply: {recs[j].program.degree_name}")
print(f"  expected utility of best admission: {value:.1f}")
