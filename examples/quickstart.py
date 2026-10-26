"""End-to-end demo on synthetic data:
stated needs -> shortlist -> familiarity check -> two-section discovery -> portfolio.

Run:  python examples/quickstart.py
"""

from pathlib import Path

from aisc.catalog import load_programs
from aisc.contracts import AdmitEstimate, StudentProfile
from aisc.discovery import recommend
from aisc.pipeline import interactive_discovery, prepare_profile
from aisc.portfolio import dp_varied_cost

ROOT = Path(__file__).resolve().parents[1]
catalog = load_programs(ROOT / "data" / "synthetic" / "programs.jsonl")
by_id = {p.program_id: p for p in catalog}

# A student who only has "MS Finance" in mind but likes maths and some coding.
# Only 4 of 15 dimensions are answered: the pipeline completes the rest with C1.2 v1
# (anchors as a prior for unanswered dimensions only; answered ones stay as stated).
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

# Scripted answers to the familiarity check: in production this is the UI.
ALREADY_KNOWS = {"syn-12", "syn-04"}  # MBA, MS Business Analytics


def ask(program_ids: list[str]) -> dict[str, str]:
    print(f"Familiarity check — student is asked about {len(program_ids)} programs")
    return {pid: ("heard" if pid in ALREADY_KNOWS else "unknown") for pid in program_ids}


def show(title, recs):
    print(f"\n== {title} ==")
    for r in recs:
        p = r.program
        print(f"  {p.degree_name:<42} CIP {p.cip}  fit={r.relevance:.2f}")


result = interactive_discovery(student, catalog, ask, n_shortlist=8, k_new=5, k_known=3)
show("你可能沒想過 (discoveries)", result.discoveries)
show("你已經知道、而且確實適合 (known & good fit)", result.known_good)

# Reference only: C2 v0 (additive novelty + MMR) was rejected at L3 (F1).
show(
    "Reference: C2 v0, REJECTED at L3",
    recommend(prepare_profile(student, catalog), catalog, k=5, novelty_weight=0.4),
)

# C3 does not exist yet: admission probabilities are explicit assumptions.
shortlist_for_portfolio = result.discoveries + result.known_good
assumed = [
    AdmitEstimate(r.program.program_id, mean=m, low=max(0.0, m - 0.15), high=m + 0.15,
                  method="assumed")
    for r, m in zip(shortlist_for_portfolio, [0.35, 0.25, 0.45, 0.30, 0.50, 0.40, 0.55, 0.20],
                    strict=False)
]  # fmt: skip
p_admit = [a.mean for a in assumed]
utility = [round(100 * r.relevance) for r in shortlist_for_portfolio[: len(assumed)]]
chosen, value = dp_varied_cost(p_admit, utility, [1] * len(assumed), budget=3)
print("\n== Portfolio for 3 applications (assumed admission probabilities) ==")
for j in chosen:
    print(f"  apply: {shortlist_for_portfolio[j].program.degree_name}  (P(admit) assumed)")
print(f"  expected utility of best admission: {value:.1f}")
