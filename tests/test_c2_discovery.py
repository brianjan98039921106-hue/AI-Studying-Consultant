"""C2 discovery: verification (L2), behaviour (L3-lite) and probes (L4)."""

import numpy as np
import pytest

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import contributions, label_similarity, novelty, recommend, relevance
from aisc.profiling import blend_profile

# --- C2.1 / C2.2 / C2.4 units -------------------------------------------------


def test_relevance_is_cosine():
    v = np.ones(len(NEED_KEYS))
    assert relevance(v, v) == pytest.approx(1.0)
    assert relevance(v, np.zeros(len(NEED_KEYS))) == 0.0


def test_label_similarity_levels(by_id):
    assert label_similarity(by_id["syn-03"], by_id["syn-11"]) == 1.0  # both 14.35
    assert label_similarity(by_id["syn-01"], by_id["syn-04"]) == 0.5  # both 52.xx
    assert label_similarity(by_id["syn-01"], by_id["syn-05"]) == 0.0


def test_novelty_cases(by_id):
    assert novelty(by_id["syn-01"], []) == 1.0
    assert novelty(by_id["syn-05"], [by_id["syn-01"]]) == 1.0
    # A program the student reports as familiar is never novel.
    assert novelty(by_id["syn-05"], [by_id["syn-01"]], [by_id["syn-05"]]) == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_contributions_are_faithful_decomposition(seed):
    rng = np.random.default_rng(seed)
    s, p = rng.random(len(NEED_KEYS)), rng.random(len(NEED_KEYS))
    c = contributions(s, p)
    assert sum(c.values()) == pytest.approx(1.0)
    rebuilt = sum(c[k] for k in NEED_KEYS) * float(s @ p)
    assert rebuilt == pytest.approx(float(s @ p))
    top = max(c, key=c.get)
    assert c[top] == pytest.approx(float((s * p).max() / (s @ p)))


# --- C2.3 behaviour -------------------------------------------------------------


def test_finance_student_discovers_non_business_programs(by_id, catalog):
    student = StudentProfile(
        student_id="t1",
        needs={"quant": 0.8, "finance": 0.9, "business": 0.6, "programming": 0.5},
        anchors=["syn-01"],
    )
    recs = recommend(blend_profile(student, by_id), catalog, k=5, novelty_weight=0.4)
    ids = [r.program.program_id for r in recs]
    assert "syn-01" not in ids
    assert {r.program.cip_family for r in recs[:3]} - {"52"}
    assert {"syn-02", "syn-03"} & set(ids)


def test_pure_fit_returns_exact_match(by_id, catalog):
    student = StudentProfile(student_id="t2", needs=dict(by_id["syn-04"].attributes))
    recs = recommend(student, catalog, k=1, novelty_weight=0.0, diversity_weight=0.0)
    assert recs[0].program.program_id == "syn-04"


def test_hard_constraints_filter(catalog):
    student = StudentProfile(
        student_id="t3", needs={"quant": 0.8}, hard_constraints={"spring_intake": True}
    )
    recs = recommend(student, catalog, k=12)
    assert recs and all(r.program.spring_intake is not False for r in recs)


def test_min_relevance_floor(catalog):
    student = StudentProfile(student_id="t4", needs={"design": 1.0})
    recs = recommend(student, catalog, k=12, min_relevance=0.5)
    assert all(r.relevance >= 0.5 for r in recs)


def test_invalid_weights_rejected(catalog):
    with pytest.raises(ValueError):
        recommend(StudentProfile("x", {"quant": 1.0}), catalog, novelty_weight=1.5)


# --- L4 blind-spot probes ---------------------------------------------------------


@pytest.mark.probe
@pytest.mark.xfail(strict=True, reason="B-C2-3: cosine ignores intensity mismatch")
def test_probe_cosine_ignores_intensity():
    """Mild interest in math should not be a perfect match for the most
    math-intensive program. Cosine scores it 1.0."""
    student = np.zeros(len(NEED_KEYS))
    program = np.zeros(len(NEED_KEYS))
    student[NEED_KEYS.index("quant")] = 0.2
    program[NEED_KEYS.index("quant")] = 1.0
    assert relevance(student, program) < 0.9


@pytest.mark.probe
@pytest.mark.xfail(strict=True, reason="B-C2-2: novelty weight can push out the best-fit program")
def test_probe_best_fit_survives_novelty_weighting(by_id, catalog):
    """The single best-fit non-anchor program should stay in the top 5.
    With α=0.4 it is displaced only because it shares the anchor's CIP family."""
    student = StudentProfile(
        student_id="probe",
        needs=dict(by_id["syn-04"].attributes),  # wants exactly MS Business Analytics
        anchors=["syn-01"],  # knows MS Finance (same CIP family 52)
    )
    recs = recommend(student, catalog, k=5, novelty_weight=0.4)
    assert "syn-04" in [r.program.program_id for r in recs]
