"""C2 discovery: verification (L2), behaviour (L3-lite) and probes (L4)."""

import numpy as np
import pytest

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.discovery import (
    contributions,
    discover,
    label_similarity,
    novelty,
    recommend,
    relevance,
)
from aisc.profiling import blend_profile, complete_neutral

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
        needs={
            "quant_intensity": 0.8,
            "domain_finance": 0.9,
            "riasec_e": 0.6,
            "programming_intensity": 0.5,
        },
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
    student = complete_neutral(
        StudentProfile(
            student_id="t3",
            needs={"quant_intensity": 0.8},
            hard_constraints={"spring_intake": True},
        )
    )
    recs = recommend(student, catalog, k=12)
    assert recs and all(r.program.spring_intake is not False for r in recs)


def test_min_relevance_floor(catalog):
    student = complete_neutral(StudentProfile(student_id="t4", needs={"domain_design": 1.0}))
    recs = recommend(student, catalog, k=12, min_relevance=0.5)
    assert all(r.relevance >= 0.5 for r in recs)


def test_invalid_weights_rejected(catalog):
    with pytest.raises(ValueError):
        recommend(
            complete_neutral(StudentProfile("x", {"quant_intensity": 1.0})),
            catalog,
            novelty_weight=1.5,
        )


# --- L4 blind-spot probes ---------------------------------------------------------


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C2-3 (v0 reference only): cosine ignores intensity mismatch",
)
def test_probe_cosine_ignores_intensity():
    """Mild interest in math should not be a perfect match for the most
    math-intensive program. Cosine (v0 `relevance`, kept as reference) scores it 1.0.
    The absolute threshold only makes sense on the cosine scale; the scale-free
    restatement below is what W-26 fixed."""
    student = np.zeros(len(NEED_KEYS))
    program = np.zeros(len(NEED_KEYS))
    student[NEED_KEYS.index("quant_intensity")] = 0.2
    program[NEED_KEYS.index("quant_intensity")] = 1.0
    assert relevance(student, program) < 0.9


_B_C2_2_V0 = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C2-2 (v0 reference only): novelty weight pushes out the best-fit program",
)


@pytest.mark.probe
@pytest.mark.parametrize(
    "version,familiar",
    [
        pytest.param("v0", [], marks=_B_C2_2_V0, id="v0-reference"),
        pytest.param("v1", [], id="v1-unknown"),
        pytest.param("v1", ["syn-04"], id="v1-known"),
    ],
)
def test_probe_best_fit_survives(by_id, catalog, version, familiar):
    """The single best-fit non-anchor program must remain visible (top 5).

    v0 hides it with α=0.4 because it shares the anchor's CIP family; v0 stays
    as the reference and keeps failing. v1 (sectioned) must show it — as a
    discovery if unknown, in the known-good section if the student knows it.
    """
    student = StudentProfile(
        student_id="probe",
        needs=dict(by_id["syn-04"].attributes),  # wants exactly MS Business Analytics
        anchors=["syn-01"],  # knows MS Finance (same CIP family 52)
        familiar=familiar,
    )
    if version == "v0":
        shown = [r.program.program_id for r in recommend(student, catalog, k=5, novelty_weight=0.4)]
    else:
        shown = discover(student, catalog, k_new=5, k_known=5).all_ids
    assert "syn-04" in shown


# --- B-C2-3 restated scale-free (W-26) ------------------------------------------------
#
# The original probe above uses an absolute threshold (< 0.9) that only makes sense on
# the cosine scale. To compare scorers on different scales, the same blind spot is
# restated as a ranking property: when two programs point in the same direction as the
# student and differ only in magnitude, the one whose magnitude matches the student
# must rank strictly first. Cosine ties in exactly these cases (verified while writing
# this probe: with a non-zero background on other dimensions cosine already prefers the
# matched program, so those cases would not test the blind spot).


def _single_dim_case(student_level, other_level):
    q = NEED_KEYS.index("quant_intensity")
    w, a, b = (np.zeros(len(NEED_KEYS)) for _ in range(3))
    w[q], a[q], b[q] = student_level, student_level, other_level
    return w, np.vstack([a, b])


def _parallel_case(student_level, other_level):
    """Student mild on every dimension; programs identical in direction."""
    w = np.full(len(NEED_KEYS), student_level)
    return w, np.vstack([w.copy(), np.full(len(NEED_KEYS), other_level)])


@pytest.mark.probe
@pytest.mark.parametrize(
    "case",
    [
        _single_dim_case(0.2, 1.0),
        _single_dim_case(1.0, 0.2),
        _single_dim_case(0.5, 1.0),
        _parallel_case(0.3, 0.9),
    ],
    ids=["over-intense", "under-intense", "mild-vs-max", "mild-everywhere"],
)
def test_probe_intensity_matched_program_ranks_first(case):
    from aisc.discovery import DEFAULT_SCORER

    w, X = case
    scores = DEFAULT_SCORER.score_matrix(w, X)
    assert scores[0] > scores[1]


def test_default_scorer_preference_is_weak_on_the_absolute_scale():
    """Documents, rather than hides, how small the W-26 correction is: on the
    original probe the default scorer gives 0.989 instead of cosine's 1.0, so it
    would still fail the original absolute threshold (< 0.9)."""
    from aisc.discovery import DEFAULT_SCORER

    w, X = _single_dim_case(0.2, 1.0)
    assert DEFAULT_SCORER.score(w, X[1]) == pytest.approx(0.75 + 0.25 * (1 - 0.64 / 15))
    assert DEFAULT_SCORER.score(w, X[1]) > 0.9
