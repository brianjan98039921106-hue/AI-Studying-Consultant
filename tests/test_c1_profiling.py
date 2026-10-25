"""C1 profiling: verification (L2) and blind-spot probes (L4)."""

import pytest

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.profiling import blend_profile, needs_from_anchors


def test_blend_moves_toward_anchors_and_shrinks_uncertainty(by_id):
    student = StudentProfile(student_id="t", needs={"design": 0.0}, anchors=["syn-08"])
    blended = blend_profile(student, by_id)
    implied = needs_from_anchors([by_id["syn-08"]])
    assert 0.0 < blended.needs["design"] <= implied[NEED_KEYS.index("design")]
    assert all(blended.uncertainty[k] < 0.5 for k in NEED_KEYS)


def test_no_anchors_returns_profile_unchanged(by_id):
    student = StudentProfile(student_id="t", needs={"quant": 0.4})
    assert blend_profile(student, by_id) is student


def test_more_anchors_means_lower_uncertainty(by_id):
    one = blend_profile(StudentProfile("a", {}, anchors=["syn-01"]), by_id)
    three = blend_profile(StudentProfile("b", {}, anchors=["syn-01", "syn-02", "syn-04"]), by_id)
    assert all(three.uncertainty[k] < one.uncertainty[k] for k in NEED_KEYS)


@pytest.mark.probe
@pytest.mark.xfail(strict=True, reason="B-C1-2: anchors chosen for prestige override stated needs")
def test_probe_prestige_anchors_do_not_override_strong_stated_need(by_id):
    """A design-oriented student lists famous finance/MBA programs because of
    brand. Desired behaviour: design stays the top need. v0 fails this."""
    student = StudentProfile(
        student_id="probe",
        needs={"design": 0.9, "tech_product": 0.7},
        uncertainty={"design": 0.3, "tech_product": 0.3},
        anchors=["syn-01", "syn-12", "syn-04"],  # MS Finance, MBA, MS Business Analytics
    )
    blended = blend_profile(student, by_id)
    assert max(blended.needs, key=blended.needs.get) == "design"
