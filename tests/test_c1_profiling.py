"""C1 profiling: verification (L2) and blind-spot probes (L4)."""

import pytest

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.profiling import (
    anchor_prior,
    blend_profile,
    blend_profile_v1,
    complete_neutral,
    needs_from_anchors,
)


def test_blend_moves_toward_anchors_and_shrinks_uncertainty(by_id):
    student = StudentProfile(student_id="t", needs={"domain_design": 0.0}, anchors=["syn-08"])
    blended = blend_profile(student, by_id)
    implied = needs_from_anchors([by_id["syn-08"]])
    assert 0.0 < blended.needs["domain_design"] <= implied[NEED_KEYS.index("domain_design")]
    assert all(blended.uncertainty[k] < 0.5 for k in NEED_KEYS)


def test_no_anchors_returns_profile_unchanged(by_id):
    student = StudentProfile(student_id="t", needs={"quant_intensity": 0.4})
    assert blend_profile(student, by_id) is student


def test_more_anchors_means_lower_uncertainty(by_id):
    one = blend_profile(StudentProfile("a", {}, anchors=["syn-01"]), by_id)
    three = blend_profile(StudentProfile("b", {}, anchors=["syn-01", "syn-02", "syn-04"]), by_id)
    assert all(three.uncertainty[k] < one.uncertainty[k] for k in NEED_KEYS)


_B_C1_2_V0 = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C1-2 (v0 reference only): prestige-driven anchors override stated needs",
)


@pytest.mark.probe
@pytest.mark.parametrize(
    "version",
    [pytest.param("v0", marks=_B_C1_2_V0, id="v0-reference"), pytest.param("v1", id="v1")],
)
def test_probe_prestige_anchors_do_not_override_strong_stated_need(by_id, version):
    """A design-oriented student lists famous finance/MBA programs because of
    brand. Desired behaviour: design (domain or artistic interest) stays the top
    need. v0 fails this; v1 must not."""
    student = StudentProfile(
        student_id="probe",
        needs={"domain_design": 0.9, "riasec_a": 0.8, "domain_tech_product": 0.7},
        uncertainty={"domain_design": 0.3, "riasec_a": 0.3, "domain_tech_product": 0.3},
        anchors=["syn-01", "syn-12", "syn-04"],  # MS Finance, MBA, MS Business Analytics
    )
    blended = (blend_profile if version == "v0" else blend_profile_v1)(student, by_id)
    assert max(blended.needs, key=blended.needs.get) in {"domain_design", "riasec_a"}


# --- v1 (W-23) -------------------------------------------------------------------------


def test_complete_neutral_fills_only_unanswered():
    out = complete_neutral(StudentProfile("s", {"quant_intensity": 0.9}, {"quant_intensity": 0.1}))
    assert out.needs["quant_intensity"] == 0.9 and out.uncertainty["quant_intensity"] == 0.1
    assert out.needs["domain_design"] == 0.5 and set(out.needs) == set(NEED_KEYS)


def test_anchor_prior_sd_shrinks_with_more_anchors(by_id):
    cat = list(by_id.values())
    _, one = anchor_prior([by_id["syn-07"]], cat, prestige_beta=0)
    _, four = anchor_prior([by_id[i] for i in ("syn-07", "syn-09", "syn-10", "syn-11")], cat,
                           prestige_beta=0)  # fmt: skip
    assert four == pytest.approx(one / 2)


def test_anchor_prior_widens_for_prestigious_anchors(by_id):
    cat = list(by_id.values())
    _, famous = anchor_prior([by_id["syn-12"], by_id["syn-01"]], cat)  # prestige .9, .75
    _, modest = anchor_prior([by_id["syn-09"], by_id["syn-11"]], cat)  # prestige .3, .35
    _, famous_no_beta = anchor_prior([by_id["syn-12"], by_id["syn-01"]], cat, prestige_beta=0)
    assert famous > famous_no_beta == pytest.approx(modest)


def test_v1_leaves_answered_dimensions_alone_by_default(by_id):
    student = StudentProfile("s", {"domain_design": 0.9}, {"domain_design": 0.2},
                             anchors=["syn-01"])  # fmt: skip
    out = blend_profile_v1(student, by_id)
    assert out.needs["domain_design"] == pytest.approx(0.9)
    assert out.uncertainty["domain_design"] == pytest.approx(0.2)


def test_v1_moves_unanswered_dimensions_toward_anchors(by_id):
    student = StudentProfile("s", {"domain_design": 0.9}, anchors=["syn-02"])  # FinMath
    out = blend_profile_v1(student, by_id, prestige_beta=0)
    fin = by_id["syn-02"].attributes["domain_finance"]  # 0.9
    assert 0.5 < out.needs["domain_finance"] < fin
    assert out.uncertainty["domain_finance"] < 0.3


def test_v1_without_anchors_equals_neutral(by_id):
    student = StudentProfile("s", {"quant_intensity": 0.7})
    assert blend_profile_v1(student, by_id).needs == complete_neutral(student).needs


@pytest.mark.parametrize(
    "answered_scale,prestige_beta,fixed",
    [
        (1.0, 0.0, False),  # v0-like weighting, no prestige correction: blind spot remains
        (1.0, 3.0, True),  # prestige widening alone fixes it
        (float("inf"), 0.0, True),  # role change alone fixes it
        (float("inf"), 3.0, True),  # v1 default: both
    ],
)
def test_b_c1_2_each_v1_mechanism_fixes_it_independently(
    by_id, answered_scale, prestige_beta, fixed
):
    student = StudentProfile(
        student_id="ablation",
        needs={"domain_design": 0.9, "riasec_a": 0.8, "domain_tech_product": 0.7},
        uncertainty={"domain_design": 0.3, "riasec_a": 0.3, "domain_tech_product": 0.3},
        anchors=["syn-01", "syn-12", "syn-04"],
    )
    out = blend_profile_v1(
        student, by_id, answered_scale=answered_scale, prestige_beta=prestige_beta
    )
    top = max(out.needs, key=out.needs.get)
    assert (top in {"domain_design", "riasec_a"}) is fixed


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C1-32: an anchor's unmeasured (placeholder) dimensions shrink the student's prior SD",
)
def test_probe_unmeasured_anchor_dims_carry_no_evidence():
    from aisc.contracts import Program
    from aisc.dimensions import DIMENSION_SETS, dims_keys
    from aisc.profiling import blend_profile_v1

    keys = dims_keys("v2")
    # the layers a catalogue without curriculum evidence cannot measure (A-C0-15)
    gap = tuple(d.key for d in DIMENSION_SETS["v2"] if d.layer in ("intensity", "orientation"))
    attrs = {k: 0.3 for k in keys} | {k: 0.5 for k in gap}
    anchor = Program("a", "S", "MS X", "11.0701", attrs, dims="v2", unmeasured=gap, synthetic=True)
    other = Program("b", "S", "MS Y", "52.0201", attrs, dims="v2", synthetic=True)
    catalog = {"a": anchor, "b": other}
    needs = {"riasec_i": 0.8}
    with_anchor = blend_profile_v1(StudentProfile("s", needs, anchors=["a"], dims="v2"), catalog)
    without = blend_profile_v1(StudentProfile("s", needs, dims="v2"), catalog)
    for k in gap:
        assert with_anchor.uncertainty[k] == pytest.approx(without.uncertainty[k])
        assert with_anchor.needs[k] == pytest.approx(without.needs[k])
