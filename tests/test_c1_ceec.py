"""C1.1 CEEC prior (L2)."""

import math

import pytest

from aisc.contracts import StudentProfile
from aisc.profiling import apply_prior, riasec_prior_from_ceec

SCORES = {"R": 20, "I": 80, "A": 33, "S": 66, "E": 50, "C": 99}


def test_scores_scale_linearly_to_unit_interval():
    prior = riasec_prior_from_ceec(SCORES, years_since_test=0)
    assert prior.needs["riasec_i"] == pytest.approx(80 / 99)
    assert prior.needs["riasec_c"] == pytest.approx(1.0)
    # Manual thresholds: >= 66 "likes", <= 33 "dislikes"
    assert prior.needs["riasec_s"] == pytest.approx(66 / 99)


def test_sigma_grows_with_years_and_is_capped():
    s = [riasec_prior_from_ceec(SCORES, y).sigma["riasec_r"] for y in (0, 2, 4, 20)]
    assert s[0] == pytest.approx(0.15)
    assert s[1] == pytest.approx(0.25)
    assert s[0] < s[1] < s[2] <= s[3] == pytest.approx(0.5)


def test_unclear_code_inflates_sigma():
    clear = riasec_prior_from_ceec(SCORES, 2, differentiation=5).sigma["riasec_r"]
    unclear = riasec_prior_from_ceec(SCORES, 2, differentiation=2).sigma["riasec_r"]
    assert unclear == pytest.approx(clear * 1.5)


@pytest.mark.parametrize(
    "bad",
    [
        {k: v for k, v in SCORES.items() if k != "C"},
        {**SCORES, "X": 1},
        {**SCORES, "R": 120},
    ],
)
def test_invalid_scores_rejected(bad):
    with pytest.raises(ValueError):
        riasec_prior_from_ceec(bad, 1)


def test_negative_years_rejected():
    with pytest.raises(ValueError):
        riasec_prior_from_ceec(SCORES, -1)


def test_apply_prior_fills_unelicited_dimensions():
    prior = riasec_prior_from_ceec(SCORES, 4)
    out = apply_prior(StudentProfile("s", {"quant_intensity": 0.7}), prior)
    assert out.needs["riasec_i"] == pytest.approx(80 / 99)
    assert out.uncertainty["riasec_i"] == pytest.approx(0.35)
    assert out.needs["quant_intensity"] == 0.7  # non-RIASEC untouched


def test_apply_prior_precision_weights_elicited_dimensions():
    prior = riasec_prior_from_ceec(SCORES, 0)  # sd 0.15 -> precision 44.4
    profile = StudentProfile("s", {"riasec_i": 0.2}, uncertainty={"riasec_i": 0.15})
    out = apply_prior(profile, prior)
    assert out.needs["riasec_i"] == pytest.approx((80 / 99 + 0.2) / 2)  # equal precision
    assert out.uncertainty["riasec_i"] == pytest.approx(0.15 / math.sqrt(2))


def test_old_prior_yields_to_confident_new_evidence():
    """B-C1-7: a 6-year-old code must not drag a confident current answer far."""
    prior = riasec_prior_from_ceec(SCORES, 6)  # sd 0.45
    profile = StudentProfile("s", {"riasec_i": 0.2}, uncertainty={"riasec_i": 0.1})
    out = apply_prior(profile, prior)
    assert abs(out.needs["riasec_i"] - 0.2) < 0.06
