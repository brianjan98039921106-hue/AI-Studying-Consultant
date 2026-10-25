"""Shared data contracts."""

import pytest

from aisc.contracts import AdmitEstimate, StudentProfile, to_vector
from aisc.dimensions import NEED_KEYS


def test_vector_order_follows_dimension_registry():
    vec = to_vector({NEED_KEYS[2]: 0.7})
    assert vec[2] == 0.7 and vec.sum() == pytest.approx(0.7)


@pytest.mark.parametrize("bad", [{"not_a_need": 0.5}, {NEED_KEYS[0]: 1.5}, {NEED_KEYS[0]: -0.1}])
def test_invalid_vectors_rejected(bad):
    with pytest.raises(ValueError):
        _ = StudentProfile(student_id="x", needs=bad).vector


def test_admit_estimate_requires_ordered_interval():
    AdmitEstimate("p", mean=0.3, low=0.2, high=0.5, method="assumed")
    with pytest.raises(ValueError):
        AdmitEstimate("p", mean=0.6, low=0.2, high=0.5, method="assumed")
