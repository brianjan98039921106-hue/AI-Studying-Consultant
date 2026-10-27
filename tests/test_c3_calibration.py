"""C3.3 calibration checks (L2)."""

import numpy as np
import pytest

from aisc.feasibility import brier, brier_skill, by_group, calibration_slope, reliability_table


def _sample(n, a=0.0, b=1.0, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    z = np.log(p / (1 - p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(a + b * z)))).astype(int)
    return p, y


def test_calibration_slope_recovers_known_miscalibration():
    p, y = _sample(40_000)
    a, b = calibration_slope(p, y)
    assert a == pytest.approx(0.0, abs=0.05) and b == pytest.approx(1.0, abs=0.05)
    p, y = _sample(40_000, a=0.3, b=0.6, seed=1)
    a, b = calibration_slope(p, y)
    assert a == pytest.approx(0.3, abs=0.05) and b == pytest.approx(0.6, abs=0.05)


def test_brier_and_skill():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    p, y = _sample(5000, seed=2)
    assert brier_skill(p, y) > 0.1  # informative and calibrated beats the constant
    assert brier_skill(np.full(len(y), y.mean()), y) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        brier_skill([0.5, 0.5], [1, 1], base_rate=1.0)


def test_reliability_table_and_groups():
    p, y = _sample(20_000, seed=3)
    rows = reliability_table(p, y, bins=5)
    assert sum(r[2] for r in rows) == len(p)
    for lo, hi, _, mp, obs in rows:
        assert lo <= mp <= hi and abs(mp - obs) < 0.05
    g = ["tw" if i % 3 == 0 else "intl" for i in range(len(p))]
    res = by_group(brier, p, y, g)
    assert set(res) == {"tw", "intl"}


def test_validation():
    with pytest.raises(ValueError):
        brier([0.5], [1, 0])
    with pytest.raises(ValueError):
        brier([1.2], [1])
    with pytest.raises(ValueError):
        brier([0.5], [2])
    with pytest.raises(ValueError):
        reliability_table([0.5], [1], bins=0)
    with pytest.raises(ValueError):
        by_group(brier, [0.5, 0.4], [1, 0], ["a"])
