"""C3.3 calibration checks for admission predictions on held-out outcomes
(V-C3-2 ... V-C3-4; THEORY "校準評估工具").

All functions take predicted probabilities and observed 0/1 outcomes. They only
measure; nothing here changes a model.
    brier, brier_skill          V-C3-2: better than the constant base-rate forecast?
    calibration_slope           Cox (1958): logistic regression of y on logit(p);
                                ideal intercept 0, slope 1 (slope < 1: overconfident)
    reliability_table           binned mean prediction vs observed rate
    by_group                    any of the above per group (V-C3-4, e.g. Taiwanese
                                vs international applicants)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _check(pred: Sequence[float], y: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p, o = np.asarray(pred, float), np.asarray(y, float)
    if p.shape != o.shape or p.ndim != 1 or not len(p):
        raise ValueError("pred and y must be non-empty 1-d sequences of equal length")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("predictions must lie in [0, 1]")
    if not np.all((o == 0) | (o == 1)):
        raise ValueError("outcomes must be 0 or 1")
    return p, o


def brier(pred: Sequence[float], y: Sequence[int]) -> float:
    p, o = _check(pred, y)
    return float(np.mean((p - o) ** 2))


def brier_skill(pred: Sequence[float], y: Sequence[int], base_rate: float | None = None) -> float:
    """1 - Brier / Brier(constant base rate); > 0 means better than the constant.
    The base rate defaults to the observed mean (pass an official rate otherwise)."""
    p, o = _check(pred, y)
    r = float(o.mean()) if base_rate is None else float(base_rate)
    ref = float(np.mean((r - o) ** 2))
    if ref == 0:
        raise ValueError("all outcomes equal the base rate; skill is undefined")
    return 1 - float(np.mean((p - o) ** 2)) / ref


def calibration_slope(pred: Sequence[float], y: Sequence[int], eps: float = 1e-6):
    """(intercept, slope) of logit P(y = 1) = a + b * logit(pred), by Newton's method
    (with a tiny ridge for stability)."""
    p, o = _check(pred, y)
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(100):
        q = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (q - o) + 1e-8 * beta
        H = X.T @ (X * (q * (1 - q))[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def reliability_table(pred: Sequence[float], y: Sequence[int], bins: int = 10):
    """[(bin_low, bin_high, n, mean_pred, observed_rate)] for non-empty equal-width bins."""
    p, o = _check(pred, y)
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((float(edges[b]), float(edges[b + 1]), int(m.sum()),
                        float(p[m].mean()), float(o[m].mean())))  # fmt: skip
    return out


def by_group(
    metric: Callable, pred: Sequence[float], y: Sequence[int], groups: Sequence[str]
) -> dict[str, object]:
    """Apply `metric(pred, y)` within each group (V-C3-4)."""
    p, o = _check(pred, y)
    g = np.asarray(groups, dtype=object)
    if g.shape != p.shape:
        raise ValueError("groups must match pred")
    return {k: metric(p[g == k], o[g == k]) for k in sorted(set(groups))}
