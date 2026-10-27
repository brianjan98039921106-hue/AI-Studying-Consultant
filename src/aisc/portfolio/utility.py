"""C4.4 v0 utility elicitation (W-45, B-C4-3; THEORY "多屬性效用", A-C4-4, A-C4-8).

    U(program) = 100 * sum_k w_k * v_k        v_k in [0, 1] (linear value functions)
    weights w from an AHP pairwise-comparison matrix: principal eigenvector, with
    consistency ratio CR = ((lambda_max - n) / (n - 1)) / RI_n (Saaty 1980);
    CR >= 0.1 means the comparisons should be revisited.

Fit (C2 relevance) is one attribute among others (tuition, location, duration, ...).
Utilities feed C4 through `pipeline.plan_applications(..., utility=...)`; nothing here
touches the recommendations (D9). Not validated with students yet (W-42).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

# Saaty's random consistency index for n = 1 .. 10
RANDOM_INDEX = (0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
CR_THRESHOLD = 0.1


def ahp_weights(matrix) -> tuple[np.ndarray, float]:
    """(weights summing to 1, consistency ratio) of a reciprocal comparison matrix
    where matrix[i][j] = how much more important attribute i is than j."""
    A = np.asarray(matrix, float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n) or n < 1:
        raise ValueError("need a square comparison matrix")
    if n > len(RANDOM_INDEX):
        raise ValueError(f"at most {len(RANDOM_INDEX)} attributes")
    if np.any(~np.isfinite(A)) or np.any(A <= 0):
        raise ValueError("comparisons must be positive finite numbers")
    if not np.allclose(A * A.T, 1.0, atol=1e-9) or not np.allclose(np.diag(A), 1.0):
        raise ValueError("the matrix must be reciprocal (a_ji = 1 / a_ij, a_ii = 1)")
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ri = RANDOM_INDEX[n - 1]
    cr = 0.0 if ri == 0 else max(0.0, (lam - n) / (n - 1)) / ri
    return w, cr


def scale_linear(x: float, worst: float, best: float) -> float:
    """Linear value function: worst -> 0, best -> 1, clipped (best < worst allowed,
    e.g. tuition where lower is better)."""
    if worst == best:
        raise ValueError("worst and best must differ")
    if not math.isfinite(x):
        raise ValueError("value must be finite")
    return float(min(1.0, max(0.0, (x - worst) / (best - worst))))


def multi_attribute_utility(
    values: Mapping[str, Mapping[str, float]], weights: Mapping[str, float]
) -> dict[str, float]:
    """Additive utility on a 0-100 scale for every programme. `values[pid][attr]` must
    already be scaled to [0, 1]; every programme needs every weighted attribute."""
    total = sum(weights.values())
    if not weights or total <= 0 or any(w < 0 or not math.isfinite(w) for w in weights.values()):
        raise ValueError("weights must be non-negative, finite and not all zero")
    out = {}
    for pid, vals in values.items():
        missing = [a for a in weights if a not in vals]
        if missing:
            raise ValueError(f"{pid}: no value for {missing}")
        if any(not 0 <= vals[a] <= 1 for a in weights):
            raise ValueError(f"{pid}: values must be scaled to [0, 1]")
        out[pid] = 100 * sum(weights[a] * vals[a] for a in weights) / total
    return out
