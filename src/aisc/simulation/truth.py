"""Ground-truth utility used only inside the simulator.

Two definitions are provided on purpose (A-C5-5, B-C5-1). A scorer that
matches the simulator's own truth function will always look good, so every
L3 result must be reported under both and trusted only if they agree.

- "distance": 1 - ||w - x|| / sqrt(d). Sensitive to intensity: mild interest
  does not match an extreme program.
- "cosine": direction only. Matches the C2 v0 scorer's assumption.
"""

from __future__ import annotations

import math

import numpy as np

from aisc.contracts import Program

TRUTH_KINDS = ("distance", "cosine")


def true_utility(w: np.ndarray, x: np.ndarray, kind: str = "distance") -> float:
    """Utility in [0, 1] of a program with attributes x for true needs w."""
    if kind == "distance":
        return float(1.0 - np.linalg.norm(w - x) / math.sqrt(len(w)))
    if kind == "cosine":
        denom = float(np.linalg.norm(w) * np.linalg.norm(x))
        return 0.0 if denom == 0 else float(w @ x / denom)
    raise ValueError(f"Unknown truth kind {kind!r}; expected one of {TRUTH_KINDS}")


def utilities(w: np.ndarray, catalog: list[Program], kind: str = "distance") -> np.ndarray:
    return np.array([true_utility(w, p.vector, kind) for p in catalog])


def relevant_ids(
    w: np.ndarray, catalog: list[Program], kind: str = "distance", top_frac: float = 0.1
) -> set[str]:
    """Programs in the top `top_frac` of the catalog by true utility."""
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must lie in (0, 1]")
    n = max(1, math.ceil(top_frac * len(catalog)))
    order = np.argsort(-utilities(w, catalog, kind), kind="stable")
    return {catalog[i].program_id for i in order[:n]}


def best_fit_id(
    w: np.ndarray, catalog: list[Program], exclude: set[str], kind: str = "distance"
) -> str | None:
    """Highest-utility program not in `exclude` (typically the anchors)."""
    best, best_u = None, -math.inf
    for p in catalog:
        if p.program_id in exclude:
            continue
        u = true_utility(w, p.vector, kind)
        if u > best_u:
            best, best_u = p.program_id, u
    return best
