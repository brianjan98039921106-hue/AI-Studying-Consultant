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
from collections.abc import Mapping

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


Hidden = Mapping[str, np.ndarray]


def _full(p: Program, hidden: Hidden | None) -> np.ndarray:
    """Program vector, extended with its hidden domains when the simulator has them (W-67)."""
    return p.vector if hidden is None else np.concatenate([p.vector, hidden[p.program_id]])


def _need(w: np.ndarray, w_hidden: np.ndarray | None, hidden: Hidden | None) -> np.ndarray:
    if (hidden is None) != (w_hidden is None):
        raise ValueError("pass both `hidden` (programs) and `w_hidden` (student), or neither")
    return w if w_hidden is None else np.concatenate([w, w_hidden])


def utilities(
    w: np.ndarray,
    catalog: list[Program],
    kind: str = "distance",
    hidden: Hidden | None = None,
    w_hidden: np.ndarray | None = None,
) -> np.ndarray:
    """True utility of every program. With `hidden`/`w_hidden`, the truth also covers
    the hidden domains that no component can see (W-67, A-C5-10)."""
    full_w = _need(w, w_hidden, hidden)
    return np.array([true_utility(full_w, _full(p, hidden), kind) for p in catalog])


def relevant_ids(
    w: np.ndarray,
    catalog: list[Program],
    kind: str = "distance",
    top_frac: float = 0.1,
    hidden: Hidden | None = None,
    w_hidden: np.ndarray | None = None,
) -> set[str]:
    """Programs in the top `top_frac` of the catalog by true utility."""
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must lie in (0, 1]")
    n = max(1, math.ceil(top_frac * len(catalog)))
    order = np.argsort(-utilities(w, catalog, kind, hidden, w_hidden), kind="stable")
    return {catalog[i].program_id for i in order[:n]}


def best_fit_id(
    w: np.ndarray,
    catalog: list[Program],
    exclude: set[str],
    kind: str = "distance",
    hidden: Hidden | None = None,
    w_hidden: np.ndarray | None = None,
) -> str | None:
    """Highest-utility program not in `exclude` (typically the anchors)."""
    full_w = _need(w, w_hidden, hidden)
    best, best_u = None, -math.inf
    for p in catalog:
        if p.program_id in exclude:
            continue
        u = true_utility(full_w, _full(p, hidden), kind)
        if u > best_u:
            best, best_u = p.program_id, u
    return best
