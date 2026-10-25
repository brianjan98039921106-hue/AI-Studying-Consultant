"""C5.2 Metric library. Pure functions, each covered by hand-computed tests.

Rule (CLAUDE.md): never report novel_relevant_rate without relevant_rate.
`list_quality` returns both together for that reason.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

# --- C1: profile recovery ----------------------------------------------------------


def rmse(estimate: Sequence[float], truth: Sequence[float]) -> float:
    a, b = np.asarray(estimate, float), np.asarray(truth, float)
    if a.shape != b.shape:
        raise ValueError("Shapes differ")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b (Kendall 1945), handles ties. O(n^2); fine for n <= few thousand.

    Returns nan when either sequence is constant.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("Shapes differ")
    n = len(x)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        dx = np.sign(x[i + 1 :] - x[i])
        dy = np.sign(y[i + 1 :] - y[i])
        prod = dx * dy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dy == 0) & (dx != 0)))
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float("nan") if denom == 0 else (concordant - discordant) / denom


# --- C2: list quality ---------------------------------------------------------------


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(recommended[:k]) & relevant) / len(relevant)


def relevant_rate(recommended: Sequence[str], relevant: set[str]) -> float:
    """Share of the list that is truly relevant (precision)."""
    if not recommended:
        return float("nan")
    return sum(r in relevant for r in recommended) / len(recommended)


def novel_relevant_rate(
    recommended: Sequence[str], relevant: set[str], familiar: set[str]
) -> float:
    """Share of the list that is relevant AND unknown to the student (V-C2-2 / V-C2-5)."""
    if not recommended:
        return float("nan")
    return sum(r in relevant and r not in familiar for r in recommended) / len(recommended)


def list_quality(
    recommended: Sequence[str], relevant: set[str], familiar: set[str]
) -> dict[str, float]:
    """Relevance and novelty reported together, never one without the other."""
    return {
        "relevant_rate": relevant_rate(recommended, relevant),
        "novel_relevant_rate": novel_relevant_rate(recommended, relevant, familiar),
        "unfamiliar_rate": (
            float("nan")
            if not recommended
            else sum(r not in familiar for r in recommended) / len(recommended)
        ),
    }


def best_fit_retained(recommended: Sequence[str], best_id: str | None) -> float:
    """1.0 if the true best-fit program appears in the list (V-C2-3)."""
    return float("nan") if best_id is None else float(best_id in recommended)


def catalog_coverage(lists: Iterable[Sequence[str]], catalog_size: int) -> float:
    """Share of the catalogue that appears in at least one list."""
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    return len({r for lst in lists for r in lst}) / catalog_size


def popularity_kl(
    recommended_prestige: Sequence[float],
    catalog_prestige: Sequence[float],
    bins: int = 5,
    eps: float = 1e-6,
) -> float:
    """KL(recommended || catalogue) over equal-width prestige bins (V-C2-4).

    0 means recommendations mirror the catalogue's prestige mix; large values
    mean they concentrate in some prestige band (popularity bias, or its reverse).
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    p, _ = np.histogram(np.asarray(recommended_prestige, float), bins=edges)
    q, _ = np.histogram(np.asarray(catalog_prestige, float), bins=edges)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


# --- C3: probabilities --------------------------------------------------------------


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p, o = np.asarray(probs, float), np.asarray(outcomes, float)
    if p.shape != o.shape:
        raise ValueError("Shapes differ")
    return float(np.mean((p - o) ** 2))


def interval_coverage(
    lows: Sequence[float], highs: Sequence[float], truths: Sequence[float]
) -> float:
    lo, hi, t = (np.asarray(v, float) for v in (lows, highs, truths))
    if not lo.shape == hi.shape == t.shape:
        raise ValueError("Shapes differ")
    return float(np.mean((lo <= t) & (t <= hi)))


# --- aggregation ---------------------------------------------------------------------


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error, ignoring nan."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), se
