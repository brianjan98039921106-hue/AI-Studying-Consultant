"""C2.1 relevance scorers (W-26).

Every scorer provides
    score(w, x)          -> float
    score_matrix(w, X)   -> scores for every row of X (vectorised, must equal `score`)
    explain(w, x)        -> {dimension: contribution}, an exact additive decomposition:
                            sum(explain(w, x).values()) == score(w, x)   (B-C2-6)

Candidates (docs/THEORY.md, W-26):
    cosine            v0 reference: direction only (B-C2-3)
    layered(γ)        cosine − γ · mean over the intensity layer of (w_d − x_d)²
                      (A-C2-6: only intensity dimensions encode "how deep")
    blend(λ)          (1 − λ) · cosine + λ · (1 − mean over all dims of (w_d − x_d)²)
    sqdist            1 − mean over all dims of (w_d − x_d)²
                      — ranks exactly like the "distance" truth function, so any L3
                        result for it under that truth is circular (B-C5-1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisc.dimensions import NEED_KEYS, keys_in_layer

INTENSITY_IDX = np.array([NEED_KEYS.index(k) for k in keys_in_layer("intensity")])


def _cos_terms(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-dimension cosine terms w_d x_d / (|w||x|); rows sum to the cosine."""
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = (X * w) / norms[:, None]
    return np.where(norms[:, None] == 0, 0.0, terms)


@dataclass(frozen=True)
class Scorer:
    """Relevance = sum over dimensions of additive terms (exactly decomposable).

    term_d = cos_weight * cos_term_d
             + sq_weight * (1/D - (w_d - x_d)^2 / D)               for all dims   (blend/sqdist)
             - intensity_gamma * (w_d - x_d)^2 / |I|               for d in I     (layered)
    """

    name: str
    cos_weight: float = 1.0
    sq_weight: float = 0.0
    intensity_gamma: float = 0.0

    def terms(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        D = X.shape[1]
        sq = (X - w) ** 2
        out = self.cos_weight * _cos_terms(w, X)
        if self.sq_weight:
            out = out + self.sq_weight * (1.0 / D - sq / D)
        if self.intensity_gamma:
            pen = np.zeros_like(sq)
            pen[:, INTENSITY_IDX] = sq[:, INTENSITY_IDX] / len(INTENSITY_IDX)
            out = out - self.intensity_gamma * pen
        return out

    def score_matrix(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.terms(w, X).sum(axis=1)

    def score(self, w: np.ndarray, x: np.ndarray) -> float:
        return float(self.score_matrix(w, x[None, :])[0])

    def explain(self, w: np.ndarray, x: np.ndarray) -> dict[str, float]:
        t = self.terms(w, x[None, :])[0]
        return {k: float(v) for k, v in zip(NEED_KEYS, t, strict=True)}


COSINE = Scorer("cosine")
SQDIST = Scorer("sqdist", cos_weight=0.0, sq_weight=1.0)


def layered(gamma: float) -> Scorer:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return Scorer(f"layered(γ={gamma:g})", intensity_gamma=gamma)


def blend(lam: float) -> Scorer:
    if not 0 <= lam <= 1:
        raise ValueError("lam must lie in [0, 1]")
    return Scorer(f"blend(λ={lam:g})", cos_weight=1.0 - lam, sq_weight=lam)


# W-26: selected by the pre-registered maximin rule on the archetype generator,
# validated on the continuous generator and re-confirmed (and re-selected) on fresh
# seeds; never significantly worse than cosine in 16 conditions
# (results/synthetic-l3-c2-scorer-findings.md).
DEFAULT_SCORER: Scorer = blend(0.25)
