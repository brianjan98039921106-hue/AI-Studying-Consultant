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

from aisc.dimensions import LAYERS, dims_for_length, dims_keys, keys_in_layer


def layer_index(layer: str, n_dims: int) -> np.ndarray:
    """Column indices of a layer for vectors of length n_dims (v1 = 15, v2 = 51)."""
    dims = dims_for_length(n_dims)
    keys = dims_keys(dims)
    return np.array([keys.index(k) for k in keys_in_layer(layer, dims)])


INTENSITY_IDX = layer_index("intensity", 15)  # v1, kept for existing callers


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
            idx = layer_index("intensity", D)
            pen = np.zeros_like(sq)
            pen[:, idx] = sq[:, idx] / len(idx)
            out = out - self.intensity_gamma * pen
        return out

    def score_matrix(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.terms(w, X).sum(axis=1)

    def score(self, w: np.ndarray, x: np.ndarray) -> float:
        return float(self.score_matrix(w, x[None, :])[0])

    def explain(self, w: np.ndarray, x: np.ndarray) -> dict[str, float]:
        t = self.terms(w, x[None, :])[0]
        return {k: float(v) for k, v in zip(dims_keys(dims_for_length(len(t))), t, strict=True)}


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


@dataclass(frozen=True)
class LayerBalanced:
    """Layer-balanced relevance (W-16c, docs/dimensions-v2.md §4): the base scorer is
    applied to each layer separately and the layer scores are combined with weights
    that sum to 1. Keeps the 41-dimension domain layer of v2 from outweighing the
    interest, intensity and orientation layers. Explanations stay exact: each layer's
    terms are the base scorer's terms scaled by the layer weight.
    """

    name: str
    weights: tuple[tuple[str, float], ...]  # (layer, weight) for every layer
    base: Scorer

    def __post_init__(self) -> None:
        layers = [lay for lay, _ in self.weights]
        if sorted(layers) != sorted(LAYERS):
            raise ValueError(f"weights must cover exactly the layers {LAYERS}")
        ws = [w for _, w in self.weights]
        if any(w < 0 for w in ws) or abs(sum(ws) - 1.0) > 1e-9:
            raise ValueError("layer weights must be non-negative and sum to 1")

    def terms(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.zeros_like(X, dtype=float)
        for layer, weight in self.weights:
            idx = layer_index(layer, X.shape[1])
            out[:, idx] = weight * self.base.terms(w[idx], X[:, idx])
        return out

    def score_matrix(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.terms(w, X).sum(axis=1)

    def score(self, w: np.ndarray, x: np.ndarray) -> float:
        return float(self.score_matrix(w, x[None, :])[0])

    def explain(self, w: np.ndarray, x: np.ndarray) -> dict[str, float]:
        t = self.terms(w, x[None, :])[0]
        return {k: float(v) for k, v in zip(dims_keys(dims_for_length(len(t))), t, strict=True)}


def layer_balanced(weights: dict[str, float], base: Scorer | None = None) -> LayerBalanced:
    b = base if base is not None else blend(0.25)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    norm = tuple((lay, weights[lay] / total) for lay in LAYERS)
    label = ",".join(f"{lay[:3]}={w:.2f}" for lay, w in norm)
    return LayerBalanced(f"layered-balance({label};{b.name})", norm, b)


# W-26: selected by the pre-registered maximin rule on the archetype generator,
# validated on the continuous generator and re-confirmed (and re-selected) on fresh
# seeds; never significantly worse than cosine in 16 conditions
# (results/synthetic-l3-c2-scorer-findings.md).
DEFAULT_SCORER: Scorer = blend(0.25)

# W-16c: PROVISIONAL default for v2 profiles (equal layer weights) until the
# pre-registered L3 selection (docs/dimensions-v2.md §4-5) picks the weights.
DEFAULT_SCORER_V2: LayerBalanced = layer_balanced({layer: 1.0 for layer in LAYERS})


def default_scorer(n_dims: int):
    """DEFAULT_SCORER for v1 vectors, DEFAULT_SCORER_V2 for v2 vectors."""
    return DEFAULT_SCORER if dims_for_length(n_dims) == "v1" else DEFAULT_SCORER_V2
