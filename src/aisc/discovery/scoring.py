"""C2.1 relevance, C2.2 novelty, C2.4 contributions.

Two spaces are kept separate on purpose:
- need space (attribute vectors) -> relevance
- label space (CIP taxonomy)     -> novelty
A good discovery is close in need space and far in label space.
"""

from __future__ import annotations

import numpy as np

from aisc.contracts import Program
from aisc.dimensions import NEED_KEYS


def relevance(student: np.ndarray, program: np.ndarray) -> float:
    """C2.1 v0: cosine similarity.

    Known limitation B-C2-3: cosine ignores magnitude, so a student with weak
    needs everywhere matches a program with weak attributes everywhere.
    """
    denom = float(np.linalg.norm(student) * np.linalg.norm(program))
    return 0.0 if denom == 0 else float(student @ program / denom)


def relevance_matrix(student: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Vectorised `relevance` against every row of X (must stay equal to it row-wise)."""
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(student)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = X @ student / norms
    return np.where(norms == 0, 0.0, scores)


def contributions(student: np.ndarray, program: np.ndarray) -> dict[str, float]:
    """C2.4: share of the cosine numerator contributed by each dimension.

    Exact decomposition of the dot product, so the explanation is faithful to
    the v0 scorer by construction (property-tested).
    """
    products = student * program
    total = float(products.sum())
    if total == 0:
        return {k: 0.0 for k in NEED_KEYS}
    return {k: float(v / total) for k, v in zip(NEED_KEYS, products, strict=True)}


def label_similarity(a: Program, b: Program) -> float:
    """1.0 same 4-digit CIP group, 0.5 same 2-digit family, else 0.0."""
    if a.cip_4 == b.cip_4:
        return 1.0
    if a.cip_family == b.cip_family:
        return 0.5
    return 0.0


def novelty(program: Program, anchors: list[Program], familiar: list[Program] = ()) -> float:
    """C2.2 v0: 1 - max label similarity to anything the student knows.

    "Knows" = anchors plus programs the student reported as familiar (C1.4).
    Known limitation B-C2-1: label distance is a proxy for *unawareness*;
    a student may already know a program in a distant CIP family. Reporting
    familiarity directly (C1.4) is the planned fix.
    """
    if any(program.program_id == f.program_id for f in familiar):
        return 0.0
    known = list(anchors) + list(familiar)
    if not known:
        return 1.0
    return 1.0 - max(label_similarity(program, k) for k in known)
