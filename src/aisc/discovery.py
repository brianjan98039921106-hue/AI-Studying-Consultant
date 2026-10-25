"""L2 discovery: recommend programs that fit the student's needs but that
the student would not have found on their own.

The core idea separates two spaces:
- **need space** (attribute vectors) decides *relevance*;
- **label space** (CIP code / degree family) decides *novelty*.

A good discovery is close in need space but far in label space from the
student's anchors: "you want what Columbia MS Finance gives you, and here is
an MEng ORIE that gives you the same thing under a name you never searched".

Final ranking uses Maximal Marginal Relevance (MMR) so the list does not
collapse onto one cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisc.schema import Program, StudentProfile


def relevance(student: np.ndarray, program: np.ndarray) -> float:
    """Cosine similarity between need vector and program attributes."""
    denom = float(np.linalg.norm(student) * np.linalg.norm(program))
    return 0.0 if denom == 0 else float(student @ program / denom)


def label_similarity(a: Program, b: Program) -> float:
    """1.0 same 4-digit CIP group, 0.5 same 2-digit family, else 0.0."""
    if a.cip_4 == b.cip_4:
        return 1.0
    if a.cip_family == b.cip_family:
        return 0.5
    return 0.0


def novelty(program: Program, anchors: list[Program]) -> float:
    """1 - max label similarity to any anchor. 1.0 if there are no anchors."""
    if not anchors:
        return 1.0
    return 1.0 - max(label_similarity(program, a) for a in anchors)


def passes_constraints(program: Program, constraints: dict[str, object]) -> bool:
    """Hard filters. Unknown program values (None) are kept, not dropped."""
    if constraints.get("spring_intake") is True and program.spring_intake is False:
        return False
    if constraints.get("stem_opt") is True and program.stem_opt is False:
        return False
    max_tuition = constraints.get("max_tuition_usd")
    return not (
        isinstance(max_tuition, (int, float))
        and program.tuition_usd is not None
        and program.tuition_usd > max_tuition
    )


@dataclass(frozen=True)
class Recommendation:
    program: Program
    relevance: float
    novelty: float
    score: float


def recommend(
    profile: StudentProfile,
    catalog: list[Program],
    k: int = 10,
    novelty_weight: float = 0.3,
    diversity_weight: float = 0.3,
    min_relevance: float = 0.0,
) -> list[Recommendation]:
    """Rank programs by (1-a)*relevance + a*novelty, then diversify with MMR.

    Args:
        novelty_weight: a in [0, 1]. 0 = pure fit, 1 = pure novelty.
        diversity_weight: MMR lambda in [0, 1]. Penalises programs whose
            attribute vectors are similar to ones already selected.
        min_relevance: programs below this fit are never shown, however novel.
    """
    if not 0 <= novelty_weight <= 1 or not 0 <= diversity_weight <= 1:
        raise ValueError("Weights must lie in [0, 1]")

    by_id = {p.program_id: p for p in catalog}
    anchors = [by_id[pid] for pid in profile.anchors if pid in by_id]
    anchor_ids = {p.program_id for p in anchors}
    student = profile.vector

    candidates: list[Recommendation] = []
    for p in catalog:
        if p.program_id in anchor_ids or not passes_constraints(p, profile.hard_constraints):
            continue
        rel = relevance(student, p.vector)
        if rel < min_relevance:
            continue
        nov = novelty(p, anchors)
        score = (1 - novelty_weight) * rel + novelty_weight * nov
        candidates.append(Recommendation(p, rel, nov, score))

    selected: list[Recommendation] = []
    while candidates and len(selected) < k:

        def mmr(c: Recommendation) -> float:
            if not selected:
                return c.score
            redundancy = max(relevance(c.program.vector, s.program.vector) for s in selected)
            return (1 - diversity_weight) * c.score - diversity_weight * redundancy

        best = max(candidates, key=mmr)
        selected.append(best)
        candidates.remove(best)
    return selected
