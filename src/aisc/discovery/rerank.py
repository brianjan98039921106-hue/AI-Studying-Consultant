"""C2.3 Re-ranking: combine relevance and novelty, then diversify with MMR.

MMR: Carbonell & Goldstein (1998). Each step picks the candidate maximising
(1-λ)·score − λ·max similarity to items already selected.
"""

from __future__ import annotations

from aisc.contracts import Program, Recommendation, StudentProfile
from aisc.discovery.constraints import passes_constraints
from aisc.discovery.scoring import contributions, novelty, relevance


def recommend(
    profile: StudentProfile,
    catalog: list[Program],
    k: int = 10,
    novelty_weight: float = 0.3,
    diversity_weight: float = 0.3,
    min_relevance: float = 0.0,
) -> list[Recommendation]:
    """Rank programs by (1-α)·relevance + α·novelty, then diversify with MMR.

    Args:
        novelty_weight: α in [0, 1]. 0 = pure fit, 1 = pure novelty.
            Not yet calibrated (B-C2-2): to be set from V4 user-study data.
        diversity_weight: MMR λ in [0, 1].
        min_relevance: floor on fit; prevents "novel but irrelevant" picks.
    """
    if not 0 <= novelty_weight <= 1 or not 0 <= diversity_weight <= 1:
        raise ValueError("Weights must lie in [0, 1]")
    if k < 0:
        raise ValueError("k must be non-negative")

    by_id = {p.program_id: p for p in catalog}
    anchors = [by_id[pid] for pid in profile.anchors if pid in by_id]
    familiar = [by_id[pid] for pid in profile.familiar if pid in by_id]
    anchor_ids = {p.program_id for p in anchors}
    student = profile.vector

    candidates: list[Recommendation] = []
    for p in catalog:
        if p.program_id in anchor_ids or not passes_constraints(p, profile.hard_constraints):
            continue
        rel = relevance(student, p.vector)
        if rel < min_relevance:
            continue
        nov = novelty(p, anchors, familiar)
        score = (1 - novelty_weight) * rel + novelty_weight * nov
        candidates.append(Recommendation(p, rel, nov, score, contributions(student, p.vector)))

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
