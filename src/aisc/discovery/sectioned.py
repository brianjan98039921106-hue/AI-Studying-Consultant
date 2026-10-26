"""C2.3 v1: fit-first ranking presented in two sections (W-28).

Replaces v0's additive novelty score, which L3 rejected (F1: it lowered both
relevance and novel-relevant rate under every simulator setting). v0
(`rerank.recommend`) is kept as the reference implementation.

Pipeline position:
    shortlist()  -> candidates for the C1.4 familiarity check
    discover()   -> after the check, split candidates into
                    discoveries (not known) and known_good (known), by fit
"""

from __future__ import annotations

from aisc.contracts import DiscoveryResult, Program, Recommendation, StudentProfile
from aisc.discovery.constraints import passes_constraints
from aisc.discovery.scoring import contributions, relevance


def _ranked(profile: StudentProfile, catalog: list[Program], min_relevance: float):
    anchors = set(profile.anchors)
    target = profile.vector
    scored = []
    for p in catalog:
        if p.program_id in anchors or not passes_constraints(p, profile.hard_constraints):
            continue
        rel = relevance(target, p.vector)
        if rel >= min_relevance:
            scored.append((rel, p))
    scored.sort(key=lambda t: (-t[0], t[1].program_id))
    return scored


def shortlist(
    profile: StudentProfile, catalog: list[Program], n: int = 25, min_relevance: float = 0.0
) -> list[str]:
    """Top-n programs by fit (anchors excluded): the ones to ask about in C1.4.

    Size n should be at least k_new / (expected unfamiliar rate) so that enough
    unknown programs remain after the check (B-C2-7).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    profile.require_complete("C2 shortlist")
    return [p.program_id for _, p in _ranked(profile, catalog, min_relevance)[:n]]


def discover(
    profile: StudentProfile,
    catalog: list[Program],
    k_new: int = 10,
    k_known: int = 5,
    min_relevance: float = 0.0,
    asked: list[str] | None = None,
) -> DiscoveryResult:
    """Split fit-ranked programs by the student's reported familiarity."""
    if k_new < 0 or k_known < 0:
        raise ValueError("k_new and k_known must be non-negative")
    profile.require_complete("C2 discover")
    familiar = set(profile.familiar)
    target = profile.vector
    new: list[Recommendation] = []
    known: list[Recommendation] = []
    for rel, p in _ranked(profile, catalog, min_relevance):
        is_known = p.program_id in familiar
        bucket = known if is_known else new
        limit = k_known if is_known else k_new
        if len(bucket) < limit:
            bucket.append(
                Recommendation(
                    program=p,
                    relevance=rel,
                    novelty=0.0 if is_known else 1.0,
                    score=rel,
                    contributions=contributions(target, p.vector),
                )
            )
        if len(new) >= k_new and len(known) >= k_known:
            break
    return DiscoveryResult(discoveries=new, known_good=known, asked=list(asked or []))
