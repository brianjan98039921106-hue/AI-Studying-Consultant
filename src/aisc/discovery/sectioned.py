"""C2.3 v1: fit-first ranking presented in two sections (W-28).

Replaces v0's additive novelty score, which L3 rejected (F1: it lowered both
relevance and novel-relevant rate under every simulator setting). v0
(`rerank.recommend`) is kept as the reference implementation.

Relevance comes from a C2.1 `Scorer` (default `scorers.DEFAULT_SCORER`); each
Recommendation's `contributions` is that scorer's exact additive explanation, so
the values sum to `relevance` (B-C2-6).

Pipeline position:
    shortlist()  -> candidates for the C1.4 familiarity check
    discover()   -> after the check, split candidates into
                    discoveries (not known) and known_good (known), by fit

Family diversity (B-C2-9, A-C2-15, A-C2-16): with `diversity` = lambda > 0 the
shortlist and the discoveries are picked greedily by fit - lambda * (number already
picked from the same CIP family, first 4 digits). lambda = 0 is the pure fit order.
Explanations stay the scorer's exact decomposition of fit; only the order changes.
"""

from __future__ import annotations

import numpy as np

from aisc.contracts import (
    DiscoveryResult,
    Program,
    Recommendation,
    StudentProfile,
    require_same_dims,
)
from aisc.discovery import scorers as _scorers
from aisc.discovery.constraints import passes_constraints


def _resolve(scorer: _scorers.Scorer | None, n_dims: int = 15) -> _scorers.Scorer:
    return scorer if scorer is not None else _scorers.default_scorer(n_dims)


def _ranked(
    profile: StudentProfile,
    catalog: list[Program],
    min_relevance: float,
    scorer: _scorers.Scorer,
):
    require_same_dims(profile, catalog)
    anchors = set(profile.anchors)
    pool = [
        p
        for p in catalog
        if p.program_id not in anchors and passes_constraints(p, profile.hard_constraints)
    ]
    if not pool:
        return []
    scores = scorer.score_matrix(profile.vector, np.array([p.vector for p in pool]))
    scored = [(float(s), p) for s, p in zip(scores, pool, strict=True) if s >= min_relevance]
    scored.sort(key=lambda t: (-t[0], t[1].program_id))
    return scored


def cip_family(program: Program) -> str:
    """Diversity unit: the 4-digit CIP series (e.g. "52.13"; "9.0101" -> "09.01",
    B-C2-19). A code without a dot is returned unchanged."""
    head, dot, tail = program.cip.strip().partition(".")
    if not dot:
        return program.cip
    return f"{head.zfill(2)}.{tail[:2].ljust(2, '0')}"


def _diversified(ranked: list, n: int, diversity: float) -> list:
    """Greedy pick of n items from (fit, program) pairs sorted by fit: each step takes
    the largest fit - diversity * (count of its CIP family already picked); ties keep
    the fit order. diversity = 0 returns ranked[:n]."""
    if diversity < 0:
        raise ValueError("diversity must be non-negative")
    if diversity == 0:
        return ranked[:n]
    rest = list(ranked)
    picked: list = []
    counts: dict[str, int] = {}
    while rest and len(picked) < n:
        best = max(
            range(len(rest)),
            key=lambda i: (rest[i][0] - diversity * counts.get(cip_family(rest[i][1]), 0), -i),
        )
        item = rest.pop(best)
        picked.append(item)
        fam = cip_family(item[1])
        counts[fam] = counts.get(fam, 0) + 1
    return picked


def shortlist(
    profile: StudentProfile,
    catalog: list[Program],
    n: int = 25,
    min_relevance: float = 0.0,
    scorer: _scorers.Scorer | None = None,
    diversity: float = 0.0,
) -> list[str]:
    """Top-n programs by fit (anchors excluded): the ones to ask about in C1.4.

    Size n should be at least k_new / (expected unfamiliar rate) so that enough
    unknown programs remain after the check (B-C2-7).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    profile.require_complete("C2 shortlist")
    ranked = _ranked(profile, catalog, min_relevance, _resolve(scorer, len(profile.vector)))
    return [p.program_id for _, p in _diversified(ranked, n, diversity)]


def discover(
    profile: StudentProfile,
    catalog: list[Program],
    k_new: int = 10,
    k_known: int = 5,
    min_relevance: float = 0.0,
    asked: list[str] | None = None,
    scorer: _scorers.Scorer | None = None,
    diversity: float = 0.0,
) -> DiscoveryResult:
    """Split fit-ranked programs by the student's reported familiarity.

    `diversity` spreads the discoveries over CIP families (module docstring); the
    known-and-good section stays in fit order."""
    if k_new < 0 or k_known < 0:
        raise ValueError("k_new and k_known must be non-negative")
    if diversity < 0:
        raise ValueError("diversity must be non-negative")
    profile.require_complete("C2 discover")
    scorer = _resolve(scorer, len(profile.vector))
    familiar = set(profile.familiar)
    target = profile.vector
    ranked = _ranked(profile, catalog, min_relevance, scorer)
    unknown = [t for t in ranked if t[1].program_id not in familiar]
    known_ranked = [t for t in ranked if t[1].program_id in familiar]

    def rec(rel: float, p: Program, is_known: bool) -> Recommendation:
        return Recommendation(
            program=p,
            relevance=rel,
            novelty=0.0 if is_known else 1.0,
            score=rel,
            contributions=scorer.explain(target, p.vector),
        )

    new = [rec(r, p, False) for r, p in _diversified(unknown, k_new, diversity)]
    known = [rec(r, p, True) for r, p in known_ranked[:k_known]]
    return DiscoveryResult(discoveries=new, known_good=known, asked=list(asked or []))
