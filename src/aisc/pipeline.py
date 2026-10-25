"""System layer: composes components through their public APIs only.

    interactive_discovery:
        C1.2 v1 profile completion -> C2 shortlist -> C1.4 familiarity check (ask)
        -> C2.3 v1 discover

`ask` is whatever collects the student's answers: a UI in production, a
simulated respondent in C5. It receives program_ids and returns
{program_id: "unknown" | "heard" | "can_describe"}.
"""

from __future__ import annotations

from collections.abc import Callable

from aisc.contracts import DiscoveryResult, Program, StudentProfile
from aisc.discovery import discover, shortlist
from aisc.profiling import apply_familiarity, blend_profile_v1

AskFn = Callable[[list[str]], dict[str, str]]


def prepare_profile(profile: StudentProfile, catalog: list[Program]) -> StudentProfile:
    """Complete unanswered dimensions before ranking (B-C1-14).

    Without this, `StudentProfile.vector` treats an unanswered dimension as 0
    ("no interest") instead of "unknown". Uses C1.2 v1 with its confirmed
    defaults (results/synthetic-l3-c1-v1-findings.md): anchors act only as a
    prior for unanswered dimensions; answered dimensions are left unchanged.
    """
    return blend_profile_v1(profile, {p.program_id: p for p in catalog})


def interactive_discovery(
    profile: StudentProfile,
    catalog: list[Program],
    ask: AskFn,
    n_shortlist: int = 25,
    k_new: int = 10,
    k_known: int = 5,
    min_relevance: float = 0.0,
) -> DiscoveryResult:
    profile = prepare_profile(profile, catalog)
    candidates = shortlist(profile, catalog, n=n_shortlist, min_relevance=min_relevance)
    report = ask(candidates) if candidates else {}
    unexpected = set(report) - set(candidates)
    if unexpected:
        raise ValueError(f"ask() answered programs that were not asked: {sorted(unexpected)}")
    updated = apply_familiarity(profile, report)
    return discover(
        updated,
        catalog,
        k_new=k_new,
        k_known=k_known,
        min_relevance=min_relevance,
        asked=candidates,
    )
