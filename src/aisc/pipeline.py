"""System layer: composes components through their public APIs only.

    interactive_discovery:
        C2 shortlist -> C1.4 familiarity check (ask) -> C2.3 v1 discover

`ask` is whatever collects the student's answers: a UI in production, a
simulated respondent in C5. It receives program_ids and returns
{program_id: "unknown" | "heard" | "can_describe"}.
"""

from __future__ import annotations

from collections.abc import Callable

from aisc.contracts import DiscoveryResult, Program, StudentProfile
from aisc.discovery import discover, shortlist
from aisc.profiling import apply_familiarity

AskFn = Callable[[list[str]], dict[str, str]]


def interactive_discovery(
    profile: StudentProfile,
    catalog: list[Program],
    ask: AskFn,
    n_shortlist: int = 25,
    k_new: int = 10,
    k_known: int = 5,
    min_relevance: float = 0.0,
) -> DiscoveryResult:
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
