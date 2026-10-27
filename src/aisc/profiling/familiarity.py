"""C1.4 Familiarity elicitation.

The student is shown a short candidate list (each name with a one-line
description, to limit name-confusion false alarms, B-C1-12) and answers per
program:

    "unknown"       never heard of it
    "heard"         heard the name, cannot say what it trains for (B-C1-6)
    "can_describe"  can say what it trains for

Which levels count as "known" is a policy choice. The default treats both
"heard" and "can_describe" as known, matching the simulator's ground truth.
A strict policy (only "can_describe") keeps half-understood programs eligible
as discoveries; it cannot be evaluated in simulation yet because the
simulator has no ground truth for understanding.
"""

from __future__ import annotations

from dataclasses import replace

from aisc.contracts import StudentProfile

LEVELS: tuple[str, ...] = ("unknown", "heard", "can_describe")
DEFAULT_KNOWN: frozenset[str] = frozenset({"heard", "can_describe"})
STRICT_KNOWN: frozenset[str] = frozenset({"can_describe"})


def known_ids(report: dict[str, str], known_levels: frozenset[str] = DEFAULT_KNOWN) -> set[str]:
    bad = {lvl for lvl in report.values() if lvl not in LEVELS}
    if bad:
        raise ValueError(f"Unknown familiarity levels {sorted(bad)}; expected {LEVELS}")
    if not known_levels <= set(LEVELS):
        raise ValueError("known_levels must be a subset of LEVELS")
    return {pid for pid, lvl in report.items() if lvl in known_levels}


def apply_familiarity(
    profile: StudentProfile,
    report: dict[str, str],
    known_levels: frozenset[str] = DEFAULT_KNOWN,
) -> StudentProfile:
    """Return a profile whose `familiar` list includes the reported known programs.

    Programs reported "unknown" are removed from `familiar` if present: the
    latest answer wins.
    """
    known = known_ids(report, known_levels)
    unknown = set(report) - known
    familiar = (set(profile.familiar) - unknown) | known
    return replace(
        profile,
        needs=dict(profile.needs),
        uncertainty=dict(profile.uncertainty),
        anchors=list(profile.anchors),
        familiar=sorted(familiar),
        hard_constraints=dict(profile.hard_constraints),
        group_scores=dict(profile.group_scores),
    )
