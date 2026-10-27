"""Hard constraints applied before scoring."""

from __future__ import annotations

import math

from aisc.contracts import Program

CONSTRAINT_KEYS = ("spring_intake", "stem_opt", "max_tuition_usd")


def validate_constraints(constraints: dict[str, object]) -> None:
    """Known keys and types only (B-C2-20): spring_intake / stem_opt are bool,
    max_tuition_usd is a finite non-negative number (not a bool)."""
    unknown = sorted(set(constraints) - set(CONSTRAINT_KEYS))
    if unknown:
        raise ValueError(f"unknown hard constraints {unknown}; supported: {CONSTRAINT_KEYS}")
    for key in ("spring_intake", "stem_opt"):
        if key in constraints and not isinstance(constraints[key], bool):
            raise ValueError(f"{key} must be True or False")
    if "max_tuition_usd" in constraints:
        v = constraints["max_tuition_usd"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError("max_tuition_usd must be a finite non-negative number")


def passes_constraints(program: Program, constraints: dict[str, object]) -> bool:
    """Keep a program unless it is *known* to violate a constraint.

    Unknown values (None) are kept on purpose: dropping them would silently
    bias recommendations toward well-documented programs (B-C0-4).
    """
    validate_constraints(constraints)
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
