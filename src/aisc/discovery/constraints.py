"""Hard constraints applied before scoring."""

from __future__ import annotations

from aisc.contracts import Program


def passes_constraints(program: Program, constraints: dict[str, object]) -> bool:
    """Keep a program unless it is *known* to violate a constraint.

    Unknown values (None) are kept on purpose: dropping them would silently
    bias recommendations toward well-documented programs (B-C0-4).
    """
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
