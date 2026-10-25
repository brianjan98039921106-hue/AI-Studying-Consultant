"""Data contracts between components.

Components never import each other's internals; they exchange only the
types defined here. This keeps each component independently testable and
lets one implementation be swapped for another (e.g. a new relevance
scorer) without touching its neighbours.

    C0 catalog    -> Program
    C1 profiling  -> StudentProfile
    C2 discovery  -> Recommendation
    C3 feasibility-> AdmitEstimate
    C4 portfolio  -> PortfolioResult
    C5 simulation -> consumes all of the above against known ground truth

See docs/ARCHITECTURE.md for the full graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from aisc.dimensions import NEED_KEYS


def to_vector(values: dict[str, float]) -> np.ndarray:
    """Convert a {dimension: score} mapping to an ordered vector in [0, 1]."""
    unknown = set(values) - set(NEED_KEYS)
    if unknown:
        raise ValueError(f"Unknown need dimensions: {sorted(unknown)}")
    vec = np.array([float(values.get(k, 0.0)) for k in NEED_KEYS])
    if np.any(vec < 0) or np.any(vec > 1):
        raise ValueError("Need scores must lie in [0, 1]")
    return vec


# --- C0 output -------------------------------------------------------------


@dataclass(frozen=True)
class Program:
    """One degree at one school.

    `cip` is the 6-digit NCES CIP 2020 code (e.g. "27.0501").
    `attribute_source` records how `attributes` were measured, because
    C0 is designed to measure each program twice (curriculum text and
    occupation bridge) and compare the two (see docs/components/C0-catalog.md).
    """

    program_id: str
    school: str
    degree_name: str
    cip: str
    attributes: dict[str, float]
    duration_months: int | None = None
    spring_intake: bool | None = None
    stem_opt: bool | None = None
    tuition_usd: int | None = None
    synthetic: bool = False
    attribute_source: str = "hand_set"  # hand_set | curriculum | occupation | fused | simulated
    source_url: str | None = None
    retrieved_at: str | None = None  # ISO date
    prestige: float | None = None  # perceived prestige in [0, 1]; used by C5 and V-C2-4

    def __post_init__(self) -> None:
        if self.prestige is not None and not 0 <= self.prestige <= 1:
            raise ValueError("prestige must lie in [0, 1]")

    @cached_property
    def vector(self) -> np.ndarray:
        """Validated, cached, read-only attribute vector (Program is immutable)."""
        vec = to_vector(self.attributes)
        vec.flags.writeable = False
        return vec

    @property
    def cip_family(self) -> str:
        """2-digit CIP series, e.g. '27' for Mathematics and Statistics."""
        return self.cip.split(".")[0]

    @property
    def cip_4(self) -> str:
        """4-digit CIP group, e.g. '27.05'."""
        return self.cip[:5]


# --- C1 output -------------------------------------------------------------


@dataclass
class StudentProfile:
    """Posterior summary of a student's needs.

    `needs`: point estimate per dimension. `uncertainty`: posterior std
    (large = not yet elicited). `anchors`: program_ids the student already
    has in mind. `familiar`: program_ids the student reports knowing about;
    used by C2 to estimate true novelty (B-C2-1 in docs/REGISTER.md).
    """

    student_id: str
    needs: dict[str, float]
    uncertainty: dict[str, float] = field(default_factory=dict)
    anchors: list[str] = field(default_factory=list)
    familiar: list[str] = field(default_factory=list)
    hard_constraints: dict[str, object] = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        return to_vector(self.needs)


# --- C2 output -------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    program: Program
    relevance: float
    novelty: float
    score: float
    contributions: dict[str, float] = field(default_factory=dict)  # for C2.4 explainer


@dataclass(frozen=True)
class DiscoveryResult:
    """C2.3 v1 output: two sections instead of one novelty-penalised list.

    `discoveries`: programs the student did not report knowing, by fit.
    `known_good`:  programs the student reported knowing that also fit well
                   (confirmation, not a discovery). Anchors appear in neither.
    """

    discoveries: list[Recommendation]
    known_good: list[Recommendation]
    asked: list[str] = field(default_factory=list)  # program_ids shown in the familiarity check

    @property
    def all_ids(self) -> list[str]:
        return [r.program.program_id for r in self.discoveries + self.known_good]


# --- C3 output -------------------------------------------------------------


@dataclass(frozen=True)
class AdmitEstimate:
    """P(admit) with an interval; C4 must use the interval, not just the mean."""

    program_id: str
    mean: float
    low: float
    high: float
    method: str

    def __post_init__(self) -> None:
        if not 0 <= self.low <= self.mean <= self.high <= 1:
            raise ValueError("Require 0 <= low <= mean <= high <= 1")


# --- C4 output -------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioResult:
    chosen: list[int]
    expected_utility: float
    method: str
