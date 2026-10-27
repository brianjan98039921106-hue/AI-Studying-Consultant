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
    C6 requirements -> RequirementGap (from RequirementItem + StudentRecord)

Needs and abilities are kept apart on purpose (design principle D9):
StudentProfile holds what the student wants and is the only student input C2
ranks on; StudentRecord holds what the student has done and is read only by
C6 (requirement gaps) and C3 (admission). A requirement the student can still
meet is shown as a gap, never used to drop or demote a program.

See docs/ARCHITECTURE.md for the full graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from aisc.dimensions import dims_keys


def to_vector(values: dict[str, float], dims: str = "v1") -> np.ndarray:
    """Convert a {dimension: score} mapping to an ordered vector in [0, 1] for a
    dimension set ('v1' = 15 dims, 'v2' = 51 dims, docs/dimensions-v2.md)."""
    keys = dims_keys(dims)
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError(f"Unknown need dimensions for {dims}: {sorted(unknown)}")
    vec = np.array([float(values.get(k, 0.0)) for k in keys])
    if not np.all(np.isfinite(vec)) or np.any(vec < 0) or np.any(vec > 1):
        raise ValueError("Need scores must be finite and lie in [0, 1] (B-C2-15)")
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
    dims: str = "v1"  # dimension set of `attributes` (W-16c)

    def __post_init__(self) -> None:
        if self.prestige is not None and not 0 <= self.prestige <= 1:
            raise ValueError("prestige must lie in [0, 1]")
        dims_keys(self.dims)  # validates the name

    @cached_property
    def vector(self) -> np.ndarray:
        """Validated, cached, read-only attribute vector (Program is immutable)."""
        vec = to_vector(self.attributes, self.dims)
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
    dims: str = "v1"  # dimension set of `needs` (W-16c)
    # v2 only: stage-1 ratings of the 12 interest-area groups (group id -> [0, 1]);
    # used as the prior for areas not rated individually (docs/dimensions-v2.md §3)
    group_scores: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dims_keys(self.dims)  # validates the name
        bad = {k: v for k, v in self.uncertainty.items() if not (math.isfinite(v) and v > 0)}
        if bad:
            raise ValueError(f"uncertainty must be a finite positive SD (B-C1-25): {bad}")
        if self.group_scores and self.dims != "v2":
            raise ValueError("group_scores are defined only for the v2 dimension set")

    @property
    def vector(self) -> np.ndarray:
        """Ordered need vector. Unanswered dimensions become 0 here, which means
        "no interest", not "unknown" (B-C1-14). Ranking code must call
        `require_complete` first."""
        return to_vector(self.needs, self.dims)

    @property
    def missing_dimensions(self) -> list[str]:
        return [k for k in dims_keys(self.dims) if k not in self.needs]

    def require_complete(self, where: str) -> None:
        """Raise if any dimension is unanswered (B-C1-14)."""
        missing = self.missing_dimensions
        if missing:
            raise ValueError(
                f"{where} needs a complete profile; unanswered dimensions {missing}. "
                "Complete it first (pipeline.prepare_profile or profiling.complete_neutral)."
            )


def require_same_dims(profile: StudentProfile, programs: list[Program]) -> None:
    """Raise if a profile and programs use different dimension sets (W-16c)."""
    other = {p.dims for p in programs} - {profile.dims}
    if other:
        raise ValueError(
            f"profile uses dimension set {profile.dims!r} but programs use {sorted(other)}; "
            "migrate one side first (aisc.dimensions.migrate_needs_v1_to_v2)"
        )


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


# --- C6: requirements (D9) ----------------------------------------------------

REQUIREMENT_KINDS = ("course", "test", "portfolio", "experience", "credential", "language", "other")


@dataclass(frozen=True)
class RequirementItem:
    """One published requirement of a program, quoted from the official page.

    `category` is a key of data/reference/requirement_taxonomy.json, or None when
    the text does not map to one (then C6 reports it as 'unknown', never as met).
    `completable_after_admission`: the program lets admitted students finish it
    before or during the first term (bridge course, conditional admission).
    """

    program_id: str
    kind: str
    category: str | None
    level: str  # "required" | "recommended"
    text: str  # verbatim quote
    source_url: str
    retrieved_at: str
    accepts_online: bool | None = None
    completable_after_admission: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in REQUIREMENT_KINDS:
            raise ValueError(f"kind must be one of {REQUIREMENT_KINDS}")
        if self.level not in ("required", "recommended"):
            raise ValueError("level must be 'required' or 'recommended'")
        if not self.text.strip():
            raise ValueError("text must quote the official requirement")
        if not (self.source_url and self.retrieved_at):
            raise ValueError("requirements need source_url and retrieved_at")


@dataclass(frozen=True)
class Degree:
    level: str  # "bachelor" | "master" | "doctorate" | "professional" | "other"
    field: str  # as the student names it, e.g. "Materials Science", "Dentistry (DDS)"
    cip: str | None = None
    year: int | None = None


@dataclass
class StudentRecord:
    """What the student has done (abilities, not wants). Never read by C2.

    `completed`: taxonomy category -> evidence the student gave (course name etc.).
    `planned`:   taxonomy category -> expected completion (ISO date or free text).
    `target_apply_by`: ISO date set BY THE STUDENT; None = exploring, no deadline
    is assumed (D9).
    """

    student_id: str
    degrees: list[Degree] = field(default_factory=list)
    completed: dict[str, str] = field(default_factory=dict)
    planned: dict[str, str] = field(default_factory=dict)
    target_apply_by: str | None = None


@dataclass(frozen=True)
class RequirementGap:
    """C6 output for one program. Shown next to a recommendation; never a filter.

    `none_published`: the official page lists no requirements of this kind; this
    is NOT evidence that there is no bar (B-C6-1: implicit thresholds belong to C3).
    `before_target`: missing required items that cannot be finished after
    admission, reported only when the student set `target_apply_by`.
    """

    program_id: str
    met: tuple[RequirementItem, ...] = ()
    planned: tuple[RequirementItem, ...] = ()
    missing: tuple[RequirementItem, ...] = ()
    unknown: tuple[RequirementItem, ...] = ()
    none_published: bool = False
    before_target: tuple[RequirementItem, ...] = ()

    @property
    def missing_required(self) -> tuple[RequirementItem, ...]:
        return tuple(r for r in self.missing if r.level == "required")


# --- C4 output -------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioResult:
    chosen: list[int]
    expected_utility: float
    method: str
