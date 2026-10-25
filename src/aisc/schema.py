"""Core data structures: Program and StudentProfile."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aisc.needs import NEED_KEYS


def _vector(values: dict[str, float]) -> np.ndarray:
    unknown = set(values) - set(NEED_KEYS)
    if unknown:
        raise ValueError(f"Unknown need dimensions: {sorted(unknown)}")
    vec = np.array([float(values.get(k, 0.0)) for k in NEED_KEYS])
    if np.any(vec < 0) or np.any(vec > 1):
        raise ValueError("Need scores must lie in [0, 1]")
    return vec


@dataclass(frozen=True)
class Program:
    """A single graduate program (one degree at one school).

    `cip` is the 6-digit NCES CIP 2020 code (e.g. "27.0501"). It drives the
    *label* side of novelty: programs sharing a CIP family look "familiar".
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

    @property
    def vector(self) -> np.ndarray:
        return _vector(self.attributes)

    @property
    def cip_family(self) -> str:
        """2-digit CIP series, e.g. '27' for Mathematics and Statistics."""
        return self.cip.split(".")[0]

    @property
    def cip_4(self) -> str:
        """4-digit CIP group, e.g. '27.05'."""
        return self.cip[:5]


@dataclass
class StudentProfile:
    """What the profiling layer (L1) produces.

    `needs` is the point estimate on each dimension; `uncertainty` is the
    standard deviation of that estimate (high = not yet elicited).
    `anchors` are program_ids the student already has in mind.
    """

    student_id: str
    needs: dict[str, float]
    uncertainty: dict[str, float] = field(default_factory=dict)
    anchors: list[str] = field(default_factory=list)
    hard_constraints: dict[str, object] = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        return _vector(self.needs)


def load_programs(path: str | Path) -> list[Program]:
    """Load programs from a JSON Lines file (one program per line)."""
    programs: list[Program] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                programs.append(Program(**json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    ids = [p.program_id for p in programs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate program_id values")
    return programs
