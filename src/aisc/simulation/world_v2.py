"""C5 world for the v1 vs v2 dimension comparison (W-16c, docs/dimensions-v2.md §5).

One ground truth, two views:
    truth   v2 needs/attributes (51 dims) + 3 residual dims nobody can measure
    v2 view the 51 v2 dimensions                       (what a v2 profile can express)
    v1 view the 15 v1 dimensions, projected from v2     (what a v1 profile can express)
The residual dims keep the comparison from being circular for v2 (S-1, B-C5-1):
neither view sees everything the truth uses.

Built on the all-fields population (W-63, W-67): v1 measured needs and the six
hidden domains are converted to v2 with fixed maps (A-C5-12). Synthetic only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import (
    BASIC_INTEREST_KEYS,
    NEED_KEYS,
    NEED_KEYS_V2,
    V1_DOMAIN_TO_V2,
    migrate_needs_v1_to_v2,
)
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS, areas_in_group
from aisc.simulation.fields import HIDDEN_KEYS
from aisc.simulation.population import SyntheticStudent

N_RESIDUAL = 3
AREA_LOW = 0.05  # areas with no source: almost no interest / content
# hidden domain of the all-fields simulator -> v2 areas it becomes (A-C5-12)
HIDDEN_TO_AREAS: dict[str, tuple[str, ...]] = {
    "h_health": ("bi_medical_science", "bi_health_care_service"),
    "h_life_science": ("bi_life_science",),
    "h_materials": ("bi_engineering", "bi_physical_science"),
    "h_humanities": ("bi_humanities", "bi_creative_writing"),
    "h_education": ("bi_teaching_education",),
    "h_law": ("bi_law",),
}
if set(HIDDEN_TO_AREAS) != set(HIDDEN_KEYS):
    raise RuntimeError("HIDDEN_TO_AREAS out of sync with HIDDEN_KEYS")


def to_v2(v1_values: dict[str, float], hidden: np.ndarray | None) -> dict[str, float]:
    """v1 measured values + hidden domains -> 51 v2 values."""
    out = {k: AREA_LOW for k in BASIC_INTEREST_KEYS}
    out.update(migrate_needs_v1_to_v2(v1_values))
    if hidden is not None:
        for key, value in zip(HIDDEN_KEYS, hidden, strict=True):
            for area in HIDDEN_TO_AREAS[key]:
                out[area] = max(out[area], float(value))
    missing = set(NEED_KEYS_V2) - set(out)
    if missing:
        raise ValueError(f"v1 values incomplete: {sorted(missing)}")
    return {k: float(np.clip(out[k], 0, 1)) for k in NEED_KEYS_V2}


def to_v1(v2_values: dict[str, float]) -> dict[str, float]:
    """v2 values -> the 15 v1 dimensions (v1 domain = mean of its mapped areas)."""
    out = {k: float(v2_values[k]) for k in NEED_KEYS if k in v2_values}
    for v1_key, areas in V1_DOMAIN_TO_V2.items():
        out[v1_key] = float(np.mean([v2_values[a] for a in areas]))
    return out


def v2_catalog(
    catalog_v1: list[Program], hidden: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[list[Program], list[Program], dict[str, np.ndarray]]:
    """(v2 programs, their v1 projections, residual vectors) with shared program ids."""
    v2, v1, residual = [], [], {}
    for p in catalog_v1:
        attrs = to_v2(p.attributes, hidden[p.program_id])
        v2.append(replace(p, attributes=attrs, dims="v2"))
        v1.append(replace(p, attributes=to_v1(attrs), dims="v1"))
        residual[p.program_id] = rng.random(N_RESIDUAL)
    return v2, v1, residual


def v2_students(
    population: list[SyntheticStudent], rng: np.random.Generator, area_sd: float = 0.08
) -> list[tuple[SyntheticStudent, np.ndarray, np.ndarray]]:
    """(student, true v2 needs, true residual needs) for an all-fields population."""
    out = []
    for st in population:
        v1_true = dict(zip(NEED_KEYS, map(float, st.true_needs), strict=True))
        base = to_v2(v1_true, st.hidden_needs)
        w = np.array([base[k] for k in NEED_KEYS_V2])
        areas = [NEED_KEYS_V2.index(k) for k in BASIC_INTEREST_KEYS]
        w[areas] = np.clip(w[areas] + rng.normal(0, area_sd, len(areas)), 0, 1)
        out.append((st, w, rng.random(N_RESIDUAL)))
    return out


def stated(true: np.ndarray, keys: tuple[str, ...], rng: np.random.Generator,
           noise_sd: float, bias: float) -> dict[str, float]:  # fmt: skip
    """One noisy answer per key (acquiescence `bias`, response noise)."""
    vals = np.clip(true + bias + rng.normal(0, noise_sd, len(true)), 0, 1)
    return {k: float(v) for k, v in zip(keys, vals, strict=True)}


def group_answer(true_v2: dict[str, float], group: int) -> float:
    """Respondent model for a stage-1 group item: the strongest primary area (A-C5-12)."""
    members = [k for k, g in AREA_GROUPS.items() if g[0] == group]
    return max(true_v2[k] for k in members)


def two_stage_profile(
    student: SyntheticStudent,
    true_v2: np.ndarray,
    m: int,
    rng: np.random.Generator,
    noise_sd: float,
    bias: float,
) -> tuple[StudentProfile, int]:
    """v2 profile from the two-stage questionnaire (dimensions-v2 §2): non-domain
    layers asked once each, 12 group items, then every area of the top-m groups.
    Returns the profile and the number of items asked."""
    true = dict(zip(NEED_KEYS_V2, map(float, true_v2), strict=True))
    other = tuple(k for k in NEED_KEYS_V2 if k not in BASIC_INTEREST_KEYS)
    needs = stated(np.array([true[k] for k in other]), other, rng, noise_sd, bias)
    groups = sorted(GROUP_LABELS)
    g_true = np.array([group_answer(true, g) for g in groups])
    g_ans = np.clip(g_true + bias + rng.normal(0, noise_sd, len(groups)), 0, 1)
    group_scores = {g: float(v) for g, v in zip(groups, g_ans, strict=True)}
    top = [g for _, g in sorted(zip(-g_ans, groups, strict=True))[:m]]
    asked = tuple(dict.fromkeys(a for g in top for a in areas_in_group(g)))
    needs.update(stated(np.array([true[a] for a in asked]), asked, rng, noise_sd, bias))
    profile = StudentProfile(
        student.student_id, needs, {k: 0.3 for k in needs}, anchors=list(student.anchors),
        dims="v2", group_scores=group_scores,
    )  # fmt: skip
    return profile, len(other) + len(groups) + len(asked)
