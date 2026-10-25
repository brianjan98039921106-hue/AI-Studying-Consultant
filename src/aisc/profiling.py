"""L1 profiling helpers.

MVP scope: anchor decomposition. The adaptive questionnaire is a TODO
(see docs/ROADMAP.md, M2).

Anchor decomposition: a student who names programs they already like is
telling us something about their needs, even if they cannot articulate it.
We infer a need vector from those anchors and blend it with whatever the
student stated directly, weighting each by how confident we are.
"""

from __future__ import annotations

import numpy as np

from aisc.needs import NEED_KEYS
from aisc.schema import Program, StudentProfile


def needs_from_anchors(anchors: list[Program]) -> np.ndarray:
    """Mean attribute vector of the anchor programs."""
    if not anchors:
        raise ValueError("At least one anchor program is required")
    return np.mean([p.vector for p in anchors], axis=0)


def blend_profile(
    profile: StudentProfile,
    catalog: dict[str, Program],
    default_uncertainty: float = 0.5,
) -> StudentProfile:
    """Precision-weighted blend of stated needs and anchor-implied needs.

    For each dimension, stated value s with std sigma_s and anchor value a
    with std sigma_a combine as (s/sigma_s^2 + a/sigma_a^2) / (1/sigma_s^2 + 1/sigma_a^2).
    The anchor estimate's std shrinks as more anchors are given.
    """
    anchors = [catalog[pid] for pid in profile.anchors if pid in catalog]
    if not anchors:
        return profile

    stated = profile.vector
    sigma_s = np.array([profile.uncertainty.get(k, default_uncertainty) for k in NEED_KEYS])
    implied = needs_from_anchors(anchors)
    sigma_a = np.full(len(NEED_KEYS), default_uncertainty / np.sqrt(len(anchors)))

    w_s, w_a = 1.0 / sigma_s**2, 1.0 / sigma_a**2
    blended = (stated * w_s + implied * w_a) / (w_s + w_a)
    blended_sigma = np.sqrt(1.0 / (w_s + w_a))

    return StudentProfile(
        student_id=profile.student_id,
        needs={k: float(np.clip(v, 0, 1)) for k, v in zip(NEED_KEYS, blended, strict=True)},
        uncertainty={k: float(v) for k, v in zip(NEED_KEYS, blended_sigma, strict=True)},
        anchors=list(profile.anchors),
        hard_constraints=dict(profile.hard_constraints),
    )
