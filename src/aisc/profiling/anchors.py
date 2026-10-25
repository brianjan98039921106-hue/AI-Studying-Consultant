"""C1.2 Anchor decomposition (v0).

A student who names programs they already like is revealing something
about their needs even if they cannot articulate it. We infer a need vector
from those anchors and blend it with stated needs by precision weighting
(the posterior mean of two independent Gaussian measurements).

Known blind spot B-C1-2 (docs/REGISTER.md): Taiwanese applicants often pick
anchors by brand/ranking, not by fit, so the anchor mean partly encodes
prestige. v0 does not correct for this; C1.2 v1 adds a prestige covariate.
"""

from __future__ import annotations

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import NEED_KEYS


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

    For each dimension: (s/σs² + a/σa²) / (1/σs² + 1/σa²), with
    σa = σ0 / sqrt(#anchors). Assumes the two measurements are independent
    and unbiased (A-C1-3) — exactly the assumption B-C1-2 says is violated.
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
        familiar=list(profile.familiar),
        hard_constraints=dict(profile.hard_constraints),
    )
