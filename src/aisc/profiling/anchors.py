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

from aisc.contracts import Program, StudentProfile, require_same_dims
from aisc.dimensions import BASIC_INTEREST_KEYS, dims_keys
from aisc.profiling.hierarchy import (
    AREA_MODELS,
    area_prior,
    correlated_area_priors,
    validate_group_scores,
)


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
    require_same_dims(profile, anchors)
    keys = dims_keys(profile.dims)

    stated = profile.vector
    sigma_s = np.array([profile.uncertainty.get(k, default_uncertainty) for k in keys])
    implied = needs_from_anchors(anchors)
    sigma_a = np.full(len(keys), default_uncertainty / np.sqrt(len(anchors)))

    w_s, w_a = 1.0 / sigma_s**2, 1.0 / sigma_a**2
    blended = (stated * w_s + implied * w_a) / (w_s + w_a)
    blended_sigma = np.sqrt(1.0 / (w_s + w_a))

    return _with(
        profile,
        {k: float(np.clip(v, 0, 1)) for k, v in zip(keys, blended, strict=True)},
        {k: float(v) for k, v in zip(keys, blended_sigma, strict=True)},
    )


# --- v1 (W-23) ----------------------------------------------------------------------
#
# L3 showed that blending anchors into dimensions the student has already answered
# hurts ranking (F4a, F5, F9). v1 changes the anchors' role: they act as a prior for
# dimensions not yet answered (cold start, or questions skipped by C1.3), and enter
# answered dimensions only with a much wider sd. Anchors that are more prestigious
# than the catalogue average get a wider sd again (B-C1-2).

NEUTRAL_VALUE = 0.5  # A-C1-11
NEUTRAL_SIGMA = 0.3
STATED_SIGMA = 0.25  # A-C1-28: one stated answer ~ one item's noise (B-C1-27)


def validate_answers(profile: StudentProfile) -> None:
    """Answered dimensions must be known keys with finite values in [0, 1] (B-C1-26)."""
    keys = set(dims_keys(profile.dims))
    unknown = sorted(set(profile.needs) - keys)
    if unknown:
        raise ValueError(f"unknown dimension keys for {profile.dims}: {unknown}")
    bad = {k: v for k, v in profile.needs.items()
           if not (isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 1)}  # fmt: skip
    if bad:
        raise ValueError(f"answers must be finite values in [0, 1]: {bad}")


def complete_neutral(
    profile: StudentProfile,
    value: float = NEUTRAL_VALUE,
    sigma: float = NEUTRAL_SIGMA,
    area_model: str = "hierarchy",
) -> StudentProfile:
    """Baseline: fill unanswered dimensions with a neutral value.

    A dimension counts as answered if its key is present in `profile.needs`.
    v2 interest areas are never filled with the neutral value (B-C1-24): they get
    the group / base-rate prior of aisc.profiling.hierarchy, or with
    area_model="correlated" the conditional prior learned from O*NET (opt-in).
    """
    validate_answers(profile)
    needs = dict(profile.needs)
    uncertainty = dict(profile.uncertainty)
    priors = _area_priors(profile, area_model)
    for k in dims_keys(profile.dims):
        if k not in needs:
            needs[k], uncertainty[k] = priors.get(k) or _unanswered_prior(profile, k, value, sigma)
        else:
            uncertainty.setdefault(k, STATED_SIGMA)  # B-C1-27
    return _with(profile, needs, uncertainty)


def _area_priors(profile: StudentProfile, area_model: str) -> dict[str, tuple[float, float]]:
    """Joint priors for unrated v2 areas when area_model="correlated", else {}."""
    if area_model not in AREA_MODELS:
        raise ValueError(f"area_model must be one of {AREA_MODELS}")
    if area_model == "hierarchy" or profile.dims != "v2":
        return {}
    rated = {k: v for k, v in profile.needs.items() if k in BASIC_INTEREST_KEYS}
    return correlated_area_priors(rated, profile.group_scores)


def _unanswered_prior(
    profile: StudentProfile, key: str, value: float = 0.5, sigma: float = 0.3
) -> tuple[float, float]:
    if profile.dims == "v2" and key in BASIC_INTEREST_KEYS:
        validate_group_scores(profile.group_scores)
        return area_prior(key, profile.group_scores)
    return value, sigma


def anchor_prior(
    anchors: list[Program],
    catalog: list[Program],
    sigma0: float = 0.6,
    prestige_beta: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Anchor-implied needs and their sd, widened for prestige-driven anchors.

    sd = sigma0 / sqrt(n) * (1 + prestige_beta * z), where z is how many catalogue
    standard deviations the anchors' mean prestige lies above the catalogue mean
    (0 if below, or if prestige is unknown).
    """
    if not anchors:
        raise ValueError("At least one anchor program is required")
    mean = needs_from_anchors(anchors)
    sigma = sigma0 / np.sqrt(len(anchors))
    anchor_prestige = [a.prestige for a in anchors if a.prestige is not None]
    catalog_prestige = np.array([p.prestige for p in catalog if p.prestige is not None])
    if anchor_prestige and catalog_prestige.size > 1 and catalog_prestige.std() > 0:
        z = (np.mean(anchor_prestige) - catalog_prestige.mean()) / catalog_prestige.std()
        sigma *= 1.0 + prestige_beta * max(0.0, float(z))
    return mean, float(sigma)


def blend_profile_v1(
    profile: StudentProfile,
    catalog: dict[str, Program],
    sigma0: float = 0.6,
    prestige_beta: float = 3.0,
    answered_scale: float = float("inf"),
    default_sigma: float = STATED_SIGMA,
    area_model: str = "hierarchy",
) -> StudentProfile:
    """Anchors as a prior for unanswered dimensions.

    - unanswered: precision blend of the neutral prior and the anchor prior;
    - answered:   stated value, blended with the anchor prior only through an sd
                  multiplied by `answered_scale` (inf = anchors ignored there).
    """
    validate_answers(profile)
    # each anchor counts once (B-C1-29)
    anchors = [catalog[pid] for pid in dict.fromkeys(profile.anchors) if pid in catalog]
    if not anchors:
        return complete_neutral(profile, area_model=area_model)
    require_same_dims(profile, anchors)
    a_mean, a_sigma = anchor_prior(anchors, list(catalog.values()), sigma0, prestige_beta)
    priors = _area_priors(profile, area_model)
    # B-C1-32: a dimension any anchor has not measured (placeholder value) is no evidence
    no_anchor_info = {k for a in anchors for k in a.unmeasured}

    needs: dict[str, float] = {}
    uncertainty: dict[str, float] = {}
    for i, k in enumerate(dims_keys(profile.dims)):
        if k in profile.needs:
            s_val, s_sd = profile.needs[k], profile.uncertainty.get(k, default_sigma)
            sd_a = a_sigma * answered_scale
        else:
            s_val, s_sd = priors.get(k) or _unanswered_prior(
                profile, k, NEUTRAL_VALUE, NEUTRAL_SIGMA
            )
            sd_a = a_sigma
        w_s = 1.0 / s_sd**2
        w_a = 0.0 if np.isinf(sd_a) or k in no_anchor_info else 1.0 / sd_a**2
        needs[k] = float(np.clip((s_val * w_s + a_mean[i] * w_a) / (w_s + w_a), 0, 1))
        uncertainty[k] = float(np.sqrt(1.0 / (w_s + w_a)))
    return _with(profile, needs, uncertainty)


def _with(profile: StudentProfile, needs: dict, uncertainty: dict) -> StudentProfile:
    return StudentProfile(
        student_id=profile.student_id,
        needs=needs,
        uncertainty=uncertainty,
        anchors=list(profile.anchors),
        familiar=list(profile.familiar),
        hard_constraints=dict(profile.hard_constraints),
        dims=profile.dims,
        group_scores=dict(profile.group_scores),
    )
