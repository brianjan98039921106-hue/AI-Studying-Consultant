"""C1.1 input: CEEC interest inventory scores (大考中心興趣量表) as a RIASEC prior.

Many Taiwanese applicants took the CEEC inventory in high school. It is built
on Holland's typology and reports a 0–99 score per type, a three-letter code
and a differentiation value (區分值; the manual treats < 3 as an unclear code).

We use it only as a *prior*:
- score / 99 -> [0, 1]                                   (A-C1-7)
- prior sd grows with years since the test, capped        (A-C1-8, B-C1-7)
- an unclear code (differentiation < 3) inflates the sd
- newer evidence (short items, stated needs) is merged by precision weighting,
  so it dominates once it is more certain than the old prior.

All numeric defaults are uncalibrated assumptions (A-C1-8); W-41 re-estimates them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from aisc.contracts import StudentProfile
from aisc.dimensions import RIASEC_KEYS

CEEC_TYPES: tuple[str, ...] = ("R", "I", "A", "S", "E", "C")
CEEC_MAX_SCORE = 99
CEEC_UNCLEAR_DIFFERENTIATION = 3.0


@dataclass(frozen=True)
class RiasecPrior:
    needs: dict[str, float]  # riasec_* -> [0, 1]
    sigma: dict[str, float]  # riasec_* -> prior sd
    years_since_test: float


def riasec_prior_from_ceec(
    scores: dict[str, float],
    years_since_test: float,
    differentiation: float | None = None,
    sigma0: float = 0.15,
    growth_per_year: float = 0.05,
    sigma_max: float = 0.5,
    unclear_factor: float = 1.5,
) -> RiasecPrior:
    """Convert CEEC type scores {'R': 0-99, ...} into a RIASEC prior."""
    missing = set(CEEC_TYPES) - set(scores)
    extra = set(scores) - set(CEEC_TYPES)
    if missing or extra:
        raise ValueError(
            f"Need exactly the six types {CEEC_TYPES}; missing={missing} extra={extra}"
        )
    if any(not 0 <= scores[t] <= CEEC_MAX_SCORE for t in CEEC_TYPES):
        raise ValueError(f"CEEC scores must lie in [0, {CEEC_MAX_SCORE}]")
    if years_since_test < 0:
        raise ValueError("years_since_test must be non-negative")

    sigma = min(sigma_max, sigma0 + growth_per_year * years_since_test)
    if differentiation is not None and differentiation < CEEC_UNCLEAR_DIFFERENTIATION:
        sigma = min(sigma_max, sigma * unclear_factor)

    needs = {
        key: scores[t] / CEEC_MAX_SCORE for key, t in zip(RIASEC_KEYS, CEEC_TYPES, strict=True)
    }
    return RiasecPrior(
        needs=needs, sigma={key: sigma for key in RIASEC_KEYS}, years_since_test=years_since_test
    )


def apply_prior(
    profile: StudentProfile, prior: RiasecPrior, default_sigma: float = 0.5
) -> StudentProfile:
    """Merge a RIASEC prior into a profile.

    A dimension counts as elicited if it is present in `profile.needs`.
    - not elicited  -> take the prior value and sd;
    - elicited      -> precision-weighted mean of prior and elicited value.
    Non-RIASEC dimensions are untouched.
    """
    needs = dict(profile.needs)
    uncertainty = dict(profile.uncertainty)
    for key in RIASEC_KEYS:
        p_mean, p_sd = prior.needs[key], prior.sigma[key]
        if key not in needs:
            needs[key], uncertainty[key] = p_mean, p_sd
            continue
        s_sd = uncertainty.get(key, default_sigma)
        w_p, w_s = 1.0 / p_sd**2, 1.0 / s_sd**2
        needs[key] = (p_mean * w_p + needs[key] * w_s) / (w_p + w_s)
        uncertainty[key] = (1.0 / (w_p + w_s)) ** 0.5
    return replace(
        profile,
        needs=needs,
        uncertainty=uncertainty,
        anchors=list(profile.anchors),
        familiar=list(profile.familiar),
        hard_constraints=dict(profile.hard_constraints),
        group_scores=dict(profile.group_scores),
    )
