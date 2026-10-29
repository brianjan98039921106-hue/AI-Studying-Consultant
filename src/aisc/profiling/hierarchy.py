"""C1 v2 domain layer: area groups and priors for unanswered areas (W-16c, W-66).

docs/dimensions-v2.md §2-3. The 41 O*NET specific interest areas are asked in two
stages: first one item per area group (12 groups, F67), then the areas of the
groups the student rated highest. An area the student did not rate gets a prior:

    any of its groups rated -> mean = highest rating among those groups, sd 0.25
    otherwise               -> mean = occupation base rate of the area, sd 0.30

Never the neutral 0.5 of v1: on 41 areas that would claim interest in everything
(B-C1-24; O*NET occupations average 1.63 / 7 on an area).

Constants are copied from data/reference/ (interest_area_clusters.json and the
`occupation_mean` column of basic_interest_map.csv); tests check they still match.
Cross-links (an area reachable from more than one group) are design choices for the
weakly fitting areas of F68 and the doubtful pairing of F69 (A-C1-24).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from aisc.dimensions import BASIC_INTEREST_KEYS

GROUP_SIGMA = 0.25
BASE_SIGMA = 0.30

# group id -> (Chinese label, English label); labels describe activities (F69)
GROUP_LABELS: dict[int, tuple[str, str]] = {
    1: ("經營、銷售、金融與行政", "Business, sales, finance and administration"),
    2: ("藝術、設計、媒體與人文", "Arts, design, media and humanities"),
    3: ("教育、社會服務與社會科學", "Education, social service and social science"),
    4: ("農業、自然與動物", "Agriculture, nature and animals"),
    5: ("營建、運輸與體力工作", "Construction, transport and manual work"),
    6: ("生命科學、醫學與醫療照護", "Life science, medicine and health care"),
    7: ("物理科學、數學統計與資訊科技", "Physical science, mathematics/statistics and IT"),
    8: ("機電與工程", "Mechanics, electronics and engineering"),
    9: ("法律與公共安全", "Law and public safety"),
    10: ("運動", "Athletics"),
    11: ("烹飪藝術", "Culinary art"),
    12: ("宗教活動", "Religious activities"),
}

# area -> groups (first = primary group from the pre-registered clustering, F67;
# extra groups = cross-links, A-C1-24)
AREA_GROUPS: dict[str, tuple[int, ...]] = {
    "bi_mechanics_electronics": (8,),
    "bi_construction_woodwork": (5,),
    "bi_transportation_machine_operation": (5,),
    "bi_physical_manual_labor": (5,),
    "bi_protective_service": (9,),
    "bi_agriculture": (4,),
    "bi_nature_outdoors": (4,),
    "bi_animal_service": (4,),
    "bi_athletics": (10,),
    "bi_engineering": (8,),
    "bi_physical_science": (7,),
    "bi_life_science": (6,),
    "bi_medical_science": (6,),
    "bi_social_science": (3,),
    "bi_humanities": (2, 3),
    "bi_mathematics_statistics": (7,),
    "bi_information_technology": (7,),
    "bi_visual_arts": (2,),
    "bi_applied_arts_and_design": (2,),
    "bi_performing_arts": (2,),
    "bi_music": (2,),
    "bi_creative_writing": (2,),
    "bi_media": (2,),
    "bi_culinary_art": (11,),
    "bi_teaching_education": (3,),
    "bi_social_service": (3,),
    "bi_health_care_service": (6,),
    "bi_religious_activities": (12,),
    "bi_personal_service": (3,),
    "bi_professional_advising": (3,),
    "bi_business_initiatives": (1,),
    "bi_sales": (1,),
    "bi_marketing_advertising": (1, 2),
    "bi_finance": (1,),
    "bi_accounting": (1,),
    "bi_human_resources": (1, 3),
    "bi_office_work": (1,),
    "bi_management_administration": (1,),
    "bi_public_speaking": (3,),
    "bi_politics": (3, 9),
    "bi_law": (9, 3),
}

# occupation base rate per area: mean O*NET OI over all 891 occupations, (v - 1) / 6
BASE_RATE: dict[str, float] = {
    "bi_mechanics_electronics": 0.2269, "bi_construction_woodwork": 0.0688,
    "bi_transportation_machine_operation": 0.1241, "bi_physical_manual_labor": 0.2674,
    "bi_protective_service": 0.0726, "bi_agriculture": 0.0340, "bi_nature_outdoors": 0.0573,
    "bi_animal_service": 0.0162, "bi_athletics": 0.0282, "bi_engineering": 0.2119,
    "bi_physical_science": 0.1107, "bi_life_science": 0.1127, "bi_medical_science": 0.1131,
    "bi_social_science": 0.1225, "bi_humanities": 0.0678, "bi_mathematics_statistics": 0.1835,
    "bi_information_technology": 0.1530, "bi_visual_arts": 0.0565,
    "bi_applied_arts_and_design": 0.0708, "bi_performing_arts": 0.0303, "bi_music": 0.0191,
    "bi_creative_writing": 0.0403, "bi_media": 0.0686, "bi_culinary_art": 0.0249,
    "bi_teaching_education": 0.1861, "bi_social_service": 0.1484,
    "bi_health_care_service": 0.1574, "bi_religious_activities": 0.0088,
    "bi_personal_service": 0.1501, "bi_professional_advising": 0.1330,
    "bi_business_initiatives": 0.0919, "bi_sales": 0.0836, "bi_marketing_advertising": 0.0659,
    "bi_finance": 0.0766, "bi_accounting": 0.1233, "bi_human_resources": 0.1139,
    "bi_office_work": 0.1788, "bi_management_administration": 0.2239,
    "bi_public_speaking": 0.1791, "bi_politics": 0.0475, "bi_law": 0.0809,
}  # fmt: skip

if set(AREA_GROUPS) != set(BASIC_INTEREST_KEYS) or set(BASE_RATE) != set(BASIC_INTEREST_KEYS):
    raise RuntimeError("hierarchy constants are out of sync with BASIC_INTEREST_KEYS")


def areas_in_group(group: int) -> tuple[str, ...]:
    """Areas asked in stage 2 when `group` is chosen (primary members and cross-links)."""
    if group not in GROUP_LABELS:
        raise ValueError(f"unknown area group {group}")
    return tuple(k for k in BASIC_INTEREST_KEYS if group in AREA_GROUPS[k])


def areas_to_ask(
    group_scores: Mapping[int, float], m: int | None = None, q: int | None = None
) -> list[str]:
    """Stage-2 question order (dimensions-v2 §2, W-66): the areas of the highest-rated
    group first (areas_in_group order, cross-linked areas once), then the next group.
    `m` keeps the areas of the top m groups; `q` cuts the list to q areas. Neither has
    a default: the pre-registered L3 choice (W-66, scripts/run_l3_two_stage.py)
    found no m or q that is non-inferior to answering all areas under both
    generators (F180-F181); F115 is occupation-data evidence only. Ties between
    groups go to the lower group id."""
    validate_group_scores(group_scores)
    if m is None and q is None:
        raise ValueError("give m (groups) or q (areas)")
    if m is not None and not 1 <= m <= len(GROUP_LABELS):
        raise ValueError(f"m must lie in 1..{len(GROUP_LABELS)}")
    if q is not None and q < 0:
        raise ValueError("q must be non-negative")
    groups = sorted(GROUP_LABELS, key=lambda g: (-group_scores.get(g, -1.0), g))
    order: list[str] = []
    for g in groups[: m if m is not None else len(groups)]:
        for k in areas_in_group(g):
            if k not in order:
                order.append(k)
    return order if q is None else order[:q]


def area_prior(area: str, group_scores: Mapping[int, float]) -> tuple[float, float]:
    """(prior mean, prior sd) for an area the student has not rated."""
    if area not in AREA_GROUPS:
        raise ValueError(f"not an interest area: {area!r}")
    rated = [float(group_scores[g]) for g in AREA_GROUPS[area] if g in group_scores]
    if rated:
        return max(rated), GROUP_SIGMA
    return BASE_RATE[area], BASE_SIGMA


def validate_group_scores(group_scores: Mapping[int, float]) -> None:
    unknown = set(group_scores) - set(GROUP_LABELS)
    if unknown:
        raise ValueError(f"unknown area groups {sorted(unknown)}")
    if any(not 0 <= v <= 1 for v in group_scores.values()):
        raise ValueError("group scores must lie in [0, 1]")


# --- correlated completion (opt-in; architecture-review §2.1, A-C1-25, A-C1-26) --------

AREA_MODELS = ("hierarchy", "correlated")
_MODEL_PATH = Path(__file__).with_name("interest_area_model.json")  # package data


@lru_cache(maxsize=1)
def _area_model() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    spec = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
    names = tuple(spec["variables"])
    expected = (*BASIC_INTEREST_KEYS, *(f"group_{g}" for g in sorted(GROUP_LABELS)))
    if names != expected:
        raise RuntimeError("interest_area_model.json is out of sync with the area registry")
    return names, np.array(spec["mean"]), np.array(spec["covariance"])


def correlated_area_priors(
    rated_areas: Mapping[str, float],
    group_scores: Mapping[int, float],
    answer_sd: float = 0.15,
) -> dict[str, tuple[float, float]]:
    """(prior mean, prior sd) for every unrated area: conditional mean and sd of a
    joint normal over the 41 areas and the 12 group maxima, given the rated areas
    and group answers (observation noise `answer_sd`). Learned from O*NET
    occupations (data/reference/interest_area_model.json). Means are clipped to
    [0, 1]; sds are floored at 0.05."""
    validate_group_scores(group_scores)
    unknown = set(rated_areas) - set(BASIC_INTEREST_KEYS)
    if unknown:
        raise ValueError(f"not interest areas: {sorted(unknown)}")
    names, mu, S = _area_model()
    obs = [names.index(k) for k in rated_areas] + [
        names.index(f"group_{g}") for g in sorted(group_scores)
    ]
    y = np.array(list(rated_areas.values()) + [group_scores[g] for g in sorted(group_scores)])
    un = [i for i, k in enumerate(names) if k in BASIC_INTEREST_KEYS and k not in rated_areas]
    if not un:
        return {}
    if obs:
        Soo = S[np.ix_(obs, obs)] + np.eye(len(obs)) * max(answer_sd**2, 1e-6)
        Suo = S[np.ix_(un, obs)]
        mean = mu[un] + Suo @ np.linalg.solve(Soo, y - mu[obs])
        cov = S[np.ix_(un, un)] - Suo @ np.linalg.solve(Soo, Suo.T)
    else:
        mean, cov = mu[un], S[np.ix_(un, un)]
    sd = np.sqrt(np.clip(np.diag(cov), 0.05**2, None))
    return {
        names[i]: (float(np.clip(m, 0, 1)), float(s)) for i, m, s in zip(un, mean, sd, strict=True)
    }
