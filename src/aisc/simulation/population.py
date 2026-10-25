"""C5.1 Synthetic student population with known ground truth.

Generative story for one student (each step is a registered assumption):

1. Archetype -> true needs w*            (A-C5-1: prototype + individual noise)
2. Undergraduate background CIP family   (matches the archetype only with
                                          probability `background_match_prob`:
                                          the core "wrong major" situation)
3. Familiarity: which programs they know (A-C5-2: logistic in prestige,
                                          same-family-as-background, TW-common)
4. Anchors: chosen from familiar programs (A-C5-3: multinomial logit on
                                          (1-π)·fit + π·prestige)
5. Stated needs: w* + acquiescence + social desirability + noise (A-C5-4)

Only (4) and (5) — and optionally (3) — are visible to the components under
test, via `SyntheticStudent.observed_profile()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.dimensions import NEED_KEYS
from aisc.simulation.catalog_gen import TW_COMMON_CIP
from aisc.simulation.truth import true_utility

# Student archetypes (independent of the program prototypes, to avoid
# generating students that trivially match catalogue families).
ARCHETYPES: dict[str, dict[str, float]] = {
    "quant_finance": {"quant": 0.8, "programming": 0.5, "business": 0.6, "finance": 0.9,
                      "tech_product": 0.2, "research": 0.2, "entrepreneurship": 0.2,
                      "policy_social": 0.1},
    "tech_product": {"quant": 0.5, "programming": 0.7, "business": 0.5, "finance": 0.1,
                     "tech_product": 0.9, "design": 0.4, "research": 0.1,
                     "entrepreneurship": 0.6, "policy_social": 0.1},
    "design_creative": {"quant": 0.2, "programming": 0.3, "business": 0.3,
                        "tech_product": 0.6, "design": 0.9, "research": 0.2,
                        "entrepreneurship": 0.5, "policy_social": 0.3},
    "research_quant": {"quant": 0.9, "programming": 0.6, "business": 0.1, "finance": 0.2,
                       "tech_product": 0.3, "research": 0.9, "policy_social": 0.2},
    "social_policy": {"quant": 0.5, "programming": 0.3, "business": 0.3, "finance": 0.2,
                      "tech_product": 0.1, "design": 0.2, "research": 0.5,
                      "entrepreneurship": 0.1, "policy_social": 0.9},
    "generalist_business": {"quant": 0.4, "programming": 0.2, "business": 0.9, "finance": 0.5,
                            "tech_product": 0.4, "design": 0.2, "entrepreneurship": 0.6,
                            "policy_social": 0.2},
}  # fmt: skip

# The CIP family a student of each archetype would ideally have studied.
NATURAL_FAMILY: dict[str, str] = {
    "quant_finance": "52",
    "tech_product": "11",
    "design_creative": "50",
    "research_quant": "27",
    "social_policy": "44",
    "generalist_business": "52",
}

# Illustrative distribution of Taiwanese undergraduate backgrounds (A-C5-2).
TW_BACKGROUND_DIST: dict[str, float] = {
    "52": 0.45,
    "11": 0.15,
    "14": 0.15,
    "27": 0.10,
    "50": 0.08,
    "44": 0.07,
}


@dataclass(frozen=True)
class SimConfig:
    n_students: int = 200
    individual_sd: float = 0.15
    background_match_prob: float = 0.4
    # familiarity logit: a0 + a_prestige*prestige + a_background*[same family] + a_tw*[TW common]
    fam_a0: float = -2.5
    fam_a_prestige: float = 3.0
    fam_a_background: float = 1.5
    fam_a_tw_common: float = 1.0
    # anchors
    n_anchors: int = 3
    prestige_weight: float = 0.5  # π in A-C5-3
    anchor_temperature: float = 8.0  # β: higher = more deterministic choice
    # response model for stated needs
    noise_sd: float = 0.15
    acquiescence: float = 0.05
    desirability: dict[str, float] = field(
        default_factory=lambda: {"quant": 0.10, "business": 0.05}
    )
    reported_sigma: float = 0.3  # the uncertainty the stated profile claims
    truth: str = "distance"

    def __post_init__(self) -> None:
        if not 0 <= self.prestige_weight <= 1:
            raise ValueError("prestige_weight must lie in [0, 1]")
        if not 0 <= self.background_match_prob <= 1:
            raise ValueError("background_match_prob must lie in [0, 1]")
        unknown = set(self.desirability) - set(NEED_KEYS)
        if unknown:
            raise ValueError(f"Unknown desirability dimensions: {sorted(unknown)}")


@dataclass(frozen=True)
class SyntheticStudent:
    student_id: str
    archetype: str
    background_family: str
    true_needs: np.ndarray
    familiar: frozenset[str]
    anchors: tuple[str, ...]
    stated_needs: dict[str, float]
    stated_sigma: float

    def observed_profile(self, include_familiar: bool = False) -> StudentProfile:
        """What a component under test is allowed to see."""
        return StudentProfile(
            student_id=self.student_id,
            needs=dict(self.stated_needs),
            uncertainty={k: self.stated_sigma for k in NEED_KEYS},
            anchors=list(self.anchors),
            familiar=sorted(self.familiar) if include_familiar else [],
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def familiarity_probs(catalog: list[Program], background_family: str, cfg: SimConfig) -> np.ndarray:
    prestige = np.array([p.prestige if p.prestige is not None else 0.0 for p in catalog])
    same = np.array([p.cip_family == background_family for p in catalog], dtype=float)
    tw = np.array([p.cip in TW_COMMON_CIP for p in catalog], dtype=float)
    z = cfg.fam_a0 + cfg.fam_a_prestige * prestige + cfg.fam_a_background * same
    return _sigmoid(z + cfg.fam_a_tw_common * tw)


def generate_population(
    catalog: list[Program], cfg: SimConfig, rng: np.random.Generator
) -> list[SyntheticStudent]:
    if not catalog:
        raise ValueError("catalog is empty")
    arche_names = list(ARCHETYPES)
    bg_families = list(TW_BACKGROUND_DIST)
    bg_probs = np.array([TW_BACKGROUND_DIST[f] for f in bg_families])
    bg_probs = bg_probs / bg_probs.sum()
    desirability = np.array([cfg.desirability.get(k, 0.0) for k in NEED_KEYS])

    students: list[SyntheticStudent] = []
    for s in range(cfg.n_students):
        # 1. true needs
        arche = arche_names[rng.integers(len(arche_names))]
        proto = np.array([ARCHETYPES[arche].get(k, 0.0) for k in NEED_KEYS])
        w = np.clip(proto + rng.normal(0.0, cfg.individual_sd, size=proto.shape), 0.0, 1.0)

        # 2. background
        if rng.random() < cfg.background_match_prob:
            background = NATURAL_FAMILY[arche]
        else:
            background = bg_families[rng.choice(len(bg_families), p=bg_probs)]

        # 3. familiarity
        probs = familiarity_probs(catalog, background, cfg)
        familiar_mask = rng.random(len(catalog)) < probs
        familiar_idx = np.flatnonzero(familiar_mask)

        # 4. anchors: MNL without replacement over familiar programs
        anchors: list[str] = []
        if familiar_idx.size:
            fit = np.array([true_utility(w, catalog[i].vector, cfg.truth) for i in familiar_idx])
            prest = np.array([catalog[i].prestige or 0.0 for i in familiar_idx])
            v = (1 - cfg.prestige_weight) * fit + cfg.prestige_weight * prest
            weights = np.exp(cfg.anchor_temperature * (v - v.max()))
            n = min(cfg.n_anchors, familiar_idx.size)
            chosen = rng.choice(familiar_idx.size, size=n, replace=False, p=weights / weights.sum())
            anchors = [catalog[familiar_idx[i]].program_id for i in chosen]

        # 5. stated needs
        noise = rng.normal(0.0, cfg.noise_sd, size=w.shape)
        stated = np.clip(w + cfg.acquiescence + desirability + noise, 0.0, 1.0)

        students.append(
            SyntheticStudent(
                student_id=f"stu-{s:04d}",
                archetype=arche,
                background_family=background,
                true_needs=w,
                familiar=frozenset(catalog[i].program_id for i in familiar_idx),
                anchors=tuple(anchors),
                stated_needs={k: float(v) for k, v in zip(NEED_KEYS, stated, strict=True)},
                stated_sigma=cfg.reported_sigma,
            )
        )
    return students
