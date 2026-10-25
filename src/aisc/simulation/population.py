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
# Columns follow _ARCHETYPE_COLUMNS, which is checked against NEED_KEYS below.
_ARCHETYPE_COLUMNS = (
    "riasec_r",
    "riasec_i",
    "riasec_a",
    "riasec_s",
    "riasec_e",
    "riasec_c",
    "quant_intensity",
    "programming_intensity",
    "domain_finance",
    "domain_tech_product",
    "domain_operations",
    "domain_design",
    "domain_policy_social",
    "research_orientation",
    "entrepreneurship",
)
_ARCHETYPE_TABLE: dict[str, tuple[float, ...]] = {
    #                      R   I   A   S   E   C   qnt prg fin tch ops dsg pol res ent
    "quant_finance":       (.1, .7, .1, .1, .6, .6, .8, .5, .9, .2, .2, .0, .1, .2, .2),
    "tech_product":        (.4, .6, .4, .2, .6, .3, .5, .7, .1, .9, .3, .4, .1, .1, .6),
    "design_creative":     (.2, .3, .9, .5, .4, .1, .2, .3, .0, .6, .0, .9, .3, .2, .5),
    "research_quant":      (.2, .9, .2, .1, .1, .5, .9, .6, .2, .3, .2, .0, .2, .9, .0),
    "social_policy":       (.1, .5, .3, .9, .3, .3, .5, .3, .2, .1, .1, .2, .9, .5, .1),
    "generalist_business": (.1, .3, .2, .4, .9, .5, .4, .2, .5, .4, .4, .2, .2, .0, .6),
    "operations_systems":  (.6, .6, .1, .2, .4, .7, .7, .5, .1, .4, .9, .1, .1, .3, .2),
}  # fmt: skip
if _ARCHETYPE_COLUMNS != NEED_KEYS:
    raise RuntimeError("Archetype table columns are out of sync with NEED_KEYS")
ARCHETYPES: dict[str, dict[str, float]] = {
    name: dict(zip(NEED_KEYS, values, strict=True)) for name, values in _ARCHETYPE_TABLE.items()
}

# The CIP family a student of each archetype would ideally have studied.
NATURAL_FAMILY: dict[str, str] = {
    "quant_finance": "52",
    "tech_product": "11",
    "design_creative": "50",
    "research_quant": "27",
    "social_policy": "44",
    "generalist_business": "52",
    "operations_systems": "14",
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
        default_factory=lambda: {"quant_intensity": 0.10, "riasec_e": 0.05}
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
