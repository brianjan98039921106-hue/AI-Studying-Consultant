"""C5.1 Synthetic student population with known ground truth.

Generative story for one student (each step is a registered assumption):

1. True needs w*, from one of two model families (W-13, S-1):
     "archetype"  : prototype + individual noise          (A-C5-1)
     "continuous" : logistic-normal, no prototypes, RIASEC
                    hexagon correlations                  (A-C5-6, A-C5-7)
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
from aisc.dimensions import NEED_KEYS, RIASEC_KEYS
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

# Continuous model (A-C5-7): which CIP family each need points to. The ideal
# family of a student is the family of their strongest mapped need.
NEED_TO_FAMILY: dict[str, str] = {
    "domain_finance": "52",
    "riasec_e": "52",
    "domain_tech_product": "11",
    "programming_intensity": "11",
    "domain_operations": "14",
    "domain_design": "50",
    "riasec_a": "50",
    "domain_policy_social": "44",
    "riasec_s": "44",
    "research_orientation": "27",
    "quant_intensity": "27",
}

NEEDS_MODELS: tuple[str, ...] = ("archetype", "continuous")


def riasec_hexagon_corr(adjacent: float = 0.3, alternate: float = 0.0, opposite: float = -0.2):
    """6x6 circulant correlation following Holland's hexagon (A-C5-6)."""
    by_distance = {0: 1.0, 1: adjacent, 2: alternate, 3: opposite}
    corr = np.array([[by_distance[min(abs(i - j), 6 - abs(i - j))] for j in range(6)]
                     for i in range(6)])  # fmt: skip
    if np.linalg.eigvalsh(corr).min() <= 0:
        raise ValueError("RIASEC correlation matrix is not positive definite")
    return corr


def _continuous_cov() -> np.ndarray:
    cov = np.eye(len(NEED_KEYS))
    idx = [NEED_KEYS.index(k) for k in RIASEC_KEYS]
    cov[np.ix_(idx, idx)] = riasec_hexagon_corr()
    return cov


def natural_family_from_needs(w: np.ndarray) -> str:
    keys = list(NEED_TO_FAMILY)
    vals = [w[NEED_KEYS.index(k)] for k in keys]
    return NEED_TO_FAMILY[keys[int(np.argmax(vals))]]


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
    # true-needs model family (W-13)
    needs_model: str = "archetype"
    continuous_mu: float = -0.4  # sigmoid(-0.4 + z): mean need ≈ 0.41
    continuous_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.needs_model not in NEEDS_MODELS:
            raise ValueError(f"needs_model must be one of {NEEDS_MODELS}")
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
    natural_family: str = ""  # CIP family the true needs point to (C5 ground truth)

    @property
    def background_stratum(self) -> str:
        """D12 stratum: 'continuing' if the undergraduate background is the family the
        true needs point to, else 'switching' (W-63 refines switching types)."""
        return "continuing" if self.background_family == self.natural_family else "switching"

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
    chol = np.linalg.cholesky(_continuous_cov()) if cfg.needs_model == "continuous" else None

    students: list[SyntheticStudent] = []
    for s in range(cfg.n_students):
        # 1. true needs
        if chol is None:
            arche = arche_names[rng.integers(len(arche_names))]
            proto = np.array([ARCHETYPES[arche].get(k, 0.0) for k in NEED_KEYS])
            w = np.clip(proto + rng.normal(0.0, cfg.individual_sd, size=proto.shape), 0.0, 1.0)
            natural = NATURAL_FAMILY[arche]
        else:
            arche = "continuous"
            z = chol @ rng.standard_normal(len(NEED_KEYS))
            w = 1.0 / (1.0 + np.exp(-(cfg.continuous_mu + cfg.continuous_scale * z)))
            natural = natural_family_from_needs(w)

        # 2. background
        if rng.random() < cfg.background_match_prob:
            background = natural
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
                natural_family=natural,
            )
        )
    return students
