"""C5 all-fields extension (W-63, W-67): far-field programs, hidden domains and
field-switching students.

Hidden domains (A-C5-10) are part of the simulator's ground truth only. Programs
and students have values on them, but no component under test can see them:
`Program.attributes` and stated needs cover the 15 measured dimensions. This is
how the simulator represents needs that the current questionnaire cannot express
(B-C1-19) and lets L3 measure recommendations that look right on the measured
dimensions but are wrong on the hidden ones (B-C2-8).

All prototypes, archetypes and CIP labels below are synthetic modelling choices,
not measurements; school names are fictional.
"""

from __future__ import annotations

import numpy as np

from aisc.contracts import Program
from aisc.dimensions import NEED_KEYS

HIDDEN_KEYS: tuple[str, ...] = (
    "h_health",
    "h_life_science",
    "h_materials",
    "h_humanities",
    "h_education",
    "h_law",
)
# CIP family each hidden domain points to (for a student's natural family).
HIDDEN_TO_FAMILY: dict[str, str] = {
    "h_health": "51",
    "h_life_science": "26",
    "h_materials": "14",
    "h_humanities": "23",
    "h_education": "13",
    "h_law": "22",
}
# Macro groups define 'adjacent' (same group) vs 'far' (other group) switches (A-C5-11).
MACRO_GROUPS: dict[str, frozenset[str]] = {
    "business_data": frozenset({"52", "11", "27", "30"}),
    "engineering_design": frozenset({"14", "50"}),
    "health_life": frozenset({"51", "26"}),
    "humanities_social": frozenset({"23", "13", "22", "44"}),
}
TRANSITIONS: tuple[str, ...] = ("continuing", "adjacent", "far", "multi_degree")
HIDDEN_BASELINE = 0.05  # business-type programs and archetypes: almost no hidden-domain content


def macro_group(family: str) -> str:
    for name, fams in MACRO_GROUPS.items():
        if family in fams:
            return name
    raise ValueError(f"family {family!r} is in no macro group")


def _vec(measured: dict[str, float]) -> dict[str, float]:
    unknown = set(measured) - set(NEED_KEYS)
    if unknown:
        raise ValueError(f"unknown measured dimensions {sorted(unknown)}")
    return {k: float(measured.get(k, 0.0)) for k in NEED_KEYS}


def _hid(values: dict[str, float]) -> np.ndarray:
    unknown = set(values) - set(HIDDEN_KEYS)
    if unknown:
        raise ValueError(f"unknown hidden dimensions {sorted(unknown)}")
    return np.array([values.get(k, HIDDEN_BASELINE) for k in HIDDEN_KEYS], float)


# Far-field program prototypes: measured profiles deliberately overlap business-type
# ones on RIASEC (the B-C2-8 trap); the difference lives in the hidden domains.
FAR_PROTOTYPES: list[tuple[Program, np.ndarray]] = [
    (Program("far-01", "Synthetic School of Dentistry", "DDS (synthetic)", "51.0401",
             _vec({"riasec_r": .7, "riasec_i": .7, "riasec_s": .7, "riasec_c": .4,
                   "quant_intensity": .4, "research_orientation": .3}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_health": .95, "h_life_science": .6})),
    (Program("far-02", "Synthetic Institute of Biology", "MS Biology (synthetic)", "26.0101",
             _vec({"riasec_r": .5, "riasec_i": .9, "riasec_c": .5, "quant_intensity": .6,
                   "programming_intensity": .3, "research_orientation": .8}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_life_science": .95, "h_health": .4})),
    (Program("far-03", "Synthetic Tech Materials", "MS Materials Engineering (synthetic)",
             "14.1801",
             _vec({"riasec_r": .8, "riasec_i": .8, "riasec_c": .4, "quant_intensity": .8,
                   "programming_intensity": .3, "domain_operations": .3,
                   "research_orientation": .6}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_materials": .95})),
    (Program("far-04", "Synthetic College of Letters", "MA English (synthetic)", "23.0101",
             _vec({"riasec_a": .8, "riasec_i": .6, "riasec_s": .5, "domain_design": .2,
                   "domain_policy_social": .3, "research_orientation": .7}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_humanities": .95, "h_education": .3})),
    (Program("far-05", "Synthetic Teachers College", "MEd (synthetic)", "13.0101",
             _vec({"riasec_s": .9, "riasec_a": .4, "riasec_e": .4, "riasec_c": .4,
                   "domain_policy_social": .5, "research_orientation": .3}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_education": .95, "h_humanities": .3})),
    (Program("far-06", "Synthetic Law School", "JD (synthetic)", "22.0101",
             _vec({"riasec_e": .8, "riasec_s": .6, "riasec_i": .6, "riasec_c": .6,
                   "domain_policy_social": .7, "domain_finance": .2}),
             synthetic=True, attribute_source="simulated"),
     _hid({"h_law": .95})),
]  # fmt: skip

# Far-field student archetypes: measured needs + hidden needs + natural family.
FAR_ARCHETYPES: dict[str, tuple[dict[str, float], dict[str, float], str]] = {
    "clinical_health": ({"riasec_r": .6, "riasec_i": .7, "riasec_s": .8, "riasec_c": .4,
                         "quant_intensity": .4, "research_orientation": .3},
                        {"h_health": .9, "h_life_science": .5}, "51"),
    "life_science_lab": ({"riasec_r": .5, "riasec_i": .9, "riasec_c": .4, "quant_intensity": .6,
                          "research_orientation": .8},
                         {"h_life_science": .9, "h_health": .3}, "26"),
    "materials_engineer": ({"riasec_r": .8, "riasec_i": .7, "riasec_c": .5,
                            "quant_intensity": .7, "domain_operations": .3},
                           {"h_materials": .9}, "14"),
    "humanities": ({"riasec_a": .8, "riasec_i": .6, "riasec_s": .5,
                    "domain_policy_social": .3, "research_orientation": .6},
                   {"h_humanities": .9}, "23"),
    "educator": ({"riasec_s": .9, "riasec_a": .4, "riasec_e": .4, "domain_policy_social": .4},
                 {"h_education": .9}, "13"),
    "legal": ({"riasec_e": .8, "riasec_s": .6, "riasec_i": .5, "riasec_c": .6,
               "domain_policy_social": .7},
              {"h_law": .9}, "22"),
}  # fmt: skip


def all_fields_catalog(
    rng: np.random.Generator,
    business_catalog: list[Program],
    n_per_family: int = 10,
    within_sd: float = 0.12,
    prestige_beta: tuple[float, float] = (2.0, 5.0),
) -> tuple[list[Program], dict[str, np.ndarray]]:
    """Business-type catalogue + far-field programs, and every program's hidden vector.

    Business-type programs get hidden values near HIDDEN_BASELINE; far-field programs
    are drawn around FAR_PROTOTYPES with the same within-family noise.
    """
    from dataclasses import replace

    hidden: dict[str, np.ndarray] = {}
    base_h = np.full(len(HIDDEN_KEYS), HIDDEN_BASELINE)
    for p in business_catalog:
        hidden[p.program_id] = np.clip(base_h + rng.normal(0, 0.03, base_h.shape), 0, 1)
    programs = list(business_catalog)
    for f, (proto, h) in enumerate(FAR_PROTOTYPES):
        for i in range(n_per_family):
            vec = np.clip(proto.vector + rng.normal(0, within_sd, proto.vector.shape), 0, 1)
            pid = f"far-{f:02d}-{i:02d}"
            programs.append(
                replace(
                    proto,
                    program_id=pid,
                    school=f"Synthetic Far School {f:02d}-{i:02d}",
                    attributes={k: round(float(v), 4) for k, v in zip(NEED_KEYS, vec, strict=True)},
                    prestige=round(float(rng.beta(*prestige_beta)), 4),
                )
            )
            hidden[pid] = np.clip(h + rng.normal(0, within_sd, h.shape), 0, 1)
    return programs, hidden


def pick_backgrounds(
    transition: str, natural: str, families: list[str], rng: np.random.Generator
) -> tuple[str, ...]:
    """Undergraduate (and later) degree families for a transition type."""
    group = macro_group(natural)
    same = [f for f in families if f != natural and macro_group(f) == group]
    far = [f for f in families if macro_group(f) != group]
    if transition == "continuing":
        return (natural,)
    if transition == "adjacent":
        return (same[rng.integers(len(same))] if same else natural,)
    if transition == "far":
        return (far[rng.integers(len(far))],)
    if transition == "multi_degree":  # e.g. materials -> dentistry -> wants design
        a, b = rng.choice(len(far), size=2, replace=len(far) < 2)
        return (far[a], far[b])
    raise ValueError(f"transition must be one of {TRANSITIONS}")
