"""Synthetic program catalogue for L3 experiments.

Each prototype in data/synthetic/programs.jsonl defines a *family*; programs
are drawn around it with within-family noise (the knob for B-C0-1, "CIP is
too coarse"). Prestige is right-skewed: few programs are highly prestigious.

`TW_COMMON_CIP` marks program types that already exist as recognisable
categories in Taiwanese universities. These flags are illustrative modelling
assumptions (A-C5-2), not measured facts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from aisc.catalog import load_programs
from aisc.contracts import Program
from aisc.dimensions import NEED_KEYS

PROTOTYPES_PATH = Path(__file__).resolve().parents[3] / "data" / "synthetic" / "programs.jsonl"

# Illustrative: categories a Taiwanese undergraduate would typically recognise.
TW_COMMON_CIP: frozenset[str] = frozenset(
    {
        "52.0801",  # finance
        "14.3501",  # industrial engineering
        "27.0501",  # statistics
        "11.0701",  # computer science
        "50.0401",  # design
        "52.0213",  # technology management
        "52.0201",  # MBA / business administration
    }
)


def load_prototypes(path: str | Path = PROTOTYPES_PATH) -> list[Program]:
    return load_programs(path)


def synthetic_catalog(
    rng: np.random.Generator,
    n_per_family: int = 10,
    within_sd: float = 0.12,
    prestige_beta: tuple[float, float] = (2.0, 5.0),
    prototypes: list[Program] | None = None,
) -> list[Program]:
    """Draw `n_per_family` programs around each prototype."""
    if n_per_family < 1:
        raise ValueError("n_per_family must be >= 1")
    protos = prototypes if prototypes is not None else load_prototypes()
    programs: list[Program] = []
    for f, proto in enumerate(protos):
        base = proto.vector
        for i in range(n_per_family):
            vec = np.clip(base + rng.normal(0.0, within_sd, size=base.shape), 0.0, 1.0)
            programs.append(
                replace(
                    proto,
                    program_id=f"sim-{f:02d}-{i:02d}",
                    school=f"Synthetic School {f:02d}-{i:02d}",
                    attributes={k: round(float(v), 4) for k, v in zip(NEED_KEYS, vec, strict=True)},
                    prestige=round(float(rng.beta(*prestige_beta)), 4),
                    synthetic=True,
                    attribute_source="simulated",
                )
            )
    return programs
