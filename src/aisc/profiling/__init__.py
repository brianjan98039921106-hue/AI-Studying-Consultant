"""C1 — Student profiling.

Sub-components (docs/components/C1-profiling.md):
    C1.1 measurement model (dimensions, item bank, reliability)   — dimensions v1 done;
         CEEC inventory prior (ceec.py) done; item bank planned
    C1.2 anchor decomposition                                     — v0 (reference, failed L3);
         v1 anchors as prior for unanswered dims + prestige widening
    C1.3 adaptive elicitation (value-of-information item choice)  — planned
    C1.4 familiarity elicitation (what the student already knows) — v1 done (familiarity.py)
"""

from aisc.profiling.anchors import (
    anchor_prior,
    blend_profile,
    blend_profile_v1,
    complete_neutral,
    needs_from_anchors,
)
from aisc.profiling.ceec import RiasecPrior, apply_prior, riasec_prior_from_ceec
from aisc.profiling.familiarity import LEVELS, STRICT_KNOWN, apply_familiarity, known_ids

__all__ = [
    "LEVELS",
    "RiasecPrior",
    "STRICT_KNOWN",
    "anchor_prior",
    "apply_familiarity",
    "apply_prior",
    "blend_profile",
    "blend_profile_v1",
    "complete_neutral",
    "known_ids",
    "needs_from_anchors",
    "riasec_prior_from_ceec",
]
