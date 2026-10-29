"""C1 — Student profiling.

Sub-components (docs/components/C1-profiling.md):
    C1.1 measurement model (dimensions, item bank, reliability)   — dimensions v1 done;
         CEEC inventory prior (ceec.py) done; item bank planned
    C1.2 anchor decomposition                                     — v0 (reference, failed L3);
         v1 anchors as prior for unanswered dims + prestige widening
    C1.3 adaptive questionnaire (adaptive.py)                     — v1: 4 item-selection strategies
    C1.4 familiarity elicitation (what the student already knows) — v1 done (familiarity.py)
    C6.4 student-record intake (record_intake.py, W-61)           — abilities -> StudentRecord;
         never passed to C2 (D9)
"""

from aisc.profiling.adaptive import (
    NOISE_MODELS,
    STRATEGIES,
    AdaptiveSession,
    Item,
    cosine_scores,
    default_item_bank,
)
from aisc.profiling.anchors import (
    anchor_prior,
    blend_profile,
    blend_profile_v1,
    complete_neutral,
    needs_from_anchors,
)
from aisc.profiling.ceec import RiasecPrior, apply_prior, riasec_prior_from_ceec
from aisc.profiling.familiarity import LEVELS, STRICT_KNOWN, apply_familiarity, known_ids
from aisc.profiling.record_intake import (
    CourseMap,
    IntakeResult,
    RecordIntakeError,
    intake_student_record,
    load_course_map,
    load_student_record,
    normalize_name,
)

__all__ = [
    "AdaptiveSession",
    "CourseMap",
    "IntakeResult",
    "RecordIntakeError",
    "intake_student_record",
    "load_course_map",
    "load_student_record",
    "normalize_name",
    "Item",
    "NOISE_MODELS",
    "STRATEGIES",
    "cosine_scores",
    "default_item_bank",
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
