"""Need dimensions v1: the shared attribute space for students and programs.

Students are profiled on *needs* (what they want to do and learn), not on
program names. Programs are scored on the same dimensions so the two can be
compared directly. Every dimension is scored on [0, 1].

Four layers (docs/rubric.md, approved W-16a):
    interest     Holland RIASEC — validated construct, links to O*NET (C0.2)
                 on the program side and to the CEEC interest inventory
                 (大考中心興趣量表) on the student side
    intensity    how deep in maths / programming the student *wants* to go
                 (preference, not ability: B-C1-8)
    domain       subject area
    orientation  research vs. industry path, entrepreneurship

Status: theory-anchored but not yet empirically validated. Overlaps
pre-registered in docs/rubric.md §2 are tested by V-C1-7 (B-C1-1).
Changing this list means updating the rubric, synthetic data and tests together.
"""

from __future__ import annotations

from dataclasses import dataclass

LAYERS: tuple[str, ...] = ("interest", "intensity", "domain", "orientation")


@dataclass(frozen=True)
class NeedDimension:
    key: str
    layer: str
    label_zh: str
    label_en: str
    student_source: str  # how C1 measures it
    program_source: str  # how C0 measures it


_RIASEC_STUDENT = "CEEC inventory score (optional prior) + short items"
_RIASEC_PROGRAM = "C0.2 O*NET occupational interest profile"
_RUBRIC = "C0.3 curriculum rubric (docs/rubric.md §4)"

NEED_DIMENSIONS: tuple[NeedDimension, ...] = (
    NeedDimension("riasec_r", "interest", "實用型", "Realistic", _RIASEC_STUDENT, _RIASEC_PROGRAM),
    NeedDimension(
        "riasec_i", "interest", "研究型", "Investigative", _RIASEC_STUDENT, _RIASEC_PROGRAM
    ),
    NeedDimension("riasec_a", "interest", "藝術型", "Artistic", _RIASEC_STUDENT, _RIASEC_PROGRAM),
    NeedDimension("riasec_s", "interest", "社會型", "Social", _RIASEC_STUDENT, _RIASEC_PROGRAM),
    NeedDimension(
        "riasec_e", "interest", "企業型", "Enterprising", _RIASEC_STUDENT, _RIASEC_PROGRAM
    ),
    NeedDimension(
        "riasec_c", "interest", "事務型", "Conventional", _RIASEC_STUDENT, _RIASEC_PROGRAM
    ),
    NeedDimension(
        "quant_intensity", "intensity", "數理強度", "Quantitative intensity",
        "behaviourally anchored items", _RUBRIC,
    ),
    NeedDimension(
        "programming_intensity", "intensity", "程式強度", "Programming intensity",
        "behaviourally anchored items", _RUBRIC,
    ),
    NeedDimension(
        "domain_finance", "domain", "金融", "Finance", "paired scenario items", _RUBRIC
    ),
    NeedDimension(
        "domain_tech_product", "domain", "科技產品", "Technology / product",
        "paired scenario items", _RUBRIC,
    ),
    NeedDimension(
        "domain_operations", "domain", "營運／供應鏈", "Operations / supply chain",
        "paired scenario items", _RUBRIC,
    ),
    NeedDimension(
        "domain_design", "domain", "設計／使用者經驗", "Design / UX",
        "paired scenario items", _RUBRIC,
    ),
    NeedDimension(
        "domain_policy_social", "domain", "公共政策／社會", "Public policy / social",
        "paired scenario items", _RUBRIC,
    ),
    NeedDimension(
        "research_orientation", "orientation", "研究取向", "Research orientation",
        "scenario items", _RUBRIC,
    ),
    NeedDimension(
        "entrepreneurship", "orientation", "創業取向", "Entrepreneurship",
        "scenario items", _RUBRIC,
    ),
)  # fmt: skip

NEED_KEYS: tuple[str, ...] = tuple(d.key for d in NEED_DIMENSIONS)
N_NEEDS: int = len(NEED_KEYS)
RIASEC_KEYS: tuple[str, ...] = tuple(d.key for d in NEED_DIMENSIONS if d.layer == "interest")


def keys_in_layer(layer: str) -> tuple[str, ...]:
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r}; expected one of {LAYERS}")
    return tuple(d.key for d in NEED_DIMENSIONS if d.layer == layer)
