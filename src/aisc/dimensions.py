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


# --- v2 (W-16c, docs/dimensions-v2.md) ----------------------------------------------
# The five v1 domain dimensions are replaced by O*NET's 41 specific interest areas
# (Rounds, Putka & Lewis 2022; ratings: Liu et al. 2026). Keys follow
# aisc.catalog.occupation.basic_interest_key; tests check them against
# data/reference/basic_interest_map.csv. Chinese labels are working translations.
_BI_PROGRAM = "C0.2 O*NET Specific Interest Areas (later also C0.3 rubric)"
_BI_STUDENT = "two-stage items: 12 area groups, then areas within chosen groups (W-66)"
_BASIC_INTERESTS: tuple[tuple[str, str, str], ...] = (
    ("bi_mechanics_electronics", "機械／電子", "Mechanics/Electronics"),
    ("bi_construction_woodwork", "營建／木工", "Construction/Woodwork"),
    ("bi_transportation_machine_operation", "運輸／機具操作", "Transportation/Machine Operation"),
    ("bi_physical_manual_labor", "體力勞動", "Physical/Manual Labor"),
    ("bi_protective_service", "保護性服務", "Protective Service"),
    ("bi_agriculture", "農業", "Agriculture"),
    ("bi_nature_outdoors", "自然／戶外", "Nature/Outdoors"),
    ("bi_animal_service", "動物服務", "Animal Service"),
    ("bi_athletics", "運動", "Athletics"),
    ("bi_engineering", "工程", "Engineering"),
    ("bi_physical_science", "物理科學", "Physical Science"),
    ("bi_life_science", "生命科學", "Life Science"),
    ("bi_medical_science", "醫學", "Medical Science"),
    ("bi_social_science", "社會科學", "Social Science"),
    ("bi_humanities", "人文", "Humanities"),
    ("bi_mathematics_statistics", "數學／統計", "Mathematics/Statistics"),
    ("bi_information_technology", "資訊科技", "Information Technology"),
    ("bi_visual_arts", "視覺藝術", "Visual Arts"),
    ("bi_applied_arts_and_design", "應用藝術與設計", "Applied Arts and Design"),
    ("bi_performing_arts", "表演藝術", "Performing Arts"),
    ("bi_music", "音樂", "Music"),
    ("bi_creative_writing", "創意寫作", "Creative Writing"),
    ("bi_media", "媒體", "Media"),
    ("bi_culinary_art", "烹飪藝術", "Culinary Art"),
    ("bi_teaching_education", "教學／教育", "Teaching/Education"),
    ("bi_social_service", "社會服務", "Social Service"),
    ("bi_health_care_service", "醫療照護服務", "Health Care Service"),
    ("bi_religious_activities", "宗教活動", "Religious Activities"),
    ("bi_personal_service", "個人服務", "Personal Service"),
    ("bi_professional_advising", "專業諮詢", "Professional Advising"),
    ("bi_business_initiatives", "商業開創", "Business Initiatives"),
    ("bi_sales", "銷售", "Sales"),
    ("bi_marketing_advertising", "行銷／廣告", "Marketing/Advertising"),
    ("bi_finance", "金融", "Finance"),
    ("bi_accounting", "會計", "Accounting"),
    ("bi_human_resources", "人力資源", "Human Resources"),
    ("bi_office_work", "行政事務", "Office Work"),
    ("bi_management_administration", "管理", "Management/Administration"),
    ("bi_public_speaking", "公開演說", "Public Speaking"),
    ("bi_politics", "政治", "Politics"),
    ("bi_law", "法律", "Law"),
)
NEED_DIMENSIONS_V2: tuple[NeedDimension, ...] = (
    *[d for d in NEED_DIMENSIONS if d.layer in ("interest", "intensity")],
    *[
        NeedDimension(k, "domain", zh, en, _BI_STUDENT, _BI_PROGRAM)
        for k, zh, en in _BASIC_INTERESTS
    ],
    *[d for d in NEED_DIMENSIONS if d.layer == "orientation"],
)
NEED_KEYS_V2: tuple[str, ...] = tuple(d.key for d in NEED_DIMENSIONS_V2)
BASIC_INTEREST_KEYS: tuple[str, ...] = tuple(k for k, _, _ in _BASIC_INTERESTS)

DIMENSION_SETS: dict[str, tuple[NeedDimension, ...]] = {
    "v1": NEED_DIMENSIONS,
    "v2": NEED_DIMENSIONS_V2,
}
DIMS_VERSIONS: tuple[str, ...] = tuple(DIMENSION_SETS)


def dims_keys(dims: str = "v1") -> tuple[str, ...]:
    """Ordered keys of a dimension set ('v1' = 15 dims, 'v2' = 51 dims)."""
    if dims not in DIMENSION_SETS:
        raise ValueError(f"Unknown dimension set {dims!r}; expected one of {DIMS_VERSIONS}")
    return NEED_KEYS if dims == "v1" else NEED_KEYS_V2


def dims_for_length(n: int) -> str:
    """Dimension set of a vector of length n (the two sets differ in size)."""
    for name in DIMS_VERSIONS:
        if len(dims_keys(name)) == n:
            return name
    raise ValueError(f"No dimension set has {n} dimensions")


def keys_in_layer(layer: str, dims: str = "v1") -> tuple[str, ...]:
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r}; expected one of {LAYERS}")
    return tuple(d.key for d in DIMENSION_SETS[dims_name(dims)] if d.layer == layer)


def dims_name(dims: str) -> str:
    dims_keys(dims)  # validates
    return dims


# v1 -> v2 compatibility map for old data only (A-C1-23): NOT a measurement.
V1_DOMAIN_TO_V2: dict[str, tuple[str, ...]] = {
    "domain_finance": ("bi_finance",),
    "domain_tech_product": ("bi_information_technology", "bi_engineering"),
    "domain_operations": ("bi_management_administration",),
    "domain_design": ("bi_applied_arts_and_design",),
    "domain_policy_social": ("bi_politics", "bi_social_science"),
}


def migrate_needs_v1_to_v2(values: dict[str, float]) -> dict[str, float]:
    """Translate a v1 {dimension: score} dict to v2 keys (A-C1-23). Shared layers are
    copied; each v1 domain value is copied to its mapped areas (max when several v1
    domains map to the same area). Areas with no v1 source are left out, so callers
    must complete them (e.g. with the v2 prior, dimensions-v2 §3)."""
    unknown = set(values) - set(NEED_KEYS)
    if unknown:
        raise ValueError(f"not v1 dimensions: {sorted(unknown)}")
    out = {k: float(v) for k, v in values.items() if k in NEED_KEYS_V2}
    for v1, targets in V1_DOMAIN_TO_V2.items():
        if v1 in values:
            for t in targets:
                out[t] = max(out.get(t, 0.0), float(values[v1]))
    return out
