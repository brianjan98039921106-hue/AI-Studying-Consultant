"""Need dimensions: the shared attribute space for students and programs.

Students are profiled on *needs* (what they want to do and learn), not on
program names, because Taiwanese applicants usually do not know the US
program taxonomy. Programs are scored on the same dimensions, so a student's
need vector and a program's attribute vector can be compared directly.

Each dimension is scored on [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeedDimension:
    key: str
    label_zh: str
    label_en: str
    low_anchor: str  # what a score of 0 means
    high_anchor: str  # what a score of 1 means


NEED_DIMENSIONS: tuple[NeedDimension, ...] = (
    NeedDimension(
        "quant", "量化強度", "Quantitative intensity", "幾乎不用數學", "大量機率、最佳化、統計推論"
    ),
    NeedDimension(
        "programming", "程式強度", "Programming intensity", "不寫程式", "以寫程式與系統實作為主"
    ),
    NeedDimension(
        "business", "商業導向", "Business orientation", "與商業決策無關", "以商業決策與管理為核心"
    ),
    NeedDimension(
        "finance", "金融領域", "Finance domain", "與金融無關", "以金融市場、投資、風險為主"
    ),
    NeedDimension(
        "tech_product",
        "科技產品",
        "Tech / product domain",
        "與科技產品無關",
        "以科技產品與工程團隊為主",
    ),
    NeedDimension(
        "design", "設計導向", "Design orientation", "不涉及設計", "以設計方法、使用者研究為核心"
    ),
    NeedDimension(
        "research",
        "研究導向",
        "Research orientation",
        "純實務、職涯導向",
        "以研究、論文、博士銜接為主",
    ),
    NeedDimension(
        "entrepreneurship",
        "創業導向",
        "Entrepreneurship",
        "沒有創業資源",
        "有創業課程、孵化器、新創網絡",
    ),
    NeedDimension(
        "policy_social",
        "公共與社會",
        "Policy / social impact",
        "與公共議題無關",
        "以政策、公共部門、社會影響為主",
    ),
)

NEED_KEYS: tuple[str, ...] = tuple(d.key for d in NEED_DIMENSIONS)
N_NEEDS: int = len(NEED_KEYS)
