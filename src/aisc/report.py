"""W-50 v0: the profile report a consultant reads with the student (Markdown, zh-TW).

System layer, like pipeline.py: it only formats the contracts other components
produce and changes nothing (D9: requirement gaps are shown next to a
recommendation, never used to filter or reorder). Every number that rests on an
assumption or on synthetic data is labelled as such:
    - programmes with `synthetic=True` -> a banner saying the catalogue is not real
    - AdmitEstimate(method="assumed")  -> probabilities and expected utility marked
      as assumptions (C3 not used)
    - dimensions the student did not answer -> shown as completed by C1.2, with SD
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from aisc.contracts import DiscoveryResult, Program, Recommendation, RequirementGap, StudentProfile
from aisc.dimensions import DIMENSION_SETS, LAYERS

LAYER_ZH = {"interest": "興趣", "intensity": "強度", "domain": "領域", "orientation": "取向"}


def _label(key: str, dims: str) -> str:
    for d in DIMENSION_SETS[dims]:
        if d.key == key:
            return d.label_zh
    return key


def _reasons(rec: Recommendation, dims: str, n: int = 3) -> str:
    top = sorted(rec.contributions.items(), key=lambda kv: -kv[1])[:n]
    return "、".join(f"{_label(k, dims)}（{v:+.2f}）" for k, v in top if v > 0) or "—"


def _gap_cell(gap: RequirementGap | None) -> str:
    if gap is None:
        return "—"
    parts = []
    if gap.missing_required:
        parts.append(f"缺必修 {len(gap.missing_required)}")
    extra = len(gap.missing) - len(gap.missing_required)
    if extra:
        parts.append(f"缺建議 {extra}")
    if gap.planned:
        parts.append(f"規劃中 {len(gap.planned)}")
    if gap.unknown:
        parts.append(f"無法判讀 {len(gap.unknown)}")
    if gap.none_published:
        parts.append("官網未列條件")
    if gap.before_target:
        parts.append(f"申請前須完成 {len(gap.before_target)}")
    return "；".join(parts) or "條件皆已符合"


def _rec_table(recs, dims, gaps) -> list[str]:
    if not recs:
        return ["（沒有符合條件的學程）", ""]
    rows = [
        "| # | 學程 | 學校 | CIP | 契合度 | 主要理由（貢獻） | 條件差距（不影響排序） |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(recs, 1):
        p = r.program
        gap = (gaps or {}).get(p.program_id)
        rows.append(f"| {i} | {p.degree_name} | {p.school} | {p.cip} | {r.relevance:.2f} | "
                    f"{_reasons(r, dims)} | {_gap_cell(gap)} |")  # fmt: skip
    return [*rows, ""]


def render_profile_report(
    profile: StudentProfile,
    completed: StudentProfile,
    result: DiscoveryResult,
    gaps: Mapping[str, RequirementGap] | None = None,
    today: date | None = None,
) -> str:
    """Markdown report. `profile` is what the student answered, `completed` the
    profile actually used for ranking (pipeline.prepare_profile or the C1.3
    posterior). Core only (D13): needs, recommendations and requirement gaps. The
    application-planning section is a future extension, rendered separately by
    `aisc.application.render_plan_section`."""
    if profile.student_id != completed.student_id or profile.dims != completed.dims:
        raise ValueError("profile and completed profile must be the same student and dims")
    completed.require_complete("report")
    dims = completed.dims
    shown: list[Program] = [r.program for r in (*result.discoveries, *result.known_good)]
    L = [f"# 需求側寫報告：{profile.student_id}", "",
         f"產生日期：{(today or date.today()).isoformat()} · 需求維度 {dims}", ""]  # fmt: skip
    warn = []
    if any(p.synthetic for p in shown):
        warn.append("**學程資料是合成的示範資料，不是真實學程，不能用來做決定。**")
    if any(k not in profile.needs for k in completed.needs):
        warn.append("部分需求維度學生沒有作答，由系統依先驗補全（下表標示「補全」與不確定性）。")
    if warn:
        L += ["> " + w for w in warn] + [""]

    L += [
        "## 1. 需求剖面",
        "",
        "| 層 | 維度 | 數值 | 來源 | 不確定性（SD） |",
        "|---|---|---|---|---|",
    ]
    for layer in LAYERS:
        for d in DIMENSION_SETS[dims]:
            if d.layer != layer:
                continue
            v = completed.needs[d.key]
            src = "作答" if d.key in profile.needs else "補全"
            sd = completed.uncertainty.get(d.key)
            L.append(f"| {LAYER_ZH[layer]} | {d.label_zh} | {v:.2f} | {src} | "
                     f"{'—' if sd is None else f'{sd:.2f}'} |")  # fmt: skip
    L += ["", "## 2. 你可能沒想過的學程", "",
          "依契合度排序，且學生回報沒聽過；"
          "理由是各維度對契合度的精確貢獻（加總即為契合度）。", ""]  # fmt: skip
    L += _rec_table(result.discoveries, dims, gaps)
    L += ["## 3. 已知且適合的學程", ""]
    L += _rec_table(result.known_good, dims, gaps)
    if gaps:
        L += ["條件差距只顯示「還缺什麼」，**不會**影響上面的排序（D9）；"
              "「官網未列條件」不代表沒有門檻。", ""]  # fmt: skip
    L += ["## 4. 限制", "",
          "- 需求維度尚未經實證驗證（W-41）；推薦只依需求，能力與條件不影響排序。",
          "- 契合度是模型估計，不是錄取或滿意度的保證。",
          "- 本報告不估計錄取機率、不建議申請組合（未來擴充）。", ""]  # fmt: skip
    return "\n".join(L)
