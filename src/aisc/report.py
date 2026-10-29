"""W-50 v0: the profile report a consultant reads with the student (Markdown, zh-TW).

System layer, like pipeline.py: it only formats the contracts other components
produce and changes nothing (D9: requirement gaps are shown next to a
recommendation, never used to filter or reorder). Every number that rests on an
assumption or on synthetic data is labelled as such:
    - programmes with `synthetic=True` -> a banner saying the catalogue is not real
    - programmes with unmeasured dimensions (real catalogue v0, B-C0-18) -> a banner,
      and those dimensions are never cited as reasons (A-C0-15)
    - profiles from the low-confidence series fallback -> a banner
    - dimensions the student did not answer -> shown as completed by C1.2, with SD
Gap cards (W-61) quote each requirement verbatim with its source and, when the
StudentRecord is given, the student's own evidence next to it; course names the
intake could not map are listed for manual mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from aisc.contracts import (
    DiscoveryResult,
    Program,
    Recommendation,
    RequirementGap,
    RequirementItem,
    StudentProfile,
    StudentRecord,
)
from aisc.dimensions import DIMENSION_SETS, LAYERS

LAYER_ZH = {"interest": "興趣", "intensity": "強度", "domain": "領域", "orientation": "取向"}


def _label(key: str, dims: str) -> str:
    for d in DIMENSION_SETS[dims]:
        if d.key == key:
            return d.label_zh
    return key


REASON_MIN_NEED = 0.5  # cite only needs above the neutral value (B-C2-34)


def reason_keys(rec: Recommendation, needs: Mapping[str, float], n: int = 3) -> list[str]:
    """Dimensions shown as "why it fits": the largest positive contributions among
    dimensions the student wants (completed value > 0.5, B-C2-34), excluding the
    programme's unmeasured placeholders (A-C0-15). web/engine.js uses the same rule."""
    skip = set(rec.program.unmeasured)
    items = [(k, v) for k, v in rec.contributions.items()
             if k not in skip and v > 0 and needs.get(k, 0.0) > REASON_MIN_NEED]  # fmt: skip
    return [k for k, _ in sorted(items, key=lambda kv: -kv[1])[:n]]


def _reasons(rec: Recommendation, dims: str, needs: Mapping[str, float], n: int = 3) -> str:
    keys = reason_keys(rec, needs, n)
    return "、".join(f"{_label(k, dims)}（{rec.contributions[k]:+.2f}）" for k in keys) or "—"


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


def _rec_table(recs, dims, gaps, needs) -> list[str]:
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
                    f"{_reasons(r, dims, needs)} | {_gap_cell(gap)} |")  # fmt: skip
    return [*rows, ""]


def _item_line(it: RequirementItem) -> str:
    cat = it.category or "（無對應類別）"
    return f"{cat}：「{it.text}」（[官網]({it.source_url})，{it.retrieved_at}）"


def _gap_cards(shown: list[Program], gaps, record: StudentRecord | None) -> list[str]:
    L = ["## 4. 條件差距卡", "",
         "每張卡引用官網原文。條件差距只呈現、不影響排序（D9）。"
         "考試與成績只比對「有沒有」，**不比對分數是否達到官網最低要求**，"
         "請把下方你的紀錄與原文自行對照。", ""]  # fmt: skip
    evidence = record.completed if record else {}
    plans = record.planned if record else {}
    online = record.online_only if record else set()
    for p in shown:
        gap = gaps.get(p.program_id)
        if gap is None:
            continue
        L += [f"### {p.degree_name}（{p.school}）", ""]
        if gap.none_published:
            L += ["- 官網未列條件：不代表沒有門檻。", ""]
            continue
        for it in gap.missing:
            lvl = "必修" if it.level == "required" else "建議"
            L.append(f"- 缺少（{lvl}）：{_item_line(it)}")
        for it in gap.before_target:
            L.append(f"- 申請前須完成（錄取後不能補）：{_item_line(it)}")
        for it in gap.planned:
            mine = plans.get(it.category or "")
            L.append(f"- 規劃中：{_item_line(it)}" + (f" — 你的規劃：{mine}" if mine else ""))
        for it in gap.met:
            mine = evidence.get(it.category or "")
            note = ""
            if it.category in online and it.accepts_online is None:
                note = "（僅線上修課；官網未載明是否接受線上學分）"
            L.append(f"- 已符合：{_item_line(it)}" + (f" — 你的紀錄：{mine}{note}" if mine else ""))
        for it in gap.unknown:
            L.append(f"- 無法判讀（不當作已符合）：「{it.text}」（[官網]({it.source_url})）")
        L.append("")
    if record and record.unmapped:
        L += ["### 待人工對應的課程與考試", "",
              "以下名稱沒有自動對應到條件類別（對應表 draft，待確認），"
              "所以在上面的卡片中**不會**算成已符合；請和顧問確認後在輸入檔加上 `categories`。", "",
              "| 名稱 | 原因 |", "|---|---|"]  # fmt: skip
        L += [f"| {n} | {why} |" for n, why in record.unmapped.items()] + [""]
    return L


def render_profile_report(
    profile: StudentProfile,
    completed: StudentProfile,
    result: DiscoveryResult,
    gaps: Mapping[str, RequirementGap] | None = None,
    plan=None,
    today: date | None = None,
    record: StudentRecord | None = None,
) -> str:
    """Markdown report. `profile` is what the student answered, `completed` the
    profile actually used for ranking (pipeline.prepare_profile or the C1.3
    posterior); `plan` is an optional pipeline.ApplicationPlan (application mode);
    `record` (optional, with `gaps`) adds the student's evidence to the gap cards
    and lists course names that still need manual mapping."""
    if profile.student_id != completed.student_id or profile.dims != completed.dims:
        raise ValueError("profile and completed profile must be the same student and dims")
    if record is not None and record.student_id != profile.student_id:
        raise ValueError("record belongs to another student")
    completed.require_complete("report")
    dims = completed.dims
    shown: list[Program] = [r.program for r in (*result.discoveries, *result.known_good)]
    L = [f"# 需求側寫報告：{profile.student_id}", "",
         f"產生日期：{(today or date.today()).isoformat()} · 需求維度 {dims}", ""]  # fmt: skip
    warn = []
    if any(p.synthetic for p in shown):
        warn.append("**學程資料是合成的示範資料，不是真實學程，不能用來做決定。**")
    if plan is not None and plan.assumed:
        warn.append("錄取機率是**假設值**（未使用 C3 模型），申請組合與期望效用只供示範。")
    gap_dims = sorted({k for p in shown for k in p.unmeasured})
    if gap_dims:
        names = "、".join(_label(k, dims) for k in gap_dims)
        warn.append(f"學程的{names}**尚未測量**（只用職業興趣資料），這些維度不影響排序，"
                    "也不列為理由；同一學門的學程目前分數相同。")  # fmt: skip
    if any(p.attribute_source == "occupation_fallback" for p in shown):
        warn.append("部分學程的學門沒有職業對照，興趣剖面取自同系列學門的平均（低信心）。")
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
    L += _rec_table(result.discoveries, dims, gaps, completed.needs)
    L += ["## 3. 已知且適合的學程", ""]
    L += _rec_table(result.known_good, dims, gaps, completed.needs)
    if gaps:
        L += ["條件差距只顯示「還缺什麼」，**不會**影響上面的排序（D9）；"
              "「官網未列條件」不代表沒有門檻。", ""]  # fmt: skip
        L += _gap_cards(shown, gaps, record)
    if plan is not None:
        names = {p.program_id: f"{p.degree_name}（{p.school}）" for p in shown}
        L += ["## 5. 申請組合（申請模式）", "",
              f"- 方法：{plan.result.method}",
              "- 建議申請：" + "、".join(names.get(i, i) for i in plan.chosen_ids),
              "- 期望效用（各校錄取互相獨立的假設下）："
              f"{plan.result.expected_utility:.1f}"]  # fmt: skip
        corr = {r: v for r, v in plan.value_if_correlated.items() if r > 0}
        if corr:
            L.append("- 若各校錄取結果相關（同一份申請的整體競爭力是共同因素）："
                     + "、".join(f"ρ = {r}：{v:.1f}" for r, v in corr.items())
                     + "（相關越高越低；真實 ρ 未知）")  # fmt: skip
        L.append("")
    L += ["## 6. 限制", "",
          "- 需求維度尚未經實證驗證（W-41）；推薦只依需求，能力與條件不影響排序。",
          "- 契合度是模型估計，不是錄取或滿意度的保證。", ""]  # fmt: skip
    return "\n".join(L)
