"""Make the W-20 rating workbook for one rater (docs/w20-rater-guide.md).

    python scripts/make_w20_rating_sheet.py --rater R2 --out examples/out/w20_R2.xlsx

Programmes and pages come from data/curriculum/w21_pilot_v0.jsonl (only the school,
degree name and URLs; the W-21 scores are never copied into the sheet, so raters stay
blind to them). Read the returned sheets with scripts/w20_agreement.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
DIMS = [
    ("quant_intensity", "數理強度"),
    ("programming_intensity", "程式強度"),
    ("research_orientation", "研究取向"),
    ("entrepreneurship", "創業取向"),
]
SCORES = '"0,0.25,0.5,0.75,1,缺值"'
EVIDENCE = '"直接,缺項"'
RATING_COLS = ["編號", "CIP", "學校", "學程（官網名稱）", "主要網址",
               "其他已讀網址", "維度", "維度代碼",
               "分數", "證據類型", "證據原文（≤15 字）", "證據網址", "備註"]  # fmt: skip
PRACTICE = [
    ("練習 1", "MS in Business Analytics（虛構大學 A）",
     "必修：Statistics for Managers（迴歸模型）、Python for Analytics、SQL Databases、Optimization "
     "Models；畢業專題為企業顧問案。選修：Entrepreneurship。校級育成中心開放所有研究生申請。",
     {"quant_intensity": "0.5", "programming_intensity": "0.75", "research_orientation": "0.25",
      "entrepreneurship": "0.5"},
     "迴歸與最佳化是應用層級；兩門必修程式／資料課；顧問案是實務專案；有創業選修與育成中心"),
    ("練習 2", "MFA in Film（虛構大學 B）",
     "完整必修課表：Directing、Cinematography、Screenwriting、"
     "Editing；必須完成畢業影片（thesis film）。"
     "校級資源頁只有電影學生的提案社團。",
     {"quant_intensity": "0", "programming_intensity": "0", "research_orientation": "0.25",
      "entrepreneurship": "0.25"},
     "完整課表沒有數理與程式課（證據類型「缺項」）；畢業影片是創作型作品，最多 0.25；只有社團"),
    ("練習 3", "MS in Statistics（虛構大學 C）",
     "必修：Probability Theory（測度論證明）、Mathematical Statistics、Statistical Computing in R；"
     "必須完成論文。學程頁與學校網站都找不到創業中心或創業課。",
     {"quant_intensity": "1", "programming_intensity": "0.5", "research_orientation": "1",
      "entrepreneurship": "0"},
     "需證明的機率論；一門 R 程式課；必修論文；兩種頁面都確認沒有創業資源"),
    ("練習 4", "MS in Accounting（虛構大學 D）",
     "頁面只寫「Courses may include: Auditing, Taxation, "
     "Financial Reporting」，找不到完整課表或畢業要求。",
     {"quant_intensity": "缺值", "programming_intensity": "缺值", "research_orientation": "缺值",
      "entrepreneurship": "缺值"},
     "課表不完整時一律缺值，不給 0；創業也沒看校級頁"),
]  # fmt: skip

# Copied from docs/rubric.md §4.1-4.2 (rubric is the source of truth; tests check sync).
ANCHORS = [
    (
        "數理強度",
        "完整必修課表中沒有統計或數學課",
        "必修入門統計、研究方法或資料分析課，但沒有建模",
        "必修應用層級的迴歸、最佳化或統計建模",
        "必修隨機過程、金融數學等進階課，但課程描述看不出需要證明",
        "必修機率論、數學最佳化等需證明的課程",
    ),
    (
        "程式強度",
        "完整必修課表中沒有程式課",
        "必修課使用試算表或套裝軟體做分析，但不寫程式",
        "一門必修程式或資料課（Python、R、SQL）",
        "兩門必修程式或資料課",
        "多門必修軟體或系統課，或以工程實作為主的畢業專題",
    ),
    (
        "研究取向",
        "沒有論文、研究或畢業專題要求",
        "必修非研究型畢業作品：創作（小說、影片、演奏會、展覽）或實務專案（顧問案、專業計畫）",
        "論文或研究專題可選；或必修研究論文以外的研究型報告",
        "必修研究專題或研究報告，但不是論文",
        "必修研究論文；或此碩士本身就是博士學程的一段（例如只在博士途中授予）",
    ),
    (
        "創業取向",
        "官網與校級資源頁都沒有創業資源",
        "只有學生社團或活動（競賽、講座）",
        "有創業選修課，或校級加速器／育成中心開放本學程學生申請",
        "必修一門創業課",
        "課程內建創業專案或育成流程（真實創業，不是模擬）",
    ),
]
RULES = [
    ("一般分數", "附官網原文片段（建議 ≤ 15 個英文字）與網址；行銷用語不得作為證據"),
    ("0 分（沒有）", "證據是完整必修課表或畢業要求頁的網址，證據類型選「缺項」"),
    ("課表不完整",
     "只列部分課程（courses may include、sample curriculum）且找不到完整課表 → 缺值，不給 0"),
    ("創作或實務作品", "研究取向最多 0.25，即使頁面稱為 thesis"),
    ("博士銜接", "只有此碩士本身是博士學程的一段才算 1；學校另有雙學位不提高分數"),
    ("創業取向", "必須同時看學程頁與該校創業中心或育成中心頁；只看學程頁 → 缺值"),
    ("頁面年份", "用最新學年；只能找到 3 年以前的版本時可用，備註寫年份"),
    ("多個方向", "依主要或預設方向評分，備註寫採用哪個方向"),
]  # fmt: skip

HF = Font(name="Arial", bold=True, color="FFFFFF")
HFILL = PatternFill("solid", fgColor="44546A")
IN = PatternFill("solid", fgColor="FFF2CC")
BASE = Font(name="Arial")
WRAP = Alignment(wrap_text=True, vertical="top")


def _header(ws, cols, widths):
    for c, (h, w) in enumerate(zip(cols, widths, strict=True), 1):
        x = ws.cell(row=1, column=c, value=h)
        x.font, x.fill, x.alignment = HF, HFILL, WRAP
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A2"


def build(records: list[dict], rater: str, out: Path) -> None:
    wb = Workbook()
    lg = wb.active
    lg.title = "說明"
    lg.column_dimensions["A"].width = 110
    lines = [
        (f"W-20 學程評分表　評分者代碼：{rater}", True),
        ("先讀評分手冊（docs/w20-rater-guide.md，或隨附的手冊）。黃色儲存格是要填的地方。", False),
        (
            "步驟：1. 做「練習」4 題，對照「練習解答」；2. "
            "依序評「評分」分頁的 30 個學程 × 4 個維度；"
            "3.（選填）填「入學條件」；4. 整份檔案傳回。",
            False,
        ),
        (
            "每個學程從「主要網址」開始讀；需要時可以開同校其他官網頁，但要把網址填在「證據網址」。",
            False,
        ),
        ("不要和其他評分者討論，也不要查其他人的評分；有疑問寫在備註。", False),
        (
            "0 分（沒有某類課）：證據類型選「缺項」，證據網址填完整必修課表的頁面。"
            "課表不完整：選「缺值」。",
            False,
        ),
        ("創業取向必須同時看學程頁與該校創業中心或育成中心頁；只看學程頁就選「缺值」。", False),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = lg.cell(row=i, column=1, value=t)
        c.font, c.alignment = Font(name="Arial", bold=b, size=12 if b else 11), WRAP

    ws = wb.create_sheet("評分手冊")
    _header(ws, ["維度", "0", "0.25", "0.5", "0.75", "1"], [10, 30, 30, 30, 30, 30])
    for r, row in enumerate(ANCHORS, 2):
        for c, v in enumerate(row, 1):
            x = ws.cell(row=r, column=c, value=v)
            x.font, x.alignment = BASE, WRAP
    r0 = len(ANCHORS) + 3
    for c, h in enumerate(["證據規則", "做法"], 1):
        x = ws.cell(row=r0, column=c, value=h)
        x.font, x.fill = HF, HFILL
    for k, (a, b) in enumerate(RULES, r0 + 1):
        ws.cell(row=k, column=1, value=a).font = BASE
        x = ws.cell(row=k, column=2, value=b)
        x.font, x.alignment = BASE, WRAP

    ws = wb.create_sheet("練習")
    _header(
        ws,
        ["練習", "學程", "官網內容（虛構）", "維度", "維度代碼", "你的分數"],
        [8, 30, 60, 10, 22, 10],
    )
    r = 2
    for pid, name, text, _, _ in PRACTICE:
        for key, zh in DIMS:
            for c, v in enumerate([pid, name, text, zh, key, None], 1):
                x = ws.cell(row=r, column=c, value=v)
                x.font, x.alignment = BASE, WRAP
            ws.cell(row=r, column=6).fill = IN
            r += 1
    dv = DataValidation(type="list", formula1=SCORES, allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{r - 1}")

    ws = wb.create_sheet("評分")
    _header(ws, RATING_COLS, [6, 9, 26, 30, 30, 30, 10, 22, 9, 9, 34, 30, 30])
    r = 2
    for i, rec in enumerate(records, 1):
        others = " ".join(u for u in rec.get("urls_fetched", []) if u != rec["program_url"])
        for key, zh in DIMS:
            vals = [f"W{i:02d}", rec["cip"], rec["school"], rec["degree_name"], rec["program_url"],
                    others, zh, key]  # fmt: skip
            for c, v in enumerate(vals, 1):
                x = ws.cell(row=r, column=c, value=v)
                x.font, x.alignment = BASE, WRAP
            for c in range(9, 14):
                x = ws.cell(row=r, column=c)
                x.fill, x.font, x.alignment = IN, BASE, WRAP
            r += 1
    for col, f in (("I", SCORES), ("J", EVIDENCE)):
        dv = DataValidation(type="list", formula1=f, allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{r - 1}")

    ws = wb.create_sheet("入學條件")
    cols = ["編號", "學校", "學程", "必修先修課", "建議先修課", "考試（GRE、GMAT、執照等）",
            "作品集／試演", "面試", "其他條件", "條件頁網址"]  # fmt: skip
    _header(ws, cols, [6, 26, 30, 30, 30, 24, 18, 10, 24, 30])
    for i, rec in enumerate(records, 1):
        for c, v in enumerate([f"W{i:02d}", rec["school"], rec["degree_name"]], 1):
            ws.cell(row=i + 1, column=c, value=v).font = BASE
        for c in range(4, 11):
            ws.cell(row=i + 1, column=c).fill = IN

    ws = wb.create_sheet("練習解答")
    _header(ws, ["練習", "維度代碼", "答案", "理由"], [8, 22, 8, 70])
    r = 2
    for pid, _, _, ans, why in PRACTICE:
        for key, _ in DIMS:
            for c, v in enumerate([pid, key, ans[key], why], 1):
                x = ws.cell(row=r, column=c, value=v)
                x.font, x.alignment = BASE, WRAP
            r += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rater", required=True, help="rater code, e.g. R1 or R2")
    ap.add_argument("--pilot", type=Path, default=ROOT / "data/curriculum/w21_pilot_v0.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    recs = [json.loads(x) for x in args.pilot.read_text(encoding="utf-8").splitlines() if x.strip()]
    build(recs, args.rater, args.out)
    print(f"written: {args.out} ({len(recs)} programmes x {len(DIMS)} dimensions)")


if __name__ == "__main__":
    main()
