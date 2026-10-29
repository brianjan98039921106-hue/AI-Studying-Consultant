# W-20 黃金標準：執行步驟

> 對象：安排 W-20 評分的人。評分者本人只需要評分表（內含評分手冊與練習）。
> 評分準則：[rubric.md](rubric.md) §4（2026-10-28 修訂，v2 只評 4 個維度）。分析規則已預先登記在 [`scripts/w20_agreement.py`](../scripts/w20_agreement.py) 檔頭。

## 1. 評分者

| 項目 | 規則 |
|---|---|
| 人數 | 2 位（R1、R2），彼此獨立；差距 ≥ 0.5 的個案另找第三人裁決 |
| 條件 | 讀得懂美國研究所課程頁；最好是留學顧問或在美國讀過研究所的人 |
| 盲評 | 評分前不可看 [w21-pilot.md](w21-pilot.md) §4、`data/curriculum/w21_pilot_v0.jsonl`，也不可看另一位的評分。開發者擔任 R1 時同樣適用 |
| 工作量 | 練習 4 題約 15 分鐘；30 個學程 × 4 個維度，每個學程約 10–20 分鐘；入學條件分頁選填 |

## 2. 流程

| 步驟 | 指令或動作 |
|---|---|
| 1 | 產生評分表：`python scripts/make_w20_rating_sheet.py --rater R2 --out examples/out/w20_R2.xlsx`（R1 同理） |
| 2 | 寄給評分者；評分表的「評分手冊」分頁與 rubric §4 同步（`test_manual_matches_rubric`） |
| 3 | 收回後放 `data/private/w20/`（不進版控，內含評分者的判斷紀錄） |
| 4 | `python scripts/w20_agreement.py data/private/w20/R1.xlsx data/private/w20/R2.xlsx --pilot data/curriculum/w21_pilot_v0.jsonl --out results/w20-agreement.md` |
| 5 | κ < 0.70 的維度：修訂 rubric §4 錨點後重評；需裁決的個案交第三人，裁決結果與理由寫回 |
| 6 | 黃金標準確定後，依修訂版準則重新擷取 W-21，再跑一次 V-C0-2 |

## 3. 判定（預先登記）

| 指標 | 規則 |
|---|---|
| 評分者一致性 | 每個維度的二次加權 κ ≥ 0.70 才接受；任一人「缺值」的配對不計入 κ，另外回報 |
| 黃金標準 | 兩人差距 ≤ 0.25 取平均；差距 ≥ 0.5 等裁決；其餘不納入 |
| W-21 擷取（V-C0-2） | 與黃金標準差距 ≤ 0.25 視為一致；擷取缺值但黃金標準有分數算不一致；每個維度 ≥ 90% 才通過 |
