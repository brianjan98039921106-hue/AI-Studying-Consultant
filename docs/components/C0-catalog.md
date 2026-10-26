# C0 Program 知識庫

**目的**：建立美國研究所學程的結構化資料，每個 program 有屬性向量 $x_p$、CIP 標籤與實務欄位，並且每個屬性都能追溯來源。
**輸出契約**：`contracts.Program`（`attribute_source`、`source_url`、`retrieved_at` 必填於真實資料）
**MVP 範圍**：商管 × 數據 × 工程管理相關 CIP 族群的碩士學程（award level 7）

## 子部件

| ID | 功能 | 方法 | 輸出 | 版本路徑 |
|---|---|---|---|---|
| C0.1 | 分類骨架 | 下載 CIP 2020 代碼表與 IPEDS Completions，篩出 MVP 族群 × 碩士 | 「學校 × CIP」清單 | v1：MVP 族群；v2：全領域 |
| C0.2 | 職業橋接 | CIP → SOC（NCES/BLS crosswalk）→ O*NET 職業興趣（RIASEC）；多對多以等權或就業人數加權平均 | $x^{occ}_p$（只涵蓋 RIASEC 維度） | v1 等權；v2 就業加權 |
| C0.3 | 課程擷取 | 評分準則 + LLM 從官網擷取，每個分數須附原文片段 | $x^{cur}_p$（全部維度） | v1 人工黃金標準 30 筆；v2 LLM 擴充 |
| C0.4 | 融合與品管 | 比對 $x^{occ}$ 與 $x^{cur}$ 的對應維度；不一致超過門檻者進人工審查佇列 | 最終 $x_p$ + 品質旗標 | v1 規則；v2 以評分者信度加權 |

**為什麼要雙重測量（D2）**：C0.2 不受行銷文案影響，但太粗（B-C0-1、B-C0-2）；C0.3 細緻，但會受文案誇大與 LLM 幻覺影響（B-C0-3）。兩者偏誤來源不同，一致時可信度高，不一致時正好指出要人工查的地方。

## 理論與假設
A-C0-1 … A-C0-5（見 [REGISTER](../REGISTER.md#c0-program-知識庫)）；來源見 [THEORY](../THEORY.md#c0-program-知識庫)。

## 盲區
B-C0-1 CIP 太粗 · B-C0-2 對照表是規範性的 · B-C0-3 幻覺與誇大 · B-C0-4 資料完整度偏誤 · B-C0-5 過時

## 驗證
| 層級 | 內容 | 指標 |
|---|---|---|
| L2 | schema、來源欄位、id 唯一、分數範圍 | V-C0-1 ✅ |
| L3 | 在 C5 中注入「文案誇大」雜訊，檢查 C0.4 是否能標記 | 標記召回率 |
| L4 | 探針：同 CIP 不同性質的學程應得到不同 $x_p$ | B-C0-1 |
| L5 | 黃金標準一致率、評分者 κ、收斂效度 | V-C0-2 … V-C0-5 |

## 實作狀態
| 子部件 | 程式 | 狀態 |
|---|---|---|
| C0.1 | `catalog/reference.py`、`scripts/build_reference.py`；範圍 `data/reference/mvp_scope.json` | 程式與 L2 ✅；官方檔案到位後產生 `data/reference/` |
| C0.2 v1 | `catalog/occupation.py`（等權、OI 1–7 → [0, 1]） | 程式與 L2 ✅；B-C0-1 探針（同 CIP 無法區分） |

## 工項
W-14、W-15、W-20、W-21、W-22（見 [ROADMAP](../ROADMAP.md)）
