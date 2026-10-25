# Roadmap / TODO

預設決策（可改）：Python · MIT · 先服務**顧問** · 側寫先用**結構化問卷** · MVP 領域限定 **商管 × 數據 × 工程管理**。

## M0 — Repo 基礎 ✅
- [x] 專案結構、`pyproject.toml`、ruff、pytest
- [x] GitHub Actions CI（Python 3.11–3.13）
- [x] README、架構文件、驗證計畫、相關研究
- [x] 開發規範（`CONTRIBUTING.md`、`CLAUDE.md`）

## M1 — Program 知識庫 MVP
- [x] Program schema（`schema.py`）與合成資料
- [ ] 下載 NCES CIP 2020 代碼表，建立 `data/reference/cip2020.csv`
- [ ] 從 IPEDS Completions 篩出 MVP 領域、award level 7（碩士）的「學校 × CIP」清單
- [ ] 定義屬性評分準則（每個需求維度 0 / 0.5 / 1 各代表什麼），寫進 `docs/rubric.md`
- [ ] 人工整理 30 個真實 program 作為黃金標準
- [ ] LLM 從官網擷取屬性 → 與黃金標準比對，正確率 ≥ 90% 才擴充到 100–200 個
- [ ] 每筆資料記錄來源網址與擷取日期

## M2 — 側寫引擎
- [x] 錨點拆解（`profiling.py`）
- [ ] 需求維度的信效度檢查：維度是否重疊（相關 > 0.8 就合併）
- [ ] 題庫：每個維度 3–5 題情境題（不出現 program 名稱）
- [ ] 自適應選題：每次挑「最能改變推薦排序」的題目
- [ ] 停止條件：top-k 排序穩定或達 10 題

## M3 — 探索推薦
- [x] 契合度 × 新穎度 + MMR（`discovery.py`）
- [ ] 推薦理由產生：列出貢獻最大的 2–3 個需求維度
- [ ] 新穎度改用程式名稱 / 課程文字的 embedding 距離，補足 CIP 太粗的問題
- [ ] 參數掃描：`novelty_weight`、`diversity_weight` 對 V3、V4 指標的影響

## M4 — 可行性與申請組合
- [x] Kapur greedy + DP + 暴力法驗證（`portfolio.py`）
- [ ] P(admit) 模型：program 層級先驗 + 學生特徵，輸出區間
- [ ] 錄取相關性：Gaussian copula Monte Carlo 評估器，比較獨立 vs 相關假設
- [ ] 穩健版本：p 擾動下仍穩定的組合
- [ ] 多輪次限制：Spring / Fall、截止日、先修課完成時間

## M5 — 顧問模式
- [ ] 側寫報告（網頁 / PDF）：需求雷達圖、推薦清單、推薦理由、申請組合
- [ ] 顧問可手動調整需求向量並即時看到推薦變化

## M6 — 使用者研究
- [ ] 招募 10–20 位台灣申請者 + 2–3 位顧問
- [ ] 同意書與個資處理流程（個資法）
- [ ] 執行 V1–V6（見 `VALIDATION.md`）

## 待決策
- [ ] 公開 / 私人 repo
- [ ] 驗證夥伴：可接觸到的顧問或學生
- [ ] 是否開放其他領域（設計、公共政策、CS）
