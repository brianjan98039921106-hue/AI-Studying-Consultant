# AI Studying Consultant

> 側寫台灣留學生的真實需求，推薦他們沒想過、但更契合的美國研究所 program，並列出先修與申請條件差距。（錄取機率與申請組合暫停，介面保留：[D13](docs/ARCHITECTURE.md)）
>
> Profile Taiwanese applicants, surface US graduate programs they never considered, and optimise which ones to apply to.

<!-- AUTO:version START · 由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡 -->
**Status:** v0.4.6 · research prototype · program attributes used in simulations are **synthetic**; reference tables in `data/reference/` (CIP 2020, IPEDS 2025, O*NET) are real and sourced.
<!-- AUTO:version END -->

**範圍：** 服務所有台灣申請者，不限就學背景與目標領域（D10–D12，[docs/expansion.md](docs/expansion.md)）；資料分階段上線，目前為 v1（商管 × 數據 × 工程管理）。

## 要解決的問題

台灣學生對研究所的理解，多半建立在台灣有限的學程分類上。要找到更適合的美國 program，需要**自我洞察**和**資訊接觸**，但兩者互為前提：不知道有哪些 program，就很難描述需求；描述不出需求，就搜尋不到新的 program。結果是學生在預設的學校清單裡打轉。

這個專案把它當成**冷啟動推薦 + 探索**問題，拆成七個可以獨立開發、獨立驗證的部件：

| 部件 | 功能 |
|---|---|
| C0 Program 知識庫 | CIP 分類骨架 + 兩種獨立來源測量學程屬性 |
| C1 學生側寫 | 以 RIASEC 為骨架的需求測量、錨點拆解、自適應提問、熟悉度 |
| C2 探索推薦 | 依契合排序，再依學生回報的熟悉度分成「可能沒想過」與「已知且適合」兩區 |
| C3 錄取可行性（⏸ 暫停，D13） | 階層貝氏模型 + 選擇偏誤校正 + 校準 |
| C4 申請組合（⏸ 暫停，D13） | Kapur & Hong 精確解 + 相關性評估 + 穩健選擇 |
| C5 模擬與評估 | 已知真相的合成學生，驗證所有部件 |
| C6 先修與申請條件差距 | 顯示學程要求什麼、學生還缺什麼；只呈現不過濾 |

> 以下「目前狀態」「最近更新」「下一步」「最新驗證結論」由 `scripts/update_readme.py` 從來源檔自動產生；合併到 main 後 GitHub Actions 會自動同步，**請修改來源檔，不要直接改 README 的這些區塊**。

## 目前狀態

<!-- AUTO:status START · 由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡 -->
來源：[docs/USAGE.md §1](docs/USAGE.md#1-現在能用什麼)（各部件的驗證關卡見 [docs/VALIDATION.md §5](docs/VALIDATION.md)）

| 功能 | 部件 | 狀態 | 可以用在真實學生身上嗎？ |
|---|---|---|---|
| 輸入需求（15 個維度）＋ 未作答維度自動補全 | C1.1、C1.2 | ✅ 可用 | ⚠️ 維度尚未經實證驗證（W-41） |
| 大考中心興趣量表分數作為先驗 | C1.1 | ✅ 可用 | ⚠️ 參數未校準（A-C1-8） |
| 自適應問卷（自動挑下一題） | C1.3 | 🟡 可用，題目為抽象占位題 | ❌ 真實題目文字待 W-41 |
| 學生自填問卷（73 題真題目）＋終端機自助流程 | C1.1、C1.2 | 🧪 可用（§4.12）：全答、可跳題、v2 剖面；`examples/self_service.py` 從同意到報告 | ⚠️ 題目尚未試答（W-41）；推薦目錄是真實目錄 v0（強度與取向未測量） |
| 熟悉度確認（「你聽過這些學程嗎？」） | C1.4 | ✅ 可用 | ⚠️ 漏報／誤報率待實測 |
| 分區推薦：「你可能沒想過」＋「已知且適合」 | C2 v1 | ✅ L3 合成驗證通過 | ❌ 學程目錄目前是**合成資料** |
| 真實學程目錄 v0（IPEDS 碩士 × 職業興趣） | C0（W-77） | 🧪 實驗中（§4.11）：17,552 個真實學程（v2 維度）；強度與取向**尚未測量** | ⚠️ 只依興趣與領域排序；同一學門的學校分數相同 |
| 先修與申請條件差距卡 | C6 v0 | ✅ 比對邏輯可用；報告附差距卡（官網原文＋學生紀錄並列） | ❌ 尚無真實學程條件資料（W-20） |
| 學經歷輸入（學位、修課、考試、經驗 → `StudentRecord`） | C1、C6.4 | ✅ 可用（JSON 輸入、欄位檢查、台灣課名對應表 draft-1） | ⚠️ 課名對應表待使用者確認；只比對「有沒有修／考」，不比對最低分數（B-C6-12） |
| 申請組合最佳化（選哪幾間申請） | C4.1–C4.4 | ⏸ 暫停（D13）：程式與介面保留（`plan_applications`、`AdmitEstimate`），不在主線流程（§9.1） | — |
| 需求維度 v2（41 個跨領域興趣領域、兩階段提問的先驗） | C1.1、C1.2、C2.1 | 🧪 實驗中，可選用（`dims="v2"`）；預設仍為 v1 | ❌ L3 未通過（W-16c，F71）：領域不符下降，但暫定計分器下推薦品質低於 v1 |
| 錄取機率估計 | C3 | ⏸ 暫停（D13）：v0.2 模型、資料載入與校準工具保留，只在合成資料驗證過（§9.2） | — |
| 學生自助網頁 | Phase 5（W-53） | 🧪 v0 可用（§4.13）：瀏覽器內計算、不上傳；結果與 Python 模型逐項比對一致 | ⚠️ 題目尚未試答（W-41）；目錄強度與取向未測量 |
| 顧問介面／側寫報告 | Phase 5 | 🟡 報告 v0 可用（Markdown，§4.10）；介面未開始 | ⚠️ 會自動標示合成資料與補全的維度 |

**一句話：** 目前可以完整跑一次「需求 → 推薦 → 條件差距 → 報告」流程來理解模型怎麼運作，但學程資料是虛構的，**還不能拿結果給真實學生做決定**。錄取機率與申請組合暫停（D13）。本專案定位為**學生自助工具**（D14），問卷真題目與網頁介面開發中。
<!-- AUTO:status END -->

## 最近更新

<!-- AUTO:changelog START · 由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡 -->
來源：[docs/USAGE.md §8](docs/USAGE.md#8-變更紀錄使用方式)（最近 3 筆，新到舊）

| 日期 | 版本 | 改變 | 工項 |
|---|---|---|---|
| 2026-10-31 | v0.4.6 | 網頁新增選填的試用回饋（每個推薦的評分、剖面吻合度；複製後交回，網頁不儲存；預留 `window.AISC_FEEDBACK_SINK` 給日後的回饋資料庫）與評分腳本 `scripts/score_feedback.py`（§4.13）；新增人物設定基準工具 `scripts/run_persona_benchmark.py` | W-80、W-81 |
| 2026-10-29 | v0.4.5 | 新增學生自助網頁（`web/index.html`、`web/engine.js`、`web/data.json`；`scripts/build_web_data.py` 匯出資料與打包單一檔案）：作答到結果都在瀏覽器內，不上傳（§4.13） | W-53 |
| 2026-10-29 | v0.4.4 | 新增學生自填問卷（`aisc.profiling.questionnaire`，73 題真題目、可跳題）與終端機自助流程（`examples/self_service.py`）；新增 `pipeline.merge_identical_programs` 與 `to_type_ids`，真實目錄的清單改列不同學位類型（§4.12） | W-52、B-C2-30 |
<!-- AUTO:changelog END -->

## 下一步

<!-- AUTO:next-steps START · 由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡 -->
來源：[docs/ROADMAP.md](docs/ROADMAP.md) （2026-10-31 更新：學生自助工具，D14；C3、C4 暫停，D13）

| 順序 | 工項 | 依據 |
|---|---|---|
| 1 | **W-80** 人物設定基準：你寫 P01–P16 的預期答案、一位顧問獨立寫 P02、P06、P09、P13（[persona-benchmark.md](docs/persona-benchmark.md)），提交後執行 | 不需要學生就能衡量推薦品質，並擋住之後的退步（V-C2-9） |
| 2 | **W-40** 個資告知填入提供者與聯絡方式 → **W-82** 試用 3–10 人（含 W-41 題目試答；回饋用網頁的「複製回饋」，評分用 `score_feedback.py`） | 真人才能回答「適合又沒想過」（V-C2-10） |
| 3 | **W-77** 真實學程目錄 v0 補強度與取向（W-21；W-20 可由單一審核者人工檢查，D14） | 目前只依興趣與領域排序；W-80 與 W-82 的結果會指出最需要補的學門 |
| 4 | **W-83** 試用後 2–4 週追蹤 | 行為證據 |
<!-- AUTO:next-steps END -->

## 文件導覽

| 想知道 | 看這裡 |
|---|---|
| **怎麼使用（學生、顧問、開發者）** | [docs/USAGE.md](docs/USAGE.md) |
| 整體架構、設計原則、部件圖 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 每個部件的規格 | [docs/components/](docs/components/) |
| 每個方法的理論來源與假設 | [docs/THEORY.md](docs/THEORY.md) |
| 已知假設與演算法盲區 | [docs/REGISTER.md](docs/REGISTER.md) |
| 驗證階梯、關卡、指標門檻 | [docs/VALIDATION.md](docs/VALIDATION.md) |
| 工項與順序 | [docs/ROADMAP.md](docs/ROADMAP.md) |
| 需求維度與評分準則 | [docs/rubric.md](docs/rubric.md) |
| **所有發現的狀態總表** | [results/FINDINGS.md](results/FINDINGS.md) |

<!-- AUTO:latest-results START · 由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡 -->
最新 L3 合成驗證結論（來源：[docs/VALIDATION.md §5](docs/VALIDATION.md)）：[W-66 兩階段 m／q](results/synthetic-l3-two-stage-findings.md)、[C3 年度效果](results/synthetic-l3-admission-years-findings.md)、[C3 學程類型](results/synthetic-l3-admission-types-findings.md)、[C3 v0.2 確認](results/synthetic-l3-admission-pool-confirm-findings.md)、[V-C3-1 自陳偏誤](results/synthetic-l3-admission-bias-findings.md)、[C3 → C4](results/synthetic-l3-admission-portfolio-findings.md)、[V-C4-3 穩健選擇](results/synthetic-l3-portfolio-robust-findings.md)、[V-C4-2 錄取相關](results/synthetic-l3-portfolio-corr-findings.md)、[族群多樣化](results/synthetic-l3-diversity-findings.md)、[領域把關](results/synthetic-l3-domain-gate-findings.md)、[v2 計分器選擇](results/synthetic-l3-v2-scorer-findings.md)、[需求維度 v1 vs v2](results/synthetic-l3-dims-v2-findings.md)、[C1.3 v2 雜訊模型](results/synthetic-l3-noise-findings.md)、[端到端](results/synthetic-l3-e2e-findings.md)、[C1.3 自適應問卷](results/synthetic-l3-adaptive-findings.md)、[C1.2 v1](results/synthetic-l3-c1-v1-findings.md)、[v2：熟悉度擷取＋分區呈現](results/synthetic-l3-v2-findings.md)（前幾版：[跨生成模型](results/synthetic-l3-v1-continuous-findings.md)、[v1](results/synthetic-l3-v1-findings.md)、[v0](results/synthetic-l3-v0-findings.md)）
<!-- AUTO:latest-results END -->

## 使用方式

**完整使用說明：[docs/USAGE.md](docs/USAGE.md)**（給學生、顧問與開發者；隨開發更新，範例由 CI 實際執行）。

| 你是 | 從這裡開始 |
|---|---|
| 想先看看效果 | 安裝後執行 `python examples/quickstart.py`（[USAGE §3](docs/USAGE.md#3-最快上手跑示範不用寫程式)） |
| 留學顧問 | [USAGE §5 建議的使用流程](docs/USAGE.md#5-給顧問建議的使用流程目前階段) |
| 要把模型接進自己的工具 | [USAGE §4 逐步 API](docs/USAGE.md#4-以-python-使用逐步) |
| 想知道現在能不能用在真實學生 | [USAGE §1 現在能用什麼](docs/USAGE.md#1-現在能用什麼)：**還不行**，學程資料目前是合成的 |

### 安裝與示範

```bash
pip install -e ".[dev]"
python examples/quickstart.py
```

示範輸出分兩區：**「你可能沒想過」**（契合但學生沒聽過的學程）與**「你已經知道、而且確實適合」**；最後寫出側寫報告（`examples/out/demo_report.md`）。錄取機率與申請組合暫停（D13），示範不再計算。

### 開發者指令

```bash
pytest -rx                   # xfail = 已知盲區探針（預期中的失敗）
ruff check . && ruff format --check .
python scripts/run_l3_baseline.py --tag v2 --needs-model archetype    # L3，約 6 分鐘
python scripts/run_l3_baseline.py --tag v2-continuous --needs-model continuous
python scripts/compare_l3.py results/synthetic-l3-v2.md results/synthetic-l3-v2-continuous.md
python scripts/run_l3_e2e.py           # 端到端：問卷 → 補全 → 熟悉度 → 分區，約 6 分鐘
```

貢獻流程見 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 驗證方式一覽

每個部件的每個版本都要通過六層驗證：**L1 理論符合 → L2 程式驗證 → L3 合成還原 → L4 盲區探針 → L5 實證 → L6 外部審查**。
盲區探針以 `@pytest.mark.probe` + `xfail(strict=True, raises=AssertionError)` 寫成測試：只有行為斷言失敗才算「預期中的失敗」，程式錯誤不會被掩蓋；盲區修好時測試轉為通過，CI 會要求同步移除標記並更新登錄表。`tests/test_docs_consistency.py` 另外檢查文件之間是否互相矛盾。

## License

MIT © Po-An (Brian) Chan

程式碼與文件採 MIT 授權。`data/reference/` 中以 O\*NET 31.0 Database（USDOL/ETA，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）資料修改產生的檔案，仍依其原授權條款使用；USDOL/ETA 未審核、背書或測試這些修改。O\*NET® 為 USDOL/ETA 的商標。詳細來源與修改說明見 [`data/README.md`](data/README.md#third-party-data-and-attribution)。
