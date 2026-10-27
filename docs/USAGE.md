# 使用說明 User Guide

> 給**使用者**看的文件：學生、留學顧問，以及想把本模型接進自己工具的開發者。
> 設計原理與驗證細節不在這裡，請看 [ARCHITECTURE.md](ARCHITECTURE.md) 與 [VALIDATION.md](VALIDATION.md)。
>
> **本文件隨開發更新**（工項 W-05）：任何改變使用方式的 PR 都要同步修改本文件與 §8 變更紀錄。
> 本文件中的每一段 `python` 範例都由 `tests/test_usage_doc.py` 在 CI 中實際執行，API 改了而文件沒改，CI 會失敗。

最後更新：2026-10-28 · 對應版本 v0.3.10（研究原型）

---

## 1. 現在能用什麼

| 功能 | 部件 | 狀態 | 可以用在真實學生身上嗎？ |
|---|---|---|---|
| 輸入需求（15 個維度）＋ 未作答維度自動補全 | C1.1、C1.2 | ✅ 可用 | ⚠️ 維度尚未經實證驗證（W-41） |
| 大考中心興趣量表分數作為先驗 | C1.1 | ✅ 可用 | ⚠️ 參數未校準（A-C1-8） |
| 自適應問卷（自動挑下一題） | C1.3 | 🟡 可用，題目為抽象占位題 | ❌ 真實題目文字待 W-41 |
| 熟悉度確認（「你聽過這些學程嗎？」） | C1.4 | ✅ 可用 | ⚠️ 漏報／誤報率待實測 |
| 分區推薦：「你可能沒想過」＋「已知且適合」 | C2 v1 | ✅ L3 合成驗證通過 | ❌ 學程目錄目前是**合成資料** |
| 先修與申請條件差距卡 | C6 v0 | ✅ 比對邏輯可用 | ❌ 尚無真實學程條件資料（W-20） |
| 申請組合最佳化（選哪幾間申請） | C4.1–C4.3 | ✅ 精確解；相關錄取（C4.2）與機率區間的穩健選擇（C4.3）可選用 | ⚠️ 錄取機率與相關係數都須自行假設（C3 未完成） |
| 需求維度 v2（41 個跨領域興趣領域、兩階段提問的先驗） | C1.1、C1.2、C2.1 | 🧪 實驗中，可選用（`dims="v2"`）；預設仍為 v1 | ❌ L3 未通過（W-16c，F71）：領域不符下降，但暫定計分器下推薦品質低於 v1 |
| 錄取機率估計 | C3 | 🧪 v0 模型可用（§4.9）：階層 logistic、區間、自陳偏誤校正；只在合成資料驗證 | ❌ 尚無真實申請結果資料；區間在某些情況過窄（B-C3-8） |
| 顧問介面／側寫報告 | Phase 5 | ⬜ 未開始 | — |

**一句話：** 目前可以完整跑一次「需求 → 推薦 → 條件差距 → 申請組合」流程來理解模型怎麼運作，但學程資料是虛構的，**還不能拿結果給真實學生做決定**。

---

## 2. 安裝

需要 Python ≥ 3.11。

```bash
git clone https://github.com/brianjan98039921106-hue/AI-Studying-Consultant.git
cd AI-Studying-Consultant
pip install -e ".[dev]"
```

確認安裝成功：

```bash
pytest -q                      # 全部通過；xfail 是「已知盲區」的預期失敗
python examples/quickstart.py  # 印出一位示範學生的推薦結果
```

---

## 3. 最快上手：跑示範（不用寫程式）

```bash
python examples/quickstart.py
```

示範學生心中只想到「MS Finance」，喜歡數學、會一點程式。輸出分四段：

| 輸出區段 | 意思 | 怎麼讀 |
|---|---|---|
| **你可能沒想過 (discoveries)** | 學生沒回報聽過、但需求契合度高的學程 | 這是本專案的核心產出；`fit` 越接近 1 越契合 |
| **你已經知道、而且確實適合 (known & good fit)** | 學生聽過、而且確實契合的學程 | 確認原本的方向沒錯，不是新發現 |
| Reference: C2 v0, REJECTED | 舊版演算法，僅供對照 | 不要使用；保留是為了比較 |
| Portfolio for N applications | 在假設錄取機率下，最佳的 N 個申請組合 | 錄取機率是**假設值**，只示範計算方式 |

想換成自己的情境：打開 `examples/quickstart.py`，修改 `student` 的 `needs`（見 §4.2 的維度表）、`anchors`（心中已有的學程）與 `ALREADY_KNOWS`（聽過的學程），重新執行即可。

---

## 4. 以 Python 使用（逐步）

以下範例依序執行，後面的步驟會用到前面的變數。工作目錄須為 repo 根目錄。

### 4.1 載入學程目錄

```python
from pathlib import Path

from aisc.catalog import load_programs

catalog = load_programs(Path("data/synthetic/programs.jsonl"))
by_id = {p.program_id: p for p in catalog}
print(len(catalog), "programs; synthetic =", all(p.synthetic for p in catalog))
```

目前只有 12 筆虛構學程（`"synthetic": true`、學校名稱為占位）。真實學程資料上線後會改用 `data/programs/`，每筆都附 `source_url` 與 `retrieved_at`。

### 4.2 描述學生需求

需求是「想做什麼、想學什麼」，**不是**能力，也不是學程名稱。每個維度填 0–1；**不確定的維度不要填**，系統會依錨點補全，而不是當成 0（沒興趣）。

| 層 | 維度 key | 中文 | 0 代表 | 1 代表 |
|---|---|---|---|---|
| 興趣（RIASEC） | `riasec_r` | 實用型 | 不想動手操作實物 | 很想動手、操作工具與設備 |
| | `riasec_i` | 研究型 | 不想分析、探究問題 | 很想分析、解謎、研究 |
| | `riasec_a` | 藝術型 | 不想做創作 | 很想創作、表達 |
| | `riasec_s` | 社會型 | 不想以助人、教學為主 | 很想助人、教學、諮詢 |
| | `riasec_e` | 企業型 | 不想帶領、說服他人 | 很想帶領、說服、經營 |
| | `riasec_c` | 事務型 | 不想處理結構化事務 | 很想處理資料、流程、規則 |
| 強度 | `quant_intensity` | 數理強度 | 想避開數學 | 想深入數學 |
| | `programming_intensity` | 程式強度 | 想避開寫程式 | 想大量寫程式 |
| 領域 | `domain_finance` | 金融 | 沒興趣 | 很有興趣 |
| | `domain_tech_product` | 科技產品 | 沒興趣 | 很有興趣 |
| | `domain_operations` | 營運／供應鏈 | 沒興趣 | 很有興趣 |
| | `domain_design` | 設計／使用者經驗 | 沒興趣 | 很有興趣 |
| | `domain_policy_social` | 公共政策／社會 | 沒興趣 | 很有興趣 |
| 取向 | `research_orientation` | 研究取向 | 想走業界 | 想走研究、博士 |
| | `entrepreneurship` | 創業取向 | 不想創業 | 想創業 |

領域層目前只有 MVP 的 5 個領域；跨領域分類由 W-64 取代（見 [expansion.md](expansion.md)）。完整評分準則見 [rubric.md](rubric.md)。

```python
from aisc.contracts import StudentProfile

student = StudentProfile(
    student_id="demo",
    needs={
        "quant_intensity": 0.8,
        "domain_finance": 0.7,
        "programming_intensity": 0.5,
    },
    anchors=["syn-01"],  # 心中已有的學程（program_id），可留空
)
print("unanswered:", student.missing_dimensions)
```

### 4.3（選用）加入大考中心興趣量表分數

高中做過大考中心興趣量表的話，六型分數（0–99）可以當作 RIASEC 的先驗。時間越久、區分值越低，先驗越不確定；之後的作答會蓋過它。

```python
from aisc.profiling import apply_prior, riasec_prior_from_ceec

prior = riasec_prior_from_ceec(
    {"R": 30, "I": 80, "A": 25, "S": 40, "E": 70, "C": 65},
    years_since_test=6,
    differentiation=5,  # 區分值；< 3 視為代碼不明確
)
student = apply_prior(student, prior)
```

### 4.4（選用）自適應問卷

系統每次挑「最能改變推薦結果」的下一題。`answer` 是收集回答的函式：正式產品中是介面，這裡用固定值示範。回答範圍 0–1。

```python
from aisc.pipeline import run_questionnaire


def answer(item):
    # 題目目前是抽象占位題（item.dimension 告訴你在問哪個維度）；真實題目文字待 W-41
    scripted = {"riasec_i": 0.8, "domain_operations": 0.7, "research_orientation": 0.2}
    return scripted.get(item.dimension, 0.5)


student = run_questionnaire(student, catalog, answer, budget=8)
```

### 4.5 熟悉度確認 + 分區推薦

系統先挑出契合的短名單，問學生每個學程聽過沒有，再把結果分成兩區。`ask` 收到 program_id 清單，回傳每個學程的熟悉程度：`"unknown"`（沒聽過）、`"heard"`（聽過名字）、`"can_describe"`（能說出內容）。

```python
from aisc.pipeline import interactive_discovery


def ask(program_ids):
    already_knows = {"syn-12", "syn-04"}  # 正式產品中由學生在介面勾選
    return {pid: ("heard" if pid in already_knows else "unknown") for pid in program_ids}


result = interactive_discovery(student, catalog, ask, n_shortlist=8, k_new=5, k_known=3)
for r in result.discoveries:
    print("可能沒想過:", r.program.degree_name, round(r.relevance, 2))
for r in result.known_good:
    print("已知且適合:", r.program.degree_name, round(r.relevance, 2))
```

| 參數 | 意思 | 預設 |
|---|---|---|
| `n_shortlist` | 熟悉度確認要問幾個學程 | 25 |
| `k_new` | 「可能沒想過」最多列幾個 | 10 |
| `k_known` | 「已知且適合」最多列幾個 | 5 |
| `min_relevance` | 契合度下限 | 0.0 |
| `diversity` | 同一 CIP 族群（前 4 碼）每多一個就扣這麼多分，讓推薦分散到不同族群；模擬中會降低推薦的相關性，**不建議使用**（[F124–F127](../results/synthetic-l3-diversity-findings.md)） | 0.0 |

心中的錨點學程不會出現在任一區。

### 4.6 先修與申請條件差距

條件差距**只呈現、不過濾**：推薦清單不會因為學生缺先修而刪掉或降低排序（設計原則 D9）。差距可以補，學生也不一定馬上申請。

```python
from aisc.contracts import RequirementItem, StudentRecord
from aisc.requirements import compute_gaps, load_taxonomy

taxonomy = load_taxonomy("data/reference/requirement_taxonomy.json")
record = StudentRecord(
    student_id="demo",
    completed={"calculus_1": "微積分（一）"},
    planned={"calculus_2": "2027 春季"},
    target_apply_by=None,  # 只在探索、尚未決定申請時間就留 None
)
# 示範用虛構條件；真實條件須引用官網原文並附來源（W-20）
src = {"source_url": "https://example.edu/syn-03", "retrieved_at": "2026-10-27"}
items = [
    RequirementItem(
        program_id="syn-03",
        kind="course",
        category="calculus_1",
        level="required",
        text="One semester of calculus",
        **src,
    ),
    RequirementItem(
        program_id="syn-03",
        kind="course",
        category="calculus_2",
        level="required",
        text="Multivariable calculus",
        **src,
    ),
    RequirementItem(
        program_id="syn-03",
        kind="test",
        category="gre",
        level="recommended",
        text="GRE recommended",
        **src,
    ),
]
gaps = compute_gaps(["syn-03"], items, record, taxonomy)
gap = gaps["syn-03"]
print("已完成:", [i.category for i in gap.met])  # ['calculus_1']
print("已規劃:", [i.category for i in gap.planned])  # ['calculus_2']
print("尚缺:", [i.category for i in gap.missing])  # ['gre']（建議，非必修）
```

| 差距欄位 | 意思 |
|---|---|
| `met` | 已完成 |
| `planned` | 已規劃 |
| `missing` | 尚未完成（`missing_required` 只列必修） |
| `unknown` | 條件無法對應到類別，**不會**當作已符合 |
| `none_published` | 官網沒列條件；不代表沒有門檻 |
| `before_target` | 學生設了 `target_apply_by` 才會出現：申請前必須補、錄取後不能補的項目 |

可用的條件類別見 `data/reference/requirement_taxonomy.json`（課程、考試、作品集、經驗、證照、語言等）。

### 4.7 申請組合

給定每個學程的錄取機率與效用，找出在申請數量（或預算）限制下期望效用最高的組合。**C3 錄取模型尚未完成**，錄取機率必須以 `AdmitEstimate(method="assumed")` 明確標示為假設。

```python
from aisc.contracts import AdmitEstimate
from aisc.portfolio import dp_varied_cost

shortlist_ = result.discoveries + result.known_good
assumed = [
    AdmitEstimate(r.program.program_id, mean=0.4, low=0.25, high=0.55, method="assumed")
    for r in shortlist_
]
utility = [round(100 * r.relevance) for r in shortlist_]
chosen, value = dp_varied_cost([a.mean for a in assumed], utility, [1] * len(assumed), budget=3)
print([shortlist_[j].program.degree_name for j in chosen], round(value, 1))
```

上面的期望效用假設**各校錄取互相獨立**。實際上同一份申請在各校的結果是相關的（整體競爭力是共同因素），獨立假設會讓期望效用偏樂觀（合成情境中 ρ = 0.4 時高估 3–10%，ρ = 0.8 時 9–23%），但選出的組合通常只差 1% 左右（[F109–F112](../results/synthetic-l3-portfolio-corr-findings.md)），所以預設不變。想看敏感度時可用 C4.2（ρ 是假設值，不是估計值）：

```python
import numpy as np

from aisc.portfolio import admission_draws, portfolio_value_mc, recommend_portfolio

p = [a.mean for a in assumed]
for rho in (0.0, 0.4, 0.8):
    draws = admission_draws(p, rho, 20_000, np.random.default_rng(0))
    print(rho, round(portfolio_value_mc(utility, draws, chosen), 1))  # 同一個組合，相關越高越低
best_if_correlated = recommend_portfolio(p, utility, [1] * len(p), budget=3, rho=0.4)
```

錄取機率本身也可能估錯。合成情境中，±20% 的誤差只讓點估計選出的組合平均少 1% 以內，「最壞情況」與「最小化最大遺憾」兩種穩健準則都沒有更好（[F120–F123](../results/synthetic-l3-portfolio-robust-findings.md)），所以預設仍用點估計。需要比較時：

```python
from aisc.portfolio import minimax_regret_portfolio, worst_case_portfolio

low, high = [a.low for a in assumed], [a.high for a in assumed]
cautious = worst_case_portfolio(low, utility, [1] * len(p), budget=3)  # 全部取下限
balanced = minimax_regret_portfolio(low, high, p, utility, [1] * len(p), budget=3)
```

### 4.8（實驗中）需求維度 v2：涵蓋所有領域

v1 的 5 個領域維度只涵蓋商管 × 數據。v2 改用 O\*NET 的 41 個細分興趣領域（`bi_*`），並以 12 個領域群做兩階段提問（規格見 [dimensions-v2.md](dimensions-v2.md)）。**模擬驗證未通過，預設仍是 v1**：v2 能減少「領域不對」的推薦，但目前的暫定計分器讓整體推薦品質比 v1 差（[F70–F71](../results/synthetic-l3-dims-v2-findings.md)）。請只用來試驗，不要用於實際給學生的推薦。另有實驗用的 `aisc.discovery.domain_gated(tau, lam)`（v1 計分 ＋ 遠領域扣分），同樣未通過模擬驗證（[F81](../results/synthetic-l3-domain-gate-findings.md)），只供研究比較。

```python
from aisc.contracts import StudentProfile
from aisc.dimensions import migrate_needs_v1_to_v2
from aisc.profiling import complete_neutral
from aisc.profiling.hierarchy import GROUP_LABELS

# 第一層：學生對 12 個領域群的興趣（0–1）；第二層：只細問感興趣的群
v2 = StudentProfile(
    "demo-v2",
    {"riasec_i": 0.8, "bi_medical_science": 0.9},  # 第二層有作答的領域
    dims="v2",
    group_scores={6: 0.8, 1: 0.2},  # 6 = 生命科學、醫學與醫療照護
)
full = complete_neutral(v2)  # 未作答的領域：依領域群先驗；群也沒答 → 低基準值，不是 0.5
print(GROUP_LABELS[6][0], round(full.needs["bi_life_science"], 2), round(full.needs["bi_music"], 2))

# 舊的 v1 需求可轉換（僅供相容，不是測量）
print(sorted(migrate_needs_v1_to_v2({"domain_finance": 0.9}).items()))
```

v2 的學生只能和 v2 的學程比較（`Program(..., dims="v2")`）；混用會直接報錯。

可選用：`complete_neutral(v2, area_model="correlated")`（`pipeline.prepare_profile` 同名參數）以 O\*NET 學到的領域相關結構補全未作答領域。用職業資料檢驗時誤差大幅下降，答得越少越明顯，但未通過全部預先登記條件，所以不是預設（[F105–F107](../results/interest-area-imputation-findings.md)）。

### 4.9（實驗中）錄取機率模型 C3.2 v0

有歷年申請結果時，可以用階層 logistic 模型估計錄取機率與區間。學生特徵要先以該學程的申請者母體標準化（平均約 0）；自陳資料（如網路上的申請結果分享）標為 `"self_report"`，顧問的完整紀錄標為 `"complete"`。提供官方錄取率時，模型會校正「錄取者較常自陳」的偏誤（合成資料中 RMSE 從 0.13–0.26 降到 0.04–0.07，[F128–F131](../results/synthetic-l3-admission-bias-findings.md)）。**目前沒有真實申請結果資料**，而且當較強的申請者也較常自陳時，區間會過窄（B-C3-8），所以結果仍要當作假設值使用。

```python
from aisc.feasibility import AdmissionRecord, fit_admission_model

records = [
    AdmissionRecord("prog-a", "52.13", (0.8, 0.1), 1, "self_report"),
    AdmissionRecord("prog-a", "52.13", (-0.5, 0.3), 0, "self_report"),
    AdmissionRecord("prog-b", "52.13", (1.2, -0.4), 1, "complete"),
    AdmissionRecord("prog-b", "52.13", (-1.0, 0.2), 0, "complete"),
]
model = fit_admission_model(records, official_rates={"prog-a": 0.3, "prog-b": 0.4})
est = model.predict("prog-a", (0.5, 0.0))  # AdmitEstimate：mean、low、high（90%）
print(model.method, round(est.low, 2), round(est.mean, 2), round(est.high, 2))
new = model.predict("prog-c", (0.5, 0.0), family="52.13")  # 沒有資料的學程：用族群平均，區間較寬
```

---

## 5. 給顧問：建議的使用流程（目前階段）

| 步驟 | 做什麼 | 對應 |
|---|---|---|
| 1 | 和學生談**想做什麼**，不談學校或學程名稱；把有把握的維度填進 `needs` | §4.2 |
| 2 | 有大考中心興趣量表結果就加入 | §4.3 |
| 3 | 記下學生心中已有的學程作為 `anchors` | §4.2 |
| 4 | 執行熟悉度確認，讓學生自己回答聽過哪些 | §4.5 |
| 5 | 和學生一起看「可能沒想過」清單，討論每一個為什麼契合 | §4.5 |
| 6 | 對有興趣的學程看條件差距，規劃補修時程 | §4.6 |
| 7 | 決定申請時，再討論申請組合 | §4.7 |

目前請把它當作**討論工具**而非結論：學程資料是合成的，錄取機率是假設的。

---

## 6. 限制與注意事項

| 限制 | 影響 | 何時解決 |
|---|---|---|
| 學程目錄是合成資料 | 推薦結果不對應真實學校 | W-20～W-22（C0 真實學程） |
| 推薦用的學程目錄只涵蓋商管 × 數據 × 工程管理（合成） | 其他領域的學生推薦品質未知；全領域的官方參考表（`data/reference/*_all.csv`：1,233 個 CIP，含專業與研究型博士）已建好，但還不是可推薦的學程；v2 維度（§4.8）實驗中 | W-16c、W-20～W-22 |
| 程式內的自適應問卷仍是抽象占位題；v2 題庫中文草案（`data/items/item_bank_v2.json`，73 題）尚未回譯與認知訪談 | 還不能直接拿給學生作答 | W-41 |
| 錄取機率須自行假設 | 申請組合只是示範計算 | W-44（C3） |
| 需求維度尚未實證驗證 | 維度可能重疊或遺漏 | W-41、W-73 |

已知假設與盲區的完整清單見 [REGISTER.md](REGISTER.md)。

**個資：** 真實學生資料只能放在 `data/private/`（不會被 commit），收集須取得書面同意（見 [CONTRIBUTING.md](../CONTRIBUTING.md)）。

---

## 7. 常見問題

| 問題 | 回答 |
|---|---|
| 為什麼不直接過濾掉學生條件不夠的學程？ | 條件可以補，學生也不一定馬上申請；過濾會讓學生再次只看到自己已知的範圍（D9）。 |
| 沒填的維度會被當成沒興趣嗎？ | 不會。`pipeline` 會先用錨點補全未作答維度（B-C1-14）；已作答的維度維持原值。 |
| 「可能沒想過」裡出現學生其實聽過的學程？ | 熟悉度是學生自己回報的，可能漏報；重新執行熟悉度確認即可。 |
| 可以只拿推薦、不跑申請組合嗎？ | 可以。§4.5 就是完整的推薦輸出。 |
| 我的使用方式和本文件不同，或範例跑不起來？ | 請開 issue；本文件的範例由 CI 實際執行，跑不起來代表文件需要更新。 |

---

## 8. 變更紀錄（使用方式）

只記錄**影響使用者的改變**；演算法內部改動見 [results/FINDINGS.md](../results/FINDINGS.md) 與 [ROADMAP.md](ROADMAP.md)。

| 日期 | 版本 | 改變 | 工項 |
|---|---|---|---|
| 2026-10-27 | v0.2 | 首版使用說明：安裝、示範、逐步 API、顧問流程、限制；範例納入 CI 執行 | W-05 |
| 2026-10-27 | v0.3 | 新增實驗中的需求維度 v2（`dims="v2"`、`group_scores`、`migrate_needs_v1_to_v2`），預設仍為 v1（§4.8） | W-16c |
| 2026-10-27 | v0.3.1 | v2 的 L3 結果：未通過，v2 推薦品質目前低於 v1，請勿用於實際推薦（§1、§4.8）；用法沒有改變 | W-16c |
| 2026-10-27 | v0.3.2 | 新增實驗用計分器 `domain_gated`（v2 作答 → v1 計分 ＋ 遠領域扣分）；L3 未通過，不建議使用（§4.8） | W-16c |
| 2026-10-28 | v0.3.3 | 新增全領域官方參考表（`data/reference/*_all.csv`，範圍 `all_scope.json`）：1,209 個 CIP、含專業／研究型博士與研究所證書；推薦目錄不變（§6） | W-65、W-70 |
| 2026-10-28 | v0.3.4 | 新增 v2 題庫中文草案（`data/items/item_bank_v2.json`，取自 O\*NET 31.0 代表活動；[item-bank-v2.md](item-bank-v2.md)），尚未驗證、不可施測（§6） | W-41 |
| 2026-10-28 | v0.3.5 | 輸入驗證更嚴格：需求與學程分數不得為 NaN；`uncertainty` 須為正數；`dp_varied_cost` 接受整數值的浮點成本；`evaluate_lists` 新增 `hidden` 參數，全領域學生未傳時報錯 | 程式稽核 |
| 2026-10-28 | v0.3.6 | 更正全領域參考表：CIP 代碼表的「Moved to／Moved from」原本讀反，補回 24 個現行代碼（DO、DVM 等），共 1,233 個 CIP（v0.3.3 寫的 1,209 有誤） | B-C0-15 |
| 2026-10-28 | v0.3.7 | v2 新增可選用的相關補全 `area_model="correlated"`（§4.8）；全領域參考表新增就業加權版本（參考用，非預設） | 架構檢討、W-71 |
| 2026-10-28 | v0.3.8 | 申請組合新增 C4.2（錄取相關：`admission_draws`、`portfolio_value_mc`、`recommend_portfolio`）與 C4.3（機率區間的穩健選擇：`worst_case_portfolio`、`minimax_regret_portfolio`）；預設選擇不變（§4.7） | W-29、W-2A |
| 2026-10-28 | v0.3.9 | 分區推薦新增可選參數 `diversity`（族群多樣化）；模擬中降低相關性，預設 0、不建議使用（§4.5） | B-C2-9 |
| 2026-10-28 | v0.3.10 | 新增實驗中的錄取機率模型 C3.2 v0（`aisc.feasibility.fit_admission_model`）：階層 logistic、90% 區間、以官方錄取率校正自陳偏誤（§4.9） | W-44 |
