# 使用說明 User Guide

> 給**使用者**看的文件：學生、留學顧問，以及想把本模型接進自己工具的開發者。
> 設計原理與驗證細節不在這裡，請看 [ARCHITECTURE.md](ARCHITECTURE.md) 與 [VALIDATION.md](VALIDATION.md)。
>
> **本文件隨開發更新**（工項 W-05）：任何改變使用方式的 PR 都要同步修改本文件與 §8 變更紀錄。
> 本文件中的每一段 `python` 範例都由 `tests/test_usage_doc.py` 在 CI 中實際執行，API 改了而文件沒改，CI 會失敗。

最後更新：2026-10-29 · 對應版本 v0.4.3（研究原型）

---

## 1. 現在能用什麼

| 功能 | 部件 | 狀態 | 可以用在真實學生身上嗎？ |
|---|---|---|---|
| 輸入需求（15 個維度）＋ 未作答維度自動補全 | C1.1、C1.2 | ✅ 可用 | ⚠️ 維度尚未經實證驗證（W-41） |
| 大考中心興趣量表分數作為先驗 | C1.1 | ✅ 可用 | ⚠️ 參數未校準（A-C1-8） |
| 自適應問卷（自動挑下一題） | C1.3 | 🟡 可用，題目為抽象占位題 | ❌ 真實題目文字待 W-41 |
| 熟悉度確認（「你聽過這些學程嗎？」） | C1.4 | ✅ 可用 | ⚠️ 漏報／誤報率待實測 |
| 分區推薦：「你可能沒想過」＋「已知且適合」 | C2 v1 | ✅ L3 合成驗證通過 | ❌ 學程目錄目前是**合成資料** |
| 真實學程目錄 v0（IPEDS 碩士 × 職業興趣） | C0（W-77） | 🧪 實驗中（§4.11）：17,552 個真實學程（v2 維度）；強度與取向**尚未測量** | ⚠️ 只依興趣與領域排序；同一學門的學校分數相同 |
| 先修與申請條件差距卡 | C6 v0 | ✅ 比對邏輯可用；報告附差距卡（官網原文＋學生紀錄並列） | ❌ 尚無真實學程條件資料（W-20） |
| 學經歷輸入（學位、修課、考試、經驗 → `StudentRecord`） | C1、C6.4 | ✅ 可用（JSON 輸入、欄位檢查、台灣課名對應表 draft-1） | ⚠️ 課名對應表待使用者確認；只比對「有沒有修／考」，不比對最低分數（B-C6-12） |
| 申請組合最佳化（選哪幾間申請） | C4.1–C4.4 | ⏸ 暫停（D13）：程式與介面保留（`plan_applications`、`AdmitEstimate`），不在主線流程（§9.1） | — |
| 需求維度 v2（41 個跨領域興趣領域、兩階段提問的先驗） | C1.1、C1.2、C2.1 | 🧪 實驗中，可選用（`dims="v2"`）；預設仍為 v1 | ❌ L3 未通過（W-16c，F71）：領域不符下降，但暫定計分器下推薦品質低於 v1 |
| 錄取機率估計 | C3 | ⏸ 暫停（D13）：v0.2 模型、資料載入與校準工具保留，只在合成資料驗證過（§9.2） | — |
| 顧問介面／側寫報告 | Phase 5 | 🟡 報告 v0 可用（Markdown，§4.10）；介面未開始 | ⚠️ 會自動標示合成資料與補全的維度 |

**一句話：** 目前可以完整跑一次「需求 → 推薦 → 條件差距 → 報告」流程來理解模型怎麼運作，但學程資料是虛構的，**還不能拿結果給真實學生做決定**。錄取機率與申請組合暫停（D13）。本專案定位為**學生自助工具**（D14），問卷真題目與網頁介面開發中。

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

需求是「想做什麼、想學什麼」，**不是**能力，也不是學程名稱。每個維度填 0–1；**不確定的維度不要填**，系統會依錨點補全，而不是當成 0（沒興趣）。維度名稱拼錯或數值不在 0–1 會直接報錯（不會被默默改掉）；每個作答預設的不確定性是 0.25（相當於一題的作答雜訊，未校準）。

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

高中做過大考中心興趣量表的話，六型分數（0–99）可以當作 RIASEC 的先驗。時間越久、區分值越低，先驗越不確定。學生之後的作答會與它**依不確定性加權合併**：剛做完的量表（標準差 0.15）比單一作答（0.25）更有份量；量表做得越久，作答的比重越大（例如 6 年前的量表標準差 0.45）。

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

硬條件（`StudentProfile(..., hard_constraints={...})`）目前只支援三個鍵：`spring_intake`（True／False，要春季入學）、`stem_opt`（True／False，要 STEM OPT）、`max_tuition_usd`（學費上限，美元）。其他鍵或錯誤型別會直接報錯；學程沒有該資訊時不會被排除（避免偏向資料齊全的學程）。自適應問卷只為「可推薦的學程」挑題：錨點與被硬條件排除的學程不納入。

### 4.6 先修與申請條件差距

條件差距**只呈現、不過濾**：推薦清單不會因為學生缺先修而刪掉或降低排序（設計原則 D9）。差距可以補，學生也不一定馬上申請。學經歷只交給條件差距（C6），推薦（C2）讀不到。

#### 學經歷輸入（W-61）

每位學生一個 JSON 檔（UTF-8，`"schema": "student_record_intake/1"`）。選 JSON 而不是 CSV：一位學生有多個學位、多門課、多個考試與經驗，各自欄位不同，JSON 一個檔就能放下，CSV 要拆成好幾個檔；Python dict 用同樣的結構也可以直接傳入（`intake_student_record`）。虛構範例：[`data/synthetic/student_record_example.json`](../data/synthetic/student_record_example.json)。**真實學生的檔案只能放在 `data/private/`**，`student_id` 請用代號。

| 區塊 | 欄位（* 為必填） | 說明 |
|---|---|---|
| 最上層 | `schema`*、`student_id`*、`synthetic`、`target_apply_by` | `target_apply_by` 探索時留 `null`；只有學生自己設定日期才進入申請模式 |
| `degrees` | `level`*（`bachelor`／`master`／`doctorate`／`professional`／`other`）、`field`*、`cip`、`year` | 可多個、可跨領域；專科、副學士填 `other`。學士以上學位自動算 `bachelor_degree`；**不會**自動判斷是否等同美國學士（`bachelor_equivalency`，依學程規定） |
| `courses` | `name`*、`credits`、`grade`、`online`、`status`、`expected`、`categories` | 課名照成績單寫；`grade` 可填 0–100、A+～C-、P／通過／抵免、F／不及格／W／停修（D 各校不一，請填百分制）；不及格與停修不算已修；`status: "planned"` 要附 `expected`（預計完成時間）；`categories` 用來手動對應（`[]` 表示確認不對應任何類別） |
| `exams` | `name`*、`score`、`date`、`status`、`expected`、`category` | GRE、GMAT、TOEFL／托福、IELTS／雅思、Duolingo、LSAT、DAT、MCAT、日檢、專技高考等；分數會保留在證據中 |
| `experience` | `category`*（`work_experience`、`research_experience` 等）、`description`*、`months`、`status`、`expected` | |
| `other` | `category`*（作品集、證照、面試、其他語言等）、`evidence`*、`status`、`expected` | |

課名對應用的是**明確、有版本的對應表** [`data/reference/course_name_map_tw.json`](../data/reference/course_name_map_tw.json)（draft-1，**待使用者確認**）：全形轉半形、去空白、拆出序號（（一）、上、I、1）後**完全比對**，不做模糊比對、不用 LLM。

| 課名例子 | 對應類別 |
|---|---|
| 微積分、微積分（一）、微積分甲上、Calculus I | `calculus_1` |
| 微積分（二）、多變數微積分 | `calculus_2` |
| 線性代數 | `linear_algebra` |
| 統計學、商用統計學、心理統計學 | `statistics`（**不算** `probability`） |
| 機率與統計、數理統計 | `probability` ＋ `statistics` |
| 程式設計、計算機程式設計、Python程式設計 | `programming_intro` |
| 資料結構、演算法 | `data_structures` |
| 經濟學（一）／（二） | `microeconomics`／`macroeconomics`（未標序號的經濟學原理兩者都算） |
| 會計學、財務會計 | `accounting` |
| 財務管理、公司理財 | `finance_intro` |
| 普通化學、有機化學、普通生物學、解剖學 | 對應的理科類別 |
| 計算機概論、商用微積分、工程數學、高等微積分、計量經濟學、大一英文 | **不自動對應**：內容因校而異，列入待人工對應並附原因 |

對應表以外的課名（例如作業研究、體育）一律列入 `record.unmapped`，**不會猜**、也不會算已符合；確認後在輸入檔加上 `categories` 即可。

```python
from aisc.profiling import load_student_record

intake = load_student_record("data/synthetic/student_record_example.json")  # 虛構示範學生
print(intake.map_version, sorted(intake.record.completed))
print("規劃中:", sorted(intake.record.planned))
print("待人工對應:", sorted(intake.record.unmapped))
for e in intake.entries[:4]:
    print(e.name, "→", e.categories, e.source)  # 每門課用了哪一條規則
```

輸入有錯時會一次列出所有問題與位置（例如 `courses[1].expected: planned items need an expected completion`、`courses[2].categories: 'statistcs' is not a course category (did you mean statistics?)`），型別是 `RecordIntakeError`（`.errors` 為清單）。

#### 計算差距

也可以不經輸入檔，直接建立 `StudentRecord`：

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

學生紀錄的另外兩個欄位：`online_only`（某類別的證據全是線上課程；學程註明不接受線上學分時，該條件列為缺少）、`unmapped`（待人工對應的課名與考試）。考試與成績目前只比對「有沒有」，**不比對分數是否達到官網最低要求**（B-C6-12）；差距卡會把你的分數與官網原文並列，請自行對照。

可用的條件類別見 `data/reference/requirement_taxonomy.json`（課程、考試、作品集、經驗、證照、語言等）。

### 4.7 申請組合（⏸ 暫停）

錄取機率與申請組合目前暫停（[ARCHITECTURE D13](ARCHITECTURE.md)）：主線流程到推薦、條件差距與報告為止。程式與介面保留，用法移到 [§9.1](#91-申請組合c4暫停)。

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

第二層要細問哪些領域：`aisc.profiling.hierarchy.areas_to_ask(group_scores, m=2)`（前 m 個群的所有領域）或 `q=8`（依同樣順序取前 q 個）。m、q 沒有預設值：職業資料顯示「相關補全＋細問 4–8 個領域」可能就夠（[F115](../results/interest-area-adaptive-findings.md)），但要等 W-41 學生資料預先登記後才定；合成資料的預先登記選擇（W-66）也沒有選出任何 m／q（[F180–F184](../results/synthetic-l3-two-stage-findings.md)）。

可選用：`complete_neutral(v2, area_model="correlated")`（`pipeline.prepare_profile` 同名參數）以 O\*NET 學到的領域相關結構補全未作答領域。用職業資料檢驗時誤差大幅下降，答得越少越明顯，但未通過全部預先登記條件，所以不是預設（[F105–F107](../results/interest-area-imputation-findings.md)）。

### 4.9 錄取機率模型（⏸ 暫停）

錄取機率模型 C3 暫停（D13），用法移到 [§9.2](#92-錄取機率模型-c32-v02c3暫停)。

### 4.10 側寫報告（Markdown）

把需求剖面、兩區推薦（附理由與條件差距）整理成一份給顧問與學生一起看的報告。報告只負責呈現，不改變任何結果；合成資料、系統補全的維度都會自動標示。

```python
from aisc.pipeline import prepare_profile
from aisc.report import render_profile_report

md = render_profile_report(
    student,  # 學生實際作答的剖面
    prepare_profile(student, catalog),  # 排序時實際使用的完整剖面
    result,  # interactive_discovery 的結果
)
print(md.splitlines()[0])
```

`gaps`（選用）是 `{program_id: RequirementGap}`，會在每個學程旁顯示還缺哪些條件，並加上「條件差距卡」一節（每個條件引用官網原文與來源），但不影響排序（D9）。再加上 `record`（選用，§4.6 的 `StudentRecord`），差距卡會並列學生自己的紀錄（課名、學分、成績、考試分數），並列出待人工對應的課名。`plan`（選用）是暫停中的申請組合結果（§9.1），不給就不會出現該段。

```python
from dataclasses import replace

shown_ids = [r.program.program_id for r in result.discoveries + result.known_good]
my_record = replace(intake.record, student_id=student.student_id)  # 示範：讓虛構紀錄對上示範學生
src_ = {"source_url": "https://example.edu/fixture", "retrieved_at": "2026-10-29"}
demo_items = [  # 示範用虛構條件
    RequirementItem(shown_ids[0], "course", "probability", "required", "Probability", **src_),
    RequirementItem(shown_ids[0], "course", "calculus_1", "required", "Calculus I", **src_),
]
card_gaps = compute_gaps(shown_ids, demo_items, my_record, taxonomy)
md = render_profile_report(
    student, prepare_profile(student, catalog), result, gaps=card_gaps, record=my_record
)
print([line for line in md.splitlines() if line.startswith("## ")])
```

### 4.11（實驗中）真實學程目錄 v0

以 IPEDS 2025 的真實「學校 × 學門（CIP）」碩士學程，加上由 O\*NET 職業資料推得的興趣與 41 個領域分數建立目錄（v2 維度）。數理強度、程式強度、研究取向、創業取向**還沒有測量**：以共用暫填值代入、列在 `Program.unmeasured`，不影響 v2 預設計分器的排序，報告也不會把它們當理由（A-C0-15、B-C0-18）。同一學門的學校分數相同，所以推薦清單會被同一學門的不同學校佔滿；目前請把結果讀成「學門」推薦。

```python
from aisc.catalog.real import load_real_catalog_v0
from aisc.contracts import StudentProfile
from aisc.pipeline import interactive_discovery

real = load_real_catalog_v0()  # 讀 data/reference/，不需網路；預設碩士、畢業人數 ≥ 10
print(len(real.programs), "real programs; unmeasured:", real.programs[0].unmeasured)

student_v2 = StudentProfile(
    "demo-v2", {"riasec_i": 0.8, "bi_mathematics_statistics": 0.9, "bi_finance": 0.7}, dims="v2"
)
found = interactive_discovery(
    student_v2,
    real.programs,
    lambda ids: {i: "unknown" for i in ids},
    n_shortlist=10,
    k_new=5,
    k_known=0,
)
print([r.program.degree_name for r in found.discoveries][:2])
```

也可以先寫成檔案：`python scripts/build_real_catalog.py`（輸出 `data/catalog/real_v0.jsonl`，不進版控），再用 `load_programs` 讀取。

---

## 5. 給顧問：建議的使用流程（目前階段）

| 步驟 | 做什麼 | 對應 |
|---|---|---|
| 1 | 和學生談**想做什麼**，不談學校或學程名稱；把有把握的維度填進 `needs` | §4.2 |
| 2 | 有大考中心興趣量表結果就加入 | §4.3 |
| 3 | 記下學生心中已有的學程作為 `anchors` | §4.2 |
| 4 | 執行熟悉度確認，讓學生自己回答聽過哪些 | §4.5 |
| 5 | 和學生一起看「可能沒想過」清單，討論每一個為什麼契合 | §4.5 |
| 6 | 填學經歷輸入檔（學位、修課、考試、經驗）；和學生一起確認「待人工對應」的課名 | §4.6 |
| 7 | 對有興趣的學程看條件差距卡，規劃補修時程 | §4.6、§4.10 |

目前請把它當作**討論工具**而非結論：學程資料是合成的。選校組合與錄取機率暫不由模型提供（D13）。

---

## 6. 限制與注意事項

| 限制 | 影響 | 何時解決 |
|---|---|---|
| 示範用學程目錄是合成資料；真實目錄 v0（§4.11）的強度與取向尚未測量 | 推薦只依興趣與領域，同一學門的學校無法區分 | W-20～W-22（W-21 前導已完成，[w21-pilot.md](w21-pilot.md)） |
| 推薦用的學程目錄只涵蓋商管 × 數據 × 工程管理（合成） | 其他領域的學生推薦品質未知；全領域的官方參考表（`data/reference/*_all.csv`：1,233 個 CIP，含專業與研究型博士）已建好，但還不是可推薦的學程；v2 維度（§4.8）實驗中 | W-16c、W-20～W-22 |
| 程式內的自適應問卷仍是抽象占位題；v2 題庫中文草案（`data/items/item_bank_v2.json`，73 題）尚未回譯與認知訪談 | 還不能直接拿給學生作答 | W-41 |
| 錄取機率與申請組合暫停（D13） | 模型不建議申請哪幾間 | 範圍重新納入時（W-44、W-45） |
| 需求維度尚未實證驗證 | 維度可能重疊或遺漏 | W-41、W-73 |
| 台灣課名對應表是草案（draft-1），只涵蓋常見課名 | 系所自訂課名會列為待人工對應；同名課程深淺不一（B-C6-2） | 使用者確認例子；W-41 收集實際課名 |
| 條件差距只比對「有沒有修／考」 | TOEFL 分數或課程成績未達學程最低要求時仍顯示已符合（B-C6-12） | 學程端最低分數擷取（W-20、W-21） |

已知假設與盲區的完整清單見 [REGISTER.md](REGISTER.md)。

**個資：** 真實學生資料只能放在 `data/private/`（不會被 commit），收集須取得書面同意（見 [CONTRIBUTING.md](../CONTRIBUTING.md)）。

---

## 7. 常見問題

| 問題 | 回答 |
|---|---|
| 為什麼不直接過濾掉學生條件不夠的學程？ | 條件可以補，學生也不一定馬上申請；過濾會讓學生再次只看到自己已知的範圍（D9）。 |
| 沒填的維度會被當成沒興趣嗎？ | 不會。`pipeline` 會先用錨點補全未作答維度（B-C1-14）；已作答的維度維持原值。 |
| 「可能沒想過」裡出現學生其實聽過的學程？ | 熟悉度是學生自己回報的，可能漏報；重新執行熟悉度確認即可。 |
| 為什麼沒有錄取機率和申請組合？ | 範圍決策 D13：先把需求側寫、推薦與條件差距做好；介面保留，之後可以接回（§9）。 |
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
| 2026-10-28 | v0.3.11 | 錄取機率模型預設改為 v0.2（τ 模型平均＋以申請紀錄積分的官方錄取率錨點，區間涵蓋率較準；`anchor="mean"`、`tau_average=False` 為舊版）；新增 `pipeline.plan_applications`：由推薦＋錄取機率直接產生申請組合，並附錄取相關時的期望效用（§4.7） | W-44、W-62 |
| 2026-10-28 | v0.3.12 | 新增側寫報告 `aisc.report.render_profile_report`（Markdown：需求剖面、兩區推薦與理由、條件差距、申請組合；自動標示合成資料與假設值，§4.10）；錄取模型的校準檢查工具（§4.9） | W-50、W-44 |
| 2026-10-28 | v0.3.13 | 輸入檢查更嚴格（第五次稽核）：拼錯的維度、超出 0–1 的作答、未知或型別錯誤的硬條件、重複的學程代碼都會報錯；作答預設不確定性改為 0.25（原 0.5 讓問卷重問已答維度）；重複的錨點只算一次；自適應問卷只為可推薦的學程挑題（§4.2、§4.3、§4.5） | B-C1-26～31、B-C2-20～21 |
| 2026-10-28 | v0.3.14 | 申請組合可用多屬性效用：`ahp_weights`（成對比較權重與一致性比率）、`multi_attribute_utility`、`scale_linear`（§4.7） | W-45 |
| 2026-10-28 | v0.3.15 | v2 新增 `areas_to_ask`：依第一層群分數排出第二層要細問的領域（§4.8；m、q 無預設） | W-66 |
| 2026-10-28 | v0.3.16 | 錄取機率模型可加入學程類型（`types=`、`predict(..., ptype=)`，§4.9） | W-74 |
| 2026-10-28 | v0.3.17 | 錄取紀錄可標申請年度（`AdmissionRecord(..., year=)`），模型估計年度波動、預測針對最新年度（§4.9） | B-C3-3 |
| 2026-10-28 | v0.3.18 | 錄取機率模型新增 `converged` 與 `anchor_conflicts`，未收斂或官方錄取率與紀錄衝突時警告（§4.9） | B-C3-10～12 |
| 2026-10-28 | v0.3.19 | 申請結果資料的範本、載入器（隱私欄位檢查、標準化、缺漏處理）與時間切分校準檢查腳本（§4.9、[data-intake.md](data-intake.md)） | W-44 C3.1 |
| 2026-10-29 | v0.4.0 | 範圍調整（D13）：錄取機率模型（C3）與申請組合（C4）暫停，主線流程到推薦、條件差距與報告為止；程式與介面保留，用法移到 §9；示範程式不再計算申請組合 | D13 |
| 2026-10-29 | v0.4.1 | 新增實驗中的真實學程目錄 v0（`aisc.catalog.real.load_real_catalog_v0`、`scripts/build_real_catalog.py`，§4.11）；`Program` 新增 `unmeasured` 欄位；報告標示未測量維度與低信心剖面，且不以未測量維度當理由；錨點的未測量維度不再當作先驗證據（B-C1-32） | W-77 |
| 2026-10-29 | v0.4.2 | 新增學經歷輸入：JSON 檔（多學位、修課、考試、經驗）→ `StudentRecord`，台灣課名對應表 draft-1（待確認），無法對應的課名列出待人工對應（`aisc.profiling.load_student_record`，§4.6）；`StudentRecord` 新增 `online_only`、`unmapped`；僅線上修課不再符合「不接受線上學分」的條件；報告新增條件差距卡（`render_profile_report(..., record=)`，§4.10） | W-61 |
| 2026-10-29 | v0.4.3 | 定位改為學生自助工具（D14）：不做研究型驗證；新增個資告知範本（[privacy-notice.md](privacy-notice.md)）；用法沒有改變 | D14 |

---

## 9. 附錄：暫停的模組（D13）

以下模組不在目前的主線流程，程式與介面保留、測試照常執行，之後可以直接接回：任何來源（C3 模型、顧問估計）只要提供 `AdmitEstimate`，就能交給 `plan_applications`。這兩節的結論只在合成資料成立。

### 9.1 申請組合（C4，暫停）

給定每個學程的錄取機率與效用，找出在申請數量（或預算）限制下期望效用最高的組合。沒有 C3 估計時，錄取機率必須以 `AdmitEstimate(method="assumed")` 明確標示為假設。

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

一般使用時可以直接用 `plan_applications`，它把推薦清單、每個學程的 `AdmitEstimate` 與預算組合起來（效用預設為契合度 × 100、每間申請成本 1），並附上錄取相關時同一組合的期望效用。它只在申請模式使用，不會改變推薦結果（D9）：

```python
from aisc.pipeline import plan_applications

admit = {r.program.program_id: a for r, a in zip(shortlist_, assumed, strict=True)}
plan = plan_applications(shortlist_, admit, budget=3)  # criterion="point"（預設）
print(plan.chosen_ids, round(plan.result.expected_utility, 1), plan.result.method)
print({rho: round(v, 1) for rho, v in plan.value_if_correlated.items()})  # ρ 越高越低
```

契合度不是唯一考量（學費、地點、學制長度……，B-C4-3）。可以用成對比較請學生給屬性權重（AHP；一致性比率 ≥ 0.1 時請學生重新比較），再組成效用交給 `plan_applications(..., utility=...)`。這個工具尚未經使用者研究驗證（W-42）：

```python
from aisc.portfolio import ahp_weights, multi_attribute_utility, scale_linear

# 契合度比學費重要 3 倍
w, cr = ahp_weights([[1, 3], [1 / 3, 1]])
values = {
    r.program.program_id: {
        "fit": r.relevance,
        "cost": scale_linear(r.program.tuition_usd or 60_000, worst=90_000, best=20_000),
    }
    for r in shortlist_
}
utility_mau = multi_attribute_utility(values, {"fit": w[0], "cost": w[1]})
plan_mau = plan_applications(shortlist_, admit, budget=3, utility=utility_mau)
print(round(cr, 3), plan_mau.chosen_ids)
```

### 9.2 錄取機率模型 C3.2 v0.2（C3，暫停）

有歷年申請結果時，可以用階層 logistic 模型估計錄取機率與區間。學生特徵要先以該學程的申請者母體標準化（平均約 0）；自陳資料（如網路上的申請結果分享）標為 `"self_report"`，顧問的完整紀錄標為 `"complete"`。提供官方錄取率時，模型會校正「錄取者較常自陳」的偏誤（合成資料中 RMSE 從 0.13–0.26 降到 0.04–0.07，[F128–F131](../results/synthetic-l3-admission-bias-findings.md)）。**目前沒有真實申請結果資料**；當較強的申請者也較常自陳時，模擬中 v0.2 的區間涵蓋率仍有 86–89%，但會略為低估（[F139–F140](../results/synthetic-l3-admission-pool-confirm-findings.md)），所以結果仍要當作假設值使用。

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
print(model.method)  # hierarchical_v0.2_corrected
print(round(est.low, 2), round(est.mean, 2), round(est.high, 2))
new = model.predict("prog-c", (0.5, 0.0), family="52.13")  # 沒有資料的學程：用族群平均，區間較寬
```

紀錄來自不同申請年度時，在 `AdmissionRecord(..., year=2024)` 標上年度（要嘛全部標、要嘛都不標）：每個較舊的年度有自己的整體難度，預測一律針對最新年度（官方錄取率也請用最新年度的；[F148–F150](../results/synthetic-l3-admission-years-findings.md)）。知道各學程的申請流程類型（一般、面試制、作品集審查、專業學位）時，傳入 `types={program_id: 類型}`；預測新學程時再給 `ptype=`。模擬中這能讓新學程的錄取機率更準，類型沒有效果時也不會變差（[F145–F147](../results/synthetic-l3-admission-types-findings.md)）。

模型沒有收斂（`model.converged` 為 False）或官方錄取率與紀錄嚴重衝突（`model.anchor_conflicts` 列出學程與 logit 差距）時會發出警告：請先檢查資料，再使用這些估計。用自己的歷年申請結果時，照 [data-intake.md](data-intake.md) 準備兩個 CSV（範本在 `data/templates/`），再執行 `python scripts/validate_admission_model.py --records … --rates …`：會檢查隱私欄位、擬合模型，並用時間切分檢查校準；程式介面是 `aisc.feasibility.intake.fit_from_files`。有了留出的真實結果後，可用 `aisc.feasibility` 的 `brier_skill`（是否勝過「官方錄取率」常數預測）、`calibration_slope`（理想為截距 0、斜率 1；斜率小於 1 表示過度自信）、`reliability_table` 與 `by_group`（例如台灣／國際申請者分開）檢查校準（V-C3-2～4）。
