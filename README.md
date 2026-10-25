# AI Studying Consultant

> 側寫台灣留學生的真實需求，推薦他們沒想過、但更契合的美國研究所 program，並最佳化申請組合。
>
> Profile Taiwanese applicants, surface US graduate programs they never considered, and optimise which ones to apply to.

**Status:** v0.2 · research prototype · all program data in this repo is **synthetic**.

## 要解決的問題

台灣學生對研究所的理解，多半建立在台灣有限的學程分類上。要找到更適合的美國 program，需要**自我洞察**和**資訊接觸**，但兩者互為前提：不知道有哪些 program，就很難描述需求；描述不出需求，就搜尋不到新的 program。結果是學生在預設的學校清單裡打轉。

這個專案把它當成**冷啟動推薦 + 探索**問題，拆成六個可以獨立開發、獨立驗證的部件：

| 部件 | 功能 | 狀態 |
|---|---|---|
| C0 Program 知識庫 | CIP 分類骨架 + 兩種獨立來源測量學程屬性 | loader ✅ · 資料管線 ⬜ |
| C1 學生側寫 | 以 RIASEC 為骨架的需求測量、錨點拆解、自適應提問、熟悉度 | 15 維需求空間 ✅ · 大考中心分數先驗 ✅ · 熟悉度擷取 ✅ · 錨點拆解 v1 ✅ · 自適應問卷 v1（節省 13–28% 題目，未達 30% 門檻） |
| C2 探索推薦 | 依契合排序，再依學生回報的熟悉度分成「可能沒想過」與「已知且適合」兩區 | **v1 ✅ L3 通過**；v0 保留為參考 |
| C3 錄取可行性 | 階層貝氏模型 + 選擇偏誤校正 + 校準 | ⬜ |
| C4 申請組合 | Kapur & Hong 精確解 + 相關性評估 + 穩健選擇 | 精確解 ✅（已驗證） |
| C5 模擬與評估 | 已知真相的合成學生，驗證所有部件 | 產生器、指標、執行器 ✅ · 首次 L3 報告 ✅ |

## 文件導覽

| 想知道 | 看這裡 |
|---|---|
| 整體架構、設計原則、部件圖 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 每個部件的規格 | [docs/components/](docs/components/) |
| 每個方法的理論來源與假設 | [docs/THEORY.md](docs/THEORY.md) |
| 已知假設與演算法盲區 | [docs/REGISTER.md](docs/REGISTER.md) |
| 驗證階梯、關卡、指標門檻 | [docs/VALIDATION.md](docs/VALIDATION.md) |
| 工項與順序 | [docs/ROADMAP.md](docs/ROADMAP.md) |
| 需求維度與評分準則 | [docs/rubric.md](docs/rubric.md) |
| 最新 L3 合成驗證結論 | [C2 分區呈現](results/synthetic-l3-v2-findings.md)、[C1 錨點拆解 v1](results/synthetic-l3-c1-v1-findings.md)、[C1 自適應問卷](results/synthetic-l3-adaptive-findings.md) |

## Quickstart

```bash
pip install -e ".[dev]"
pytest -rx                   # 227 passed + 4 xfailed（xfail = 已知盲區探針）
python examples/quickstart.py
python scripts/run_l3_baseline.py --tag v2 --needs-model archetype    # L3，約 6 分鐘
python scripts/run_l3_baseline.py --tag v2-continuous --needs-model continuous
python scripts/compare_l3.py results/synthetic-l3-v2.md results/synthetic-l3-v2-continuous.md
```

## 驗證方式一覽

每個部件的每個版本都要通過六層驗證：**L1 理論符合 → L2 程式驗證 → L3 合成還原 → L4 盲區探針 → L5 實證 → L6 外部審查**。
盲區探針以 `@pytest.mark.probe` + `xfail(strict=True)` 寫成測試：盲區修好時測試會轉為通過，CI 會要求同步移除標記並更新登錄表。

## License

MIT © Po-An (Brian) Chan
