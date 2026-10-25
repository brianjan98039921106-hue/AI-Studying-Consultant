# AI Studying Consultant

> 側寫台灣留學生的真實需求，推薦他們沒想過、但更契合的美國研究所 program，並最佳化申請組合。
>
> Profile Taiwanese applicants, surface US graduate programs they never considered, and optimise which ones to apply to.

**Status:** v0.1 · research prototype · all program data in this repo is **synthetic**.

## 要解決的問題

台灣學生對研究所的理解，多半建立在台灣有限的學程分類上。美國的 program 細分程度高得多，但學生（和顧問）想找到更適合的 program，需要兩樣東西：

1. **自我洞察**：知道自己真正想要什麼
2. **資訊接觸**：知道有哪些 program 存在

兩者互為前提：不知道有哪些 program，就很難描述需求；描述不出需求，就搜尋不到新的 program。結果是學生在自己預設的學校清單裡打轉。

這個專案把它當成一個**冷啟動推薦 + 探索**問題來處理：

| 層 | 功能 | 狀態 |
|---|---|---|
| L0 Program 知識庫 | 以 NCES CIP 分類為骨架的 program 屬性資料 | schema ✅ · 真實資料 ⬜ |
| L1 學生側寫 | 以「需求維度」而非 program 名稱描述學生；**錨點拆解**從學生已知的 program 反推需求 | 錨點拆解 ✅ · 自適應問卷 ⬜ |
| L2 探索推薦 | 需求空間決定**契合度**，標籤空間（CIP）決定**新穎度**，MMR 確保多樣性 | ✅ |
| L3 錄取可行性 | P(admit) 估計 | ⬜ |
| L4 申請組合 | Kapur & Hong (2022) college application problem：greedy + 精確 DP | ✅（已驗證） |
| 顧問模式 | 側寫報告 + 推薦理由 | ⬜ |

完整架構與數學式：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · 待辦：[docs/ROADMAP.md](docs/ROADMAP.md) · 驗證計畫：[docs/VALIDATION.md](docs/VALIDATION.md) · 相關研究：[docs/related-work.md](docs/related-work.md)

## Quickstart

```bash
pip install -e ".[dev]"
pytest                       # 93 tests, incl. brute-force solver verification
python examples/quickstart.py
```

範例輸出（合成資料）：一位只知道「MS Finance」、但喜歡數學和一點程式的學生，系統推薦了 Financial Mathematics、Operations Research、Statistics 等不在商學院分類下的 program。

## 已知限制（v0.1）

- **資料是合成的**：屬性分數由人工設定，只用來測試演算法行為，不代表任何真實學校。
- **新穎度權重尚未校準**：預設 `novelty_weight=0.4` 時，契合度高但同屬商學院的 program（如 MS Business Analytics）會被壓到後面。正確權重要靠 V4 使用者研究決定。
- **錄取獨立假設**：L4 假設各校錄取彼此獨立，實際上高度相關，會高估「多申請夢幻校」的價值。
- **需求維度是初版**：9 個維度（`src/aisc/needs.py`）尚未經過信效度檢驗。

## License

MIT © Po-An (Brian) Chan
