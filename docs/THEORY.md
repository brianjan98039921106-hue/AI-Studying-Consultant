# 理論來源對照 Theory Map

每個方法都要能回答三個問題：**來自哪個理論？** 這個理論**要求什麼假設**？假設被違反時**會怎樣失敗**？
第三欄的假設編號對應 [REGISTER.md](REGISTER.md)。新增方法時，先在這裡加一列，才能開始寫程式（Gate G0）。

## C0 Program 知識庫

| 方法 | 理論／資料來源 | 引入的假設 |
|---|---|---|
| 分類骨架 | NCES Classification of Instructional Programs (CIP) 2020 | A-C0-1 同一 CIP 群組內容相近 |
| 碩士學程清單 | IPEDS Completions（award level 7 = master's） | A-C0-4 有畢業生紀錄才算存在 |
| 職業橋接 | NCES／BLS CIP 2020 ↔ SOC 2018 對照表 | A-C0-2 對照表反映畢業生會進入的職業 |
| 職業興趣剖面 | O*NET Occupational Interest Profiles（RIASEC） | A-C0-3 職業的興趣剖面可轉嫁到對應學程 |
| LLM 屬性擷取 | 擷取式 NLP；幻覺風險見 Ji et al. (2023), *ACM Computing Surveys* | A-C0-5 擷取結果可由原文佐證 |
| 一致性檢驗 | 心理計量的收斂效度（convergent validity）；評分者一致性 Cohen's κ | — |

## C1 學生側寫

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 需求維度骨架 | Holland (1997), *Making Vocational Choices*：RIASEC 六型職業興趣；O*NET Interest Profiler | A-C1-1 興趣是相對穩定、可測量的潛在特質 |
| 興趣短題 | O*NET Interest Profiler Short Form：Rounds, Su, Lewis & Rivkin (2010) 心理計量報告 | 英文常模；中文版需重新驗證（B-C1-3） |
| 台灣既有興趣資料 | 大學入學考試中心興趣量表：以 Holland 類型論為基礎，六型各 33 題，每型 0–99 分，輸出三碼興趣代碼，並對應台灣大學學群 | A-C1-6 高中時測得的興趣代碼對研究所申請仍有參考價值（B-C1-7） |
| 大考中心分數轉先驗 | 每型分數 0–99 線性換算為 [0, 1]；先驗標準差隨測驗距今年數線性增加並設上限；量表手冊以「區分值 < 3」表示代碼不明確，此時再放大標準差；與陳述需求以精度加權合併 | A-C1-7 分數可線性換算到 [0, 1]；A-C1-8 興趣隨時間的漂移可用「標準差隨年數增加」近似 |
| 成對情境題 | 強迫選擇（forced-choice）格式降低默許與社會期許偏誤 | A-C1-2 的部分對策 |
| 測量模型與信度 | 古典測驗理論（Cronbach's α、再測信度）；因素分析 | A-C1-1 |
| 作答偏誤 | Podsakoff et al. (2003), *J. Applied Psychology*：共同方法偏誤、社會期許 | A-C1-2 作答誠實且不受題目措辭影響 |
| 偏好建構 | Payne, Bettman & Johnson (1992), *Annual Review of Psychology*；Lichtenstein & Slovic (2006), *The Construction of Preference* | A-C1-4（被此理論**否定**）：偏好在提問前就存在 |
| 錨點拆解 | 兩個獨立高斯測量的貝氏合併（precision weighting） | A-C1-3 陳述與錨點是獨立、無偏的測量 |
| 自適應提問 | 電腦化適性測驗 CAT：van der Linden & Glas (2000)；適應式效用擷取：Chajewska, Koller & Parr (2000), AAAI | A-C1-5 作答模型設定正確 |
| 冷啟動提問 | Christakopoulou, Radlinski & Hofmann (2016), KDD；Zuo et al. (2022) 階層式提問 | 同上 |
| 定錨效應 | Tversky & Kahneman (1974), *Science* | 說明為何錨點會扭曲判斷（B-C1-2） |

## C2 探索推薦

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 契合度 | 向量空間相似度（cosine） | A-C2-1 契合度隨方向相似度單調增加 |
| 新穎度、意外性、覆蓋率 | Kaminskas & Bridge (2016), *ACM TiiS*；Vargas & Castells (2011), RecSys | A-C2-2 新穎價值可與契合度線性相加；A-C2-3 分類距離代理「不知道」 |
| 多樣化 | Carbonell & Goldstein (1998), SIGIR：Maximal Marginal Relevance | — |
| 熱門度偏誤 | Abdollahpouri, Burke & Mobasher (2019), FLAIRS | — |
| 清單長度 | Iyengar & Lepper (2000), *JPSP*：選擇過載 | — |
| 推薦理由 | Tintarev & Masthoff (2007)：推薦解釋的忠實性 | A-C2-4 理由反映實際的計分方式 |

## C3 錄取可行性

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 階層貝氏 logistic | 部分合併（partial pooling）：資料少的學程向群體平均收縮 | A-C3-1 過去結果可預測未來（定態） |
| 選擇偏誤 | Heckman (1979), *Econometrica* | A-C3-2 訓練樣本具代表性（GradCafe **不符合**） |
| 機率校準 | Brier (1950)；Niculescu-Mizil & Caruana (2005), ICML | — |
| 不可觀測因素 | Zhao et al. (2023), *PLOS ONE*：推薦信帶有預測訊號 | 說明區間為何必須夠寬（B-C3-5） |

## C4 申請組合

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 申請組合問題 | Kapur & Hong (2022), arXiv:2205.01869 | A-C4-1 錄取彼此獨立；A-C4-2 只就讀一所、目標是期望最佳結果 |
| 序列決策對照 | Weitzman (1979), *Econometrica*：Pandora's box | A-C4-3 所有申請同時決定（靜態） |
| 市場均衡 | Fu (2014), *Journal of Political Economy* | S-5 大量使用會改變錄取率 |
| 相關性 | Sklar 定理／Gaussian copula | — |
| 穩健最佳化 | Bertsimas & Sim (2004), *Operations Research*, "The Price of Robustness" | — |
| 多屬性效用 | Keeney & Raiffa (1976), *Decisions with Multiple Objectives*；AHP (Saaty) | A-C4-4 效用可由屬性加權組成 |

## C5 模擬與系統層

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 模型誤設測試 | 統計模擬研究的標準做法：資料生成模型 ≠ 估計模型 | 避免 S-1 循環驗證 |
| 資料洩漏 | Kaufman et al. (2012), *ACM TKDD*, "Leakage in Data Mining" | S-3 |
| 合成需求分布 | 潛在類別混合（原型 + 個體雜訊） | A-C5-1 需求由少數原型加雜訊產生 |
| 熟悉度生成 | Logistic 模型：知名度、與學生背景同分類、台灣常見分類 | A-C5-2 |
| 錨點選擇 | 離散選擇模型 multinomial logit（McFadden, 1974）：效用 = 契合度與名氣的加權 | A-C5-3 錨點只能從熟悉的學程中選 |
| 陳述需求的作答風格 | 默許傾向、社會期許（Podsakoff et al., 2003） | A-C5-4 偏誤可加成到真實需求上 |
| 真實效用（ground truth） | 兩種定義並列：加權歐氏距離（考慮強度）與 cosine（只看方向） | A-C5-5 真相函數的選擇會影響哪個估計器看起來較好（B-C5-1） |
| 排序一致性 | Kendall (1945), *Biometrika*：含同分的 τ-b | — |
| 分布差異 | Kullback & Leibler (1951) KL 散度（加平滑） | — |
| 機率評分 | Brier (1950) | — |

## 相關開源專案與資料

| 專案／資料 | 用途 |
|---|---|
| [maxkapur/OptimalApplication.jl](https://github.com/maxkapur/OptimalApplication.jl)、[maxkapur/CollegeApplication](https://github.com/maxkapur/CollegeApplication) | C4.1 的參考結果（測試已重現其 README 範例） |
| Kaggle "Graduate Admissions"（Acharya, 2018）相關 repo | 只作入門參考；目標變數為合成，不可用於 C3 校準 |
| Heidari Far & Tabrizi (2026), arXiv:2603.29881 | 約 13,000 筆 GradCafe 紀錄，結合 OpenAlex、QS、Wikidata |
| arXiv:2501.06581 | 以課程內容 + 個人偏好推薦學程 |
| [NCES CIP 2020 ↔ SOC 2018 crosswalk](https://nces.ed.gov/IPEDS/CIPCODE/Files/IES2020_CIP_SOC_Crosswalk_508C.pdf) | C0.2 |
| [O*NET Occupational Interest Profiles](https://www.onetcenter.org/reports/OIP.html) | C0.2、C1.1 |
