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
| MVP 範圍篩選（W-14） | CIP 2020 代碼前綴清單（`data/reference/mvp_scope.json`），award level 7、第一主修（MAJORNUM=1）、最近一年畢業人數 > 0 | A-C0-4、A-C0-7 |
| 職業剖面量尺與合成（W-15 v1） | O*NET 職業興趣量尺 OI 為 1–7（Rounds et al., 1999, *Development of Occupational Interest Profiles for O\*NET*），線性轉為 $(v-1)/6 \in [0,1]$；同一 SOC 下多個 O\*NET-SOC 細項取平均，一個 CIP 對應多個 SOC 時等權平均（v1；v2 以就業人數加權） | A-C0-2、A-C0-3、A-C0-6 |
| 職業剖面 v2：就業加權（W-71） | BLS Occupational Employment and Wage Statistics（OEWS）各 SOC 就業人數作為權重（沒有自由參數）；教職 SOC 另行處理（B-C0-8）。**2026-10-28 建置後未採用為預設**：權重集中於單一職業並放大教職稀釋（F102–F104） | A-C0-2、A-C0-3、A-C0-11 |
| 職業剖面 v1.1：依學位層級處理教職（W-71 的過渡版） | 對照表把多數學術 CIP 對應到「某學科大學教師」（SOC 25-1xxx），但碩士、專業學位與證書的畢業生很少成為大學教師（B-C0-8、F51、F96）。「非學術」剖面：該 CIP 有其他 SOC 有 O\*NET 資料時，排除 25-1xxx 後等權平均；只有教職 SOC 時沿用全部並標記。研究型博士（level 17）使用含教職的原剖面，因為學術是其主要出路之一。沒有自由參數；就業加權（W-71）取得 BLS OEWS 後取代等權 | A-C0-10 |
| 細分興趣橋接（W-64） | 與 RIASEC 相同的 CIP → SOC → O\*NET 路徑，元素換成 41 個細分興趣領域；OI 1–7 → $(v-1)/6$；等權平均 | A-C0-9、B-C0-14 |
| 評分準則跨領域外推（W-72） | 留一族群外推（leave-one-family-out）：在部分 CIP 族群開發，在未參與開發的族群檢驗一致性 | A-C0-8 |
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
| 錨點拆解 v1（W-23） | 角色改變：錨點只作為**尚未作答維度**的先驗（缺值補全），已作答維度以陳述為主、錨點僅以大幅放大的標準差參與；錨點平均名氣高於目錄平均時再放大標準差（名氣偏誤校正）。依據：L3 F4a、F5、F9（錨點合併在已作答維度上一致傷害排序） | A-C1-10 名氣可由外部資料衡量；A-C1-11 未作答維度的預設值（中性 0.5）不偏向任何學程 |
| 自適應提問 | 電腦化適性測驗 CAT：van der Linden & Glas (2000)；適應式效用擷取：Chajewska, Koller & Parr (2000), AAAI | A-C1-5 作答模型設定正確 |
| 冷啟動提問 | Christakopoulou, Radlinski & Hofmann (2016), KDD；Zuo et al. (2022) 階層式提問 | 同上 |
| 自適應問卷 v1（C1.3，W-25） | 序列貝氏設計：每題測一個維度，高斯共軛更新（精度加權）。四種選題：隨機、最大變異（不確定性抽樣，CAT 的基本做法）、**辨別性不確定性**（後驗變異 × 該維度在目前前段學程間的差異）、**近視資訊價值**（模擬可能的回答，選預期最能改變前 k 名推薦的題目；Chajewska, Koller & Parr 2000 的「以決策改變衡量提問價值」） | A-C1-12、A-C1-13、A-C1-14 |
| 題目雜訊估計（C1.3 v2） | 同一維度的重複題目，其答案差異只來自雜訊；以各維度離均差平方和合併估計雜訊變異數，並以共軛的逆伽瑪先驗（中心 0.25²、強度 ν₀ 份假想自由度）收縮：$\hat\sigma^2 = (\nu_0 s_0^2 + SS)/(\nu_0 + df)$；每次作答後以 $\hat\sigma$ 重新計算所有維度的後驗（Gelman et al., *Bayesian Data Analysis*, 常態–逆伽瑪模型） | A-C1-15、A-C1-16 |
| 截斷常態概似（C1.3 v2，B-C1-16） | 回答在 0 與 1 被截斷：端點上的回答用尾端機率 $\Phi(-w/\sigma)$、$1-\Phi((1-w)/\sigma)$，其餘用常態密度（Tobin 1958, *Econometrica* 26:24–36）。需求限制在 [0, 1]、雜訊 σ 在對數網格上，先驗與上一列相同（縮放逆卡方）；在網格上精確計算聯合後驗並對 σ 積分（同時取代 A-C1-16 的代入估計）。選題的前瞻計算仍用常態近似；區間用最高密度區間（HPD），因為等尾區間依構造排除恰好落在 0 或 1 的真值 | A-C1-15、A-C1-17 |
| 只用重複題學習雜訊（C1.3 v2b，B-C1-18） | 每個維度對 σ 的貢獻是條件概似 $p(y_2,\dots,y_n \mid y_1, \sigma)$，在平坦需求先驗下計算（只答一題時恰好不含 σ 的資訊），使先驗誤差不會被當成雜訊；給定 σ 後，需求後驗仍用真正的先驗。σ 先驗為縮放逆卡方，以均方根 $E[\sigma^2]^{1/2}=0.25$ 設定：$s_0^2 = 0.25^2(\nu_0-2)/\nu_0$（概念同 REML／組內估計：Patterson & Thompson 1971） | A-C1-15、A-C1-17、A-C1-18 |
| 定錨效應 | Tversky & Kahneman (1974), *Science* | 說明為何錨點會扭曲判斷（B-C1-2） |
| 熟悉度回報（C1.4） | 訊號偵測理論（Green & Swets, 1966）：回報錯誤分為漏報（miss，知道但沒勾）與誤報（false alarm，因名稱相似而誤以為知道）；只對候選短名單提問，控制作答負擔 | A-C1-9 只有短名單內的學程需要熟悉度資訊 |
| 測量恆等性（W-73） | 多群組驗證性因素分析：形態、量尺、截距恆等性逐步檢驗（Vandenberg & Lance, 2000, *Organizational Research Methods*）；判準 ΔCFI ≤ 0.01（Cheung & Rensvold, 2002, *Structural Equation Modeling*） | A-C1-19 |
| 題目差異功能 DIF（W-73） | Mantel–Haenszel 與邏輯斯迴歸 DIF（Holland & Wainer, 1993, *Differential Item Functioning*） | A-C1-19 |
| 題目參數校準（W-73） | 等級反應模型（Samejima, 1969, *Psychometrika Monograph* 17） | A-C1-12 由校準後的作答模型取代 |
| 兩階段提問（W-66） | 先粗後細：先測 RIASEC 與大領域群，依結果只深入相關子領域；切換規則以 L3 預先登記選擇 | A-C1-14 |
| 領域相關補全（C1 v2，architecture-review §2.1） | 多變量常態條件期望：未答領域 ＝ μ_U + Σ_UO Σ_OO⁻¹ (y_O − μ_O)，O 為已觀測變數（已答領域與 12 個領域群分數，觀測雜訊加在對角）；條件變異數 Σ_UU − Σ_UO Σ_OO⁻¹ Σ_OU 作為先驗不確定性（Anderson, 2003, *An Introduction to Multivariate Statistical Analysis*, ch. 2）。μ、Σ 以 O\*NET 職業的 41 領域剖面與由其算出的群分數估計，Σ 以 Ledoit–Wolf 收縮（Ledoit & Wolf, 2004, *Journal of Multivariate Analysis* 88）穩定逆矩陣 | A-C1-25、A-C1-26 |
| 領域的自適應提問（C1 v2，B-C1-21） | 貝氏實驗設計（Chaloner & Verdinelli, 1995, *Statistical Science* 10(3)）：沿用上一列的聯合常態模型，每答一題就以條件分佈更新所有領域；下一題選 (A) **A 最適**：讓 41 個領域的後驗變異數總和下降最多的領域，下降量 = Σ_k Cov(k, j)² ／ (Var(j) + σ²)；或 (U) **上信賴界**：後驗平均 + 後驗標準差最大的領域（Srinivas, Krause, Kakade & Seeger, 2010, *ICML* 的 GP-UCB 思路，目標是找出最高的領域）。兩階段提問（先 12 群、再細問最高的群）仍是對照組 | A-C1-25、A-C1-26、A-C1-27 |
| 已答領域去雜訊（C1 v2，F116） | 經驗貝氏收縮（Efron & Morris, 1975, *JASA* 70；單變量即 Kelley 的真分數估計）：已答領域也以聯合常態的後驗平均 E[w_j ｜ 全部作答] 取代原始作答，等於把作答往「其他作答所預測的值」拉，拉的程度由作答雜訊 σ² 與後驗變異數決定。σ → 0 時就是原始作答 | A-C1-25、A-C1-26、A-C1-27 |
| 跨領域領域層（W-64） | O\*NET 41 個細分興趣領域（Specific Interest Areas；研究文獻稱 basic interests），每個連到一或兩個 RIASEC 類型（Rounds, Putka & Lewis, 2022／2023，*Updating Vocational Interests Information for the O\*NET Content Model*）；職業評分以 AI＋專家混合法產生（Liu, Putka, Wu & Lewis, 2026，*Using a Hybrid Artificial Intelligence-Expert Method to Develop Basic Interest Ratings for the O\*NET System*）；O\*NET 31.0 檔案 `specific_interest_areas.csv`（891 個職業 × 41 領域，OI 1–7）與 `specific_interest_areas_to_career_interest_types.csv` | A-C1-20、A-C0-9、B-C0-14、B-C1-19 |
| 細分興趣領域分群（W-64 方案 B） | 領域之間以「跨職業評分的 Pearson 相關」為相似度，平均連結階層分群；k 以輪廓係數選擇（Rousseeuw, 1987, *J. Comput. Appl. Math.*）；以自助法的調整 Rand 指數檢驗穩定度（Hubert & Arabie, 1985, *J. Classification*） | A-C1-21、B-C1-23 |

## C2 探索推薦

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 契合度 | 向量空間相似度（cosine） | A-C2-1 契合度隨方向相似度單調增加 |
| 契合度 v1 候選（W-26） | (a) **分層懲罰**：cosine 減去強度層（數理、程式）的平方差平均 × γ；依據是評分準則把強度層定義為「想要多深」，高估與低估都是不契合（rubric §2–§4）；(b) **混合**：(1−λ)·cosine + λ·(1 − 平均平方差)；(c) **平方距離**：1 − 平均平方差（與 distance 真相的排序完全相同，屬循環比較，只作參考）。三者都可做成「各維度貢獻加總 = 分數」的精確分解 | A-C2-6、A-C2-7 |
| 兩種真相下的選擇 | Maximin（Wald, 1950）：在兩種同樣可信的真相下，選「較差那一種表現」最好的方案 | A-C2-8 |
| 領域把關（W-16c 混合設計） | 非補償式篩選：先以單一屬性淘汰不可接受的選項，再在剩下的選項中補償式比較（Einhorn, 1970, *Psychological Bulletin* 73(3)；Tversky, 1972, *Psychological Review* 79(4) 的 elimination by aspects）。做法：契合度用 v1 計分器在 v2 作答的 v1 投影上計算（不改變 v1 的排序行為）；另以 41 領域計算「學程內容加權的學生領域興趣」a = Σ x_a·w_a / Σ x_a，a 低於門檻 τ 時扣 λ·(τ − a)。λ = 0 時完全等於 v1 計分；扣分可依領域精確分解到各領域（推薦理由） | A-C2-10、B-C2-11 |
| 需求層權重估計（W-76） | 成對比較的條件 logit（Bradley & Terry, 1952, *Biometrika* 39；McFadden, 1974, conditional logit）：P(選 A) = σ(Σ_l β_l·(D_l(B) − D_l(A)))，β_l = exp(θ_l) ≥ 0，最大概似；以受訪者為單位的集群自助法估計區間（Efron & Tibshirani, 1993）；測量誤差偏誤以 SIMEX 校正（Cook & Stefanski, 1994, *JASA* 89）：對作答再加 λσ² 的雜訊（λ = 0.5、1、1.5、2，截到 [0, 1]），重估份量後以二次式外插到 λ = −1；σ 來自 W-41 重測；區間以未校正自助法區間平移校正量近似（由參數還原檢驗涵蓋率）。規格見 [layer-weights.md](layer-weights.md) | A-C2-11、A-C2-12、B-C2-12、B-C2-13 |
| 層權重的測量誤差校正：迴歸校正（W-76） | 成對特徵 D_l(B) − D_l(A) = mean_d[(x_B,d² − x_A,d²) − 2 w_d (x_B,d − x_A,d)] 對學生需求 w **線性**，因此以 E[w ｜ 作答] 取代作答時，特徵的期望值正確（Carroll, Ruppert, Stefanski & Crainiceanu, 2006, *Measurement Error in Nonlinear Models*, 2nd ed., ch. 4；logistic 迴歸的迴歸校正：Rosner, Spiegelman & Willett, 1990, *American Journal of Epidemiology* 132）。每題以線性預測 ŵ_d = μ_d + λ_d (s_d − μ_d)，λ_d = cov(第一次, 第二次) / var(第一次)（重測估計，截在 [0, 1]）；區間以受訪者自助法，每次重抽都重新估計 λ（含重測子樣本） | A-C2-13、A-C2-14、B-C2-14 |
| 新穎度、意外性、覆蓋率 | Kaminskas & Bridge (2016), *ACM TiiS*；Vargas & Castells (2011), RecSys | A-C2-2 新穎價值可與契合度線性相加；A-C2-3 分類距離代理「不知道」 |
| 多樣化 | Carbonell & Goldstein (1998), SIGIR：Maximal Marginal Relevance | — |
| 短名單族群多樣化（C2.3 v1，B-C2-9） | 以類別做多樣化的 MMR 變體（Carbonell & Goldstein, 1998；以類別相似度做清單多樣化：Ziegler, McNee, Konstan & Lausen, 2005, *WWW*「topic diversification」）：依序挑選，每一步取 契合度 − λ·（已選中同族群的數量）最大者；族群 = CIP 前 4 碼（XX.XX）。λ = 0 時完全等於現行的契合度排序。同時用於熟悉度確認的短名單與「可能沒想過」區；「已知且適合」區不變。推薦理由仍是契合度的精確分解，多樣化只改變順序 | A-C2-15、A-C2-16 |
| 分區呈現（C2.3 v1） | 依 L3 F7 的證據設計；與意外性文獻中「相關且非預期」的定義一致（Kaminskas & Bridge 2016）：以契合度排序，再依學生回報的熟悉度分成「可能沒想過」與「已知且適合」兩區，而非以分數懲罰已知學程 | A-C2-5 「新穎」以學生回報的熟悉度定義 |
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
| 相關性（C4.2，W-29） | Sklar 定理／Gaussian copula：潛在變數 Z_j = √ρ·F + √(1−ρ)·E_j，Z_j < Φ⁻¹(p_j) 即錄取，邊際機率不變；以共同隨機數的 Monte Carlo 估計 v(S)；組合以 C4.1 解為起點做加入／移除／交換的局部搜尋（小問題以窮舉作參考） | A-C4-5 |
| 穩健最佳化 | Bertsimas & Sim (2004), *Operations Research*, "The Price of Robustness" | — |
| 多屬性效用 | Keeney & Raiffa (1976), *Decisions with Multiple Objectives*；AHP (Saaty) | A-C4-4 效用可由屬性加權組成 |

## C5 模擬與系統層

| 方法 | 理論來源 | 引入的假設 |
|---|---|---|
| 模型誤設測試 | 統計模擬研究的標準做法：資料生成模型 ≠ 估計模型 | 避免 S-1 循環驗證 |
| 資料洩漏 | Kaufman et al. (2012), *ACM TKDD*, "Leakage in Data Mining" | S-3 |
| 合成需求分布 | 潛在類別混合（原型 + 個體雜訊） | A-C5-1 需求由少數原型加雜訊產生 |
| 第二套需求分布（C5.5） | 無原型的 logistic-normal：潛在常態經 sigmoid 轉換；RIASEC 六型的潛在相關依 Holland 六角形（相鄰 > 相隔 > 相對），見 Holland (1997) 與 Tracey & Rounds (1993), *Psychological Bulletin* 的結構驗證 | A-C5-6 相關強度（相鄰 0.3、相隔 0、相對 −0.2）為設定值；A-C5-7 理想科系由最強需求決定 |
| 熟悉度生成 | Logistic 模型：知名度、與學生背景同分類、台灣常見分類 | A-C5-2 |
| 錨點選擇 | 離散選擇模型 multinomial logit（McFadden, 1974）：效用 = 契合度與名氣的加權 | A-C5-3 錨點只能從熟悉的學程中選 |
| 陳述需求的作答風格 | 默許傾向、社會期許（Podsakoff et al., 2003） | A-C5-4 偏誤可加成到真實需求上 |
| 熟悉度回報模擬 | 訊號偵測：固定漏報率與誤報率，學程之間獨立 | A-C5-8 |
| 真實效用（ground truth） | 兩種定義並列：加權歐氏距離（考慮強度）與 cosine（只看方向） | A-C5-5 真相函數的選擇會影響哪個估計器看起來較好（B-C5-1） |
| 排序一致性 | Kendall (1945), *Biometrika*：含同分的 τ-b | — |
| 分布差異 | Kullback & Leibler (1951) KL 散度（加平滑） | — |
| 機率評分 | Brier (1950) | — |
| 最差群體評估（D12） | 以最差群體表現作為選擇準則（group distributionally robust 的想法：Sagawa et al., 2020, *ICLR*）；本專案延伸既有的跨真相 maximin（F37）到背景分層 | S-8 |
| 背景分布校準（W-67） | 教育部統計處大專校院畢業生學門統計（官方檔案待下載） | A-C5-9 |
| 隱藏領域維度（W-67） | 量測空間只是真實偏好空間的投影：真相函數使用「量測 15 維 ⊕ 隱藏 6 維」，受測部件只看得到量測 15 維（遺漏維度造成的誤設；S-1 的「生成模型與估計模型不同」原則） | A-C5-10 |
| 遠領域假陽性指標（W-67） | 推薦清單中「只看量測維度時相關、加上隱藏領域後不相關」的比例 | A-C5-10 |
| v1／v2 比較世界（W-16c） | 同一個真相、兩種視角：真相 = v2 51 維 ⊕ 3 個殘差維度；v1 視角為 v2 的投影（v1 領域 = 對應 v2 領域的平均）；兩種視角都看不到殘差維度，避免 v2 依構造必勝（S-1） | A-C5-12 |
| 真相層權重（W-16c） | 加權距離／加權 cosine：每個座標乘上 1/√(該層維度數)，使 5 層（興趣、強度、領域、取向、殘差）等權；只用排序，所以與直接加權等價。預先登記的 54 維等權作為另一種權重 | A-C5-13、B-C5-8 |
| v2 計分器選擇（W-16c） | 延續 F37 的 maximin（Wald, 1950）與 D12 最差群體：分數 = 在「真相函數 × 真相層權重」4 格與各需求大領域群中，相對 v1 的配對差最差的值；另以「不劣於現行參考計分器」作為資格（相關且不知道、最佳契合保留） | A-C2-8、A-C2-9 |
| 轉換者類型（W-63） | 延續、相鄰轉換、遠距轉換、多學位；相鄰與遠距由大領域群區分；遠距與多學位者的熟悉集合與錨點來自舊領域 | A-C5-11 |

## C6 先修與申請條件差距

| 方法 | 理論／資料來源 | 引入的假設 |
|---|---|---|
| 條件類別與確定性比對 | 各學程官網公布的入學條件（C0.3 擷取，附原文）；類別表為本專案自訂、版本化（`requirement_taxonomy.json`） | A-C6-1、A-C6-2 |
| 學歷等值（C6，B-C6-5） | 以各學程官網規定的學歷認證方式為準（常見為 NACES 會員機構的學歷評估）；系統只呈現規定，不自行判斷等值 | A-C6-1 |
| 探索／申請模式分離 | 設計決策 D9（不假設申請時間） | A-C6-3 |

## 相關開源專案與資料

| 專案／資料 | 用途 |
|---|---|
| [maxkapur/OptimalApplication.jl](https://github.com/maxkapur/OptimalApplication.jl)、[maxkapur/CollegeApplication](https://github.com/maxkapur/CollegeApplication) | C4.1 的參考結果（測試已重現其 README 範例） |
| Kaggle "Graduate Admissions"（Acharya, 2018）相關 repo | 只作入門參考；目標變數為合成，不可用於 C3 校準 |
| Heidari Far & Tabrizi (2026), arXiv:2603.29881 | 約 13,000 筆 GradCafe 紀錄，結合 OpenAlex、QS、Wikidata |
| arXiv:2501.06581 | 以課程內容 + 個人偏好推薦學程 |
| [NCES CIP 2020 ↔ SOC 2018 crosswalk](https://nces.ed.gov/IPEDS/CIPCODE/Files/IES2020_CIP_SOC_Crosswalk_508C.pdf) | C0.2 |
| [O*NET Occupational Interest Profiles](https://www.onetcenter.org/reports/OIP.html) | C0.2、C1.1 |
