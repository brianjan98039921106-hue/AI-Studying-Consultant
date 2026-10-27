# 41 個細分興趣領域的分群（W-64 方案 B）

- 產生日期：2026-10-27 · 資料：O\*NET 31.0 Specific Interest Areas（891 個職業，等權）
- 預先登記：見 `scripts/cluster_interest_areas.py` 檔頭

## 判定

- 依輪廓係數選出 k = **12**（6–14 中最高）
- 自助法穩定度：ARI 平均 0.806（第 5–95 百分位 0.671–0.953），門檻 0.6 → ✅ 穩定，採用分群結果
- 備案分組（RIASEC）的輪廓係數：0.145

## 各 k 的平均輪廓係數

| k | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|
| 輪廓係數 | 0.223 | 0.270 | 0.259 | 0.276 | 0.311 | 0.315 | 0.331 | 0.319 | 0.312 |

## 採用的分組

| 群 | 領域（RIASEC 連結；領域輪廓係數） |
|---|---|
| 1 | Business Initiatives（E；0.52）；Sales（E；0.38）；Marketing/Advertising（A/E；0.01）；Finance（E/C；0.43）；Accounting（C；0.41）；Human Resources（S/C；0.05）；Office Work（C；0.26）；Management/Administration（E；0.36） |
| 2 | Humanities（I/A；-0.14）；Visual Arts（A；0.45）；Applied Arts and Design（A；0.48）；Performing Arts（A；0.44）；Music（A；0.31）；Creative Writing（A；0.27）；Media（A；0.49） |
| 3 | Social Science（I/S；0.42）；Teaching/Education（S；0.44）；Social Service（S；0.29）；Personal Service（S；0.09）；Professional Advising（S/E；0.52）；Public Speaking（E；0.36）；Politics（E；-0.07） |
| 4 | Agriculture（R；0.51）；Nature/Outdoors（R；0.40）；Animal Service（R/S；0.22） |
| 5 | Construction/Woodwork（R；0.11）；Transportation/Machine Operation（R；0.11）；Physical/Manual Labor（R；0.32） |
| 6 | Life Science（I；0.65）；Medical Science（I；0.82）；Health Care Service（I/S；0.62） |
| 7 | Physical Science（I；0.12）；Mathematics/Statistics（I/C；0.52）；Information Technology（C；0.29） |
| 8 | Mechanics/Electronics（R；0.72）；Engineering（R；0.65） |
| 9 | Protective Service（R；0.46）；Law（E；0.29） |
| 10 | Athletics（R/E；0.00） |
| 11 | Culinary Art（A/S；0.00） |
| 12 | Religious Activities（S/E；0.00） |

