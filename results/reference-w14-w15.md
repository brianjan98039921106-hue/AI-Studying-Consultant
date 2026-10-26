# 參考資料建置報告（W-14、W-15）

- 產生日期：2026-10-26 · IPEDS 2025 · O*NET unknown (ratings dated up to 02/2026) · 範圍 mvp-v1
- 來源與檔案雜湊：`data/reference/SOURCES.json`（下載日期 2026-10-26）

## W-14：範圍內 CIP 與學校

- 範圍內 6 碼 CIP：84；其中有學校頒發碩士學位的：77
- 「學校 × CIP」列數：4471；不同學校：1270（B-C0-7：還不是學程）
- 碩士學位總數（第一主修）：270,050

| 族群 | CIP 數 | 有碩士的學校數 | 碩士學位數 |
|---|---|---|---|
| business_management | 17 | 1133 | 116,438 |
| business_economics | 1 | 5 | 33 |
| entrepreneurship | 5 | 57 | 1,030 |
| finance | 9 | 138 | 2,961 |
| international_business | 1 | 55 | 1,164 |
| information_systems | 4 | 60 | 2,352 |
| management_science | 4 | 360 | 48,196 |
| marketing | 5 | 122 | 2,856 |
| data_science | 2 | 119 | 6,775 |
| data_analytics | 5 | 174 | 11,594 |
| statistics | 4 | 154 | 3,576 |
| applied_mathematics | 6 | 175 | 6,217 |
| information_science | 7 | 397 | 49,238 |
| systems_engineering | 1 | 71 | 2,437 |
| industrial_engineering | 1 | 89 | 2,610 |
| operations_research | 1 | 23 | 1,082 |
| engineering_management | 4 | 153 | 6,076 |
| systems_science | 1 | 25 | 599 |
| economics | 6 | 220 | 4,816 |

## W-15：職業橋接

- 有碩士學程的 CIP 中，取得職業剖面的：73 / 77
- 沒有任何 SOC 有 O*NET 資料（或對照表為 NO MATCH）的 CIP：15.1599, 52.0299, 52.0799, 52.1399
- SOC 覆蓋率（有 O*NET 資料的 SOC ÷ 對照的 SOC）中位數：1.00

### 區辨度檢查（B-C0-2：多對多對照會把剖面稀釋成平均值）

| 維度 | 平均 | 標準差（跨 CIP） | 最小 | 最大 |
|---|---|---|---|---|
| riasec_r | 0.249 | 0.144 | 0.036 | 0.678 |
| riasec_i | 0.606 | 0.210 | 0.125 | 0.891 |
| riasec_a | 0.192 | 0.104 | 0.000 | 0.738 |
| riasec_s | 0.373 | 0.110 | 0.150 | 0.603 |
| riasec_e | 0.543 | 0.222 | 0.214 | 0.950 |
| riasec_c | 0.682 | 0.082 | 0.486 | 0.945 |

