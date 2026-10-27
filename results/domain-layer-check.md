# W-64 真實資料檢查：職業橋接能否區分領域

- 產生日期：2026-10-27
- 資料：IPEDS 2025（award level 7／17／18，第一主修）、CIP 2020–SOC 2018 對照表、O\*NET 31.0（Career Interest Types、Specific Interest Areas）
- 單位：1101 個 6 碼 CIP（全領域，不限 MVP），35 個 2 碼族群
- 預先登記：見 `scripts/check_domain_layer.py` 檔頭

## 判定：H-D1 ✅ · H-D2 ✅

| 空間 | 最近 10 個 CIP 中同族群的比例（M1） | 族群輪廓係數（silhouette） |
|---|---|---|
| S1 RIASEC | 0.612 ± 0.011 | 0.000 |
| S2 41 areas | 0.759 ± 0.010 | 0.169 |
| S3 both | 0.734 ± 0.010 | 0.118 |

- 配對差（跨 CIP）：S2 − S1 +0.148 ± 0.007；S3 − S1 +0.122 ± 0.006

## 依 2 碼族群（M1，只報告；CIP 數 ≥ 5 的族群）

| 族群 | CIP 數 | S1 | S2 | S3 |
|---|---|---|---|---|
| 01 | 50 | 0.57 | 0.82 | 0.74 |
| 03 | 21 | 0.20 | 0.50 | 0.38 |
| 04 | 13 | 0.45 | 0.55 | 0.56 |
| 05 | 39 | 0.97 | 0.97 | 0.97 |
| 09 | 22 | 0.42 | 0.56 | 0.49 |
| 11 | 24 | 0.43 | 0.76 | 0.63 |
| 13 | 89 | 0.81 | 0.88 | 0.88 |
| 14 | 56 | 0.86 | 0.91 | 0.90 |
| 15 | 37 | 0.65 | 0.72 | 0.75 |
| 16 | 44 | 0.94 | 0.95 | 0.95 |
| 19 | 23 | 0.40 | 0.32 | 0.37 |
| 22 | 23 | 0.80 | 0.90 | 0.90 |
| 23 | 11 | 0.47 | 0.48 | 0.48 |
| 26 | 87 | 0.66 | 0.93 | 0.93 |
| 27 | 15 | 0.50 | 0.77 | 0.67 |
| 30 | 50 | 0.19 | 0.27 | 0.22 |
| 31 | 9 | 0.08 | 0.23 | 0.14 |
| 38 | 14 | 0.64 | 0.64 | 0.64 |
| 39 | 18 | 0.51 | 0.89 | 0.87 |
| 40 | 40 | 0.46 | 0.90 | 0.88 |
| 42 | 29 | 0.73 | 0.86 | 0.83 |
| 43 | 26 | 0.32 | 0.83 | 0.68 |
| 44 | 12 | 0.07 | 0.14 | 0.07 |
| 45 | 34 | 0.41 | 0.62 | 0.61 |
| 50 | 64 | 0.85 | 0.91 | 0.88 |
| 51 | 159 | 0.68 | 0.81 | 0.80 |
| 52 | 68 | 0.54 | 0.69 | 0.66 |
| 54 | 7 | 0.29 | 0.40 | 0.37 |

## 例子：最近的 5 個 CIP（只報告）

### 27.0501 Statistics, General

- S1 RIASEC：27.0601 Applied Statistics, General；27.0503 Mathematics and Statistics；45.0603 Econometrics and Quantitative Economics；27.0599 Statistics, Other；27.0305 Financial Mathematics
- S3 both：27.0601 Applied Statistics, General；27.0503 Mathematics and Statistics；27.0599 Statistics, Other；27.0305 Financial Mathematics；45.0603 Econometrics and Quantitative Economics

### 26.0101 Biology/Biological Sciences, General

- S1 RIASEC：40.0501 Chemistry, General；26.0502 Microbiology, General；26.1006 Environmental Toxicology；01.1001 Food Science；01.1099 Food Science and Technology, Other
- S3 both：26.0502 Microbiology, General；40.0501 Chemistry, General；26.1006 Environmental Toxicology；26.0202 Biochemistry；01.1001 Food Science

### 51.0401 Dentistry

- S1 RIASEC：51.0502 Advanced General Dentistry；51.0509 Pediatric Dentistry/Pedodontics；51.0507 Oral/Maxillofacial Surgery；51.0508 Orthodontics/Orthodontology；51.0901 Cardiovascular Technology/Technologist
- S3 both：51.0502 Advanced General Dentistry；51.0509 Pediatric Dentistry/Pedodontics；51.0507 Oral/Maxillofacial Surgery；51.0508 Orthodontics/Orthodontology；51.0511 Prosthodontics/Prosthodontology

### 30.7102 Business Analytics

- S1 RIASEC：22.0000 Legal Studies；30.7103 Data Visualization；52.0601 Business/Managerial Economics；27.0601 Applied Statistics, General；27.0305 Financial Mathematics
- S3 both：30.7103 Data Visualization；27.0601 Applied Statistics, General；30.7101 Data Analytics, General；30.7199 Data Analytics, Other；27.0305 Financial Mathematics

### 50.0401 Design and Visual Communications, General

- S1 RIASEC：50.0404 Industrial and Product Design；50.0409 Graphic Design；50.0102 Digital Arts；50.0706 Intermedia/Multimedia；11.0801 Web Page, Digital/Multimedia and Information Resources Design
- S3 both：50.0404 Industrial and Product Design；50.0409 Graphic Design；50.0102 Digital Arts；50.0706 Intermedia/Multimedia；11.0801 Web Page, Digital/Multimedia and Information Resources Design

### 22.0101 Law

- S1 RIASEC：42.2804 Industrial and Organizational Psychology；52.1402 Marketing Research；52.0807 Investments and Securities；52.1403 International Marketing；52.1499 Marketing, Other
- S3 both：22.0202 Programs for Foreign Lawyers；22.0201 Advanced Legal Research/Studies, General；22.0203 American/U.S. Law/Legal Studies/Jurisprudence；22.0205 Banking, Corporate, Finance, and Securities Law；22.0206 Comparative Law

