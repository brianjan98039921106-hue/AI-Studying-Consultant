# W-21 課程擷取前導 v0（30 個 W-20 候選學程）

> 狀態：**前導，未經驗證**（2026-10-28）。資料：[`data/curriculum/w21_pilot_v0.jsonl`](../data/curriculum/w21_pilot_v0.jsonl)。
> 尚未與 W-20 黃金標準比對（V-C0-2），**不可用於推薦**；擷取引文也還沒逐字核對（`quote_verified: false`）。

## 1. 做了什麼

| 項目 | 內容 |
|---|---|
| 學程 | [w20-candidates.md](../results/w20-candidates.md) 的 30 個 CIP；每個 CIP 依清單順序取第一個能讀到官網課程頁的學校（2 個改用第二順位：USC → New York Film Academy、UW-Stout → UIUC） |
| 維度 | 真實目錄 v0 沒有測量的 4 個維度：數理強度、程式強度、研究取向、創業取向（B-C0-18） |
| 準則 | [rubric.md](rubric.md) §4 的錨點，0／0.25／0.5／0.75／1；每個分數附原文片段（≤ 15 字）與網址；沒有證據記為缺值 |
| 執行 | 5 個 LLM 子代理平行執行，只讀學校官網（`program_url`、`urls_fetched`），取得日期 2026-10-28 |
| W-20 的用途 | 這 30 個頁面就是 W-20 評分者要評的頁面，兩者比對同一份證據；評分者**不可**看到本檔分數 |

## 2. 覆蓋率

| 維度 | 有分數 | 缺值 | 分數分布 |
|---|---|---|---|
| 數理強度 | 25 | 5 | 0：19、0.25：3、0.5：1、0.75：1、1：1 |
| 程式強度 | 27 | 3 | 0：24、0.5：2、1：1 |
| 研究取向 | 27 | 3 | 0：6、0.25：4、0.5：9、0.75：4、1：4 |
| 創業取向 | 8 | 22 | 0.25：4、0.5：2、0.75：2 |

## 3. 發現的問題（給 W-20 與評分準則修訂）

| 問題 | 例子 | 影響 | 登錄 |
|---|---|---|---|
| **0 分的證據是「沒有」**，無法用原文片段佐證；目前只能附課表中的一行 | 多數學程的數理、程式 0 分 | 0 分與「沒找到」難以區分；V-C0-2 要另訂 0 分的佐證規則 | B-C0-19 |
| **創作型畢業作品**（小說、論文影片）被準則判為「必須完成論文 = 1」 | SNHU MFA、NYFA MFA | 研究取向高估創作型學位 | B-C0-19 |
| **創業取向的證據多在學校資源頁**，不在學程頁 | 22／30 缺值 | 只讀學程頁時創業取向幾乎都缺 | B-C0-19 |
| 課表只列部分課程（「courses may include」） | SNHU、ACE、WGU | 缺值增加或 0 分不可靠 | — |
| 部分官網被讀取工具阻擋，改用舊版課程目錄或 PDF | USC、IU、Georgia Tech、UWS | 可能過時（B-C0-5） | — |
| 引文經過網頁讀取工具的摘要模型，未能逐字核對 | 全部 | 引文可能被改寫；正式版需保存原始頁面文字比對 | A-C0-5 |

## 4. 逐筆結果（—＝缺值）

| CIP | 學校 | 學程（官網名稱） | 數理 | 程式 | 研究 | 創業 |
|---|---|---|---|---|---|---|
| 52.0201 | Western Governors University | [Master of Business Administration](https://www.wgu.edu/online-business-degrees/mba-masters-business-administration-program.html) | 0.25 | 0 | 0 | 0.75 |
| 23.1302 | Southern New Hampshire University | [MFA in Creative Writing](https://www.snhu.edu/online-degrees/masters/mfa-in-creative-writing) | — | — | 1 | 0.5 |
| 44.0701 | Walden University | [Master of Social Work (MSW)](https://academics.waldenu.edu/catalog/social-behavioral-health/master/social-work) | 0 | 0 | 0.25 | — |
| 01.8001 | Ohio State University-Main Campus | [Doctor of Veterinary Medicine (DVM)](https://vet.osu.edu/education/programs/dvm) | — | 0 | 0.5 | 0.25 |
| 51.1201 | Indiana University-Indianapolis | [Doctor of Medicine](https://cdn.medicine.iu.edu/md) | — | 0 | 1 | 0.25 |
| 11.0701 | University of Central Missouri | [Master of Science in Computer Science](https://ucmo.edu/academics/programs/masters-degrees/computer-science-masters/index.php) | 0.5 | 1 | 0.5 | — |
| 14.1001 | Georgia Institute of Technology-Main Campus | [The Masters of Science Degree](https://ece.gatech.edu/academics/graduate/graduate-student-handbook/master-of-science-degree) | — | — | 0.75 | 0.75 |
| 22.0101 | Georgetown University | [J.D.](https://www.law.georgetown.edu/academics/jd-program/full-time-program/) | 0 | 0 | 0.75 | 0.25 |
| 31.0504 | Concordia University-Irvine | [M.A. and M.S. Coaching & Athletics Administration](https://www.cui.edu/academicprograms/graduate/coaching/landing/curriculum) | 0 | 0 | 0.5 | — |
| 39.0602 | Liberty University | [Master of Divinity (M.Div.)](https://www.liberty.edu/registrar/wp-content/uploads/sites/119/MDGE-MDV-23-24.pdf) | 0 | 0 | 0.25 | — |
| 50.0702 | School of the Art Institute of Chicago | [Master of Fine Arts (MFA)](https://saic.edu/painting-drawing/graduate/curriculum-courses) | 0 | 0 | 0.5 | — |
| 03.0101 | University of Michigan-Ann Arbor | [Master of Science (Environmental Policy and Planning)](https://seas.umich.edu/academics/master-science/environmental-policy-and-planning/requirements-courses) | 0.25 | — | 0.75 | — |
| 50.0903 | Indiana University-Bloomington | [Master of Music in Performance (Piano)](https://bulletins.iu.edu/iub/music/2020-2021/graduate/curricula/index.shtml) | 0 | 0 | 0 | 0.25 |
| 30.2001 | American Public University System | [Online Master of Arts in International Relations and Global Security (MA)](https://www.apu.apus.edu/online-masters-programs/master-of-arts-in-international-relations-and-global-security) | 0 | 0 | 0.5 | — |
| 09.0900 | Georgetown University | [Master of Professional Studies in Public Relations and Corporate Communications](https://scs.georgetown.edu/programs/70/master-of-professional-studies-in-public-relations-corporate-communications/curriculum) | 0 | 0 | 0.25 | — |
| 13.1315 | American College of Education | [M.Ed. in Literacy](https://www.ace.edu/programs/master-of-education-in-literacy) | 0 | 0 | — | — |
| 01.1001 | University of Illinois Urbana-Champaign | [Food Science & Human Nutrition: Food Science, MS](https://catalog.illinois.edu/graduate/aces/food-science-human-nutrition-ms/food-science) | 0 | 0 | 0.5 | — |
| 51.0913 | University of South Carolina-Columbia | [Master of Science in Athletic Training](https://www.sc.edu/study/colleges_schools/public_health/degrees-programs/masters/msat/index.php) | 0 | 0 | 0 | — |
| 30.1401 | Johns Hopkins University | [Master's in Museum Studies (MA)](https://advanced.jhu.edu/academics/graduate/ma-museum-studies/) | 0 | 0 | 0.5 | — |
| 50.0602 | New York Film Academy | [Master of Fine Arts in Filmmaking](https://www.nyfa.edu/mfa/filmmaking) | 0 | 0 | 1 | 0.5 |
| 52.1005 | Louisiana State University and Agricultural & Mechanical College | [Master of Science in Leadership & Human Resource Development](https://lsu.edu/chse/slhrd/programs/graduate/ms-lhrd.php) | 0.25 | 0 | 0 | — |
| 52.0301 | Western Governors University | [Accounting – M.S.](https://www.wgu.edu/online-business-degrees/accounting-masters-program.html) | — | 0.5 | — | — |
| 13.1099 | University of St Thomas | [MA in Educational Studies with Special Education Concentration](https://education.stthomas.edu/teacher-degrees-licensure/k-12/ma-educational-studies/special-education/) | 0 | 0 | — | — |
| 40.0801 | Yale University | [M.S. (en route to the Ph.D)](https://catalog.yale.edu/gsas/degree-granting-departments-programs/physics) | 1 | 0 | 1 | — |
| 50.0901 | Central Washington University | [Master of Music](https://www.cwu.edu/academics/explore-programs/music-master.php) | 0 | 0 | 0.5 | — |
| 51.2001 | Massachusetts College of Pharmacy and Health Sciences | [Doctor of Pharmacy (PharmD)—Accelerated](https://www.mcphs.edu/academics/programs/pharmacy-pharmd-accelerated) | 0 | 0 | 0 | — |
| 04.0201 | University of South Florida | [Master of Architecture (M. Arch.)](https://usf.edu/arts/architecture/academics/architecture-program/master-of-architecture.aspx) | 0 | 0 | 0.75 | — |
| 51.3102 | University of Western States | [M.S. in Clinical Nutrition and Functional Medicine](https://www.uws.edu/academics/programs/clinical-nutrition-functional-medicine/ms/) | 0 | 0 | 0.25 | — |
| 43.0104 | Liberty University | [Criminal Justice (M.S.)](https://catalog.liberty.edu/graduate/colleges-schools/helms-government/criminal-justice-ms/criminal-justice-ms/) | 0 | 0 | 0.5 | — |
| 27.0305 | Washington University in St Louis | [Master of Science in Finance - Quantitative (MSFQ)](https://olin.washu.edu/programs/specialized-masters/ms-in-quantitative-finance/curriculum.php) | 0.75 | 0.5 | 0 | — |

## 5. 下一步

| 順序 | 事項 |
|---|---|
| 1 | W-20：兩位評分者獨立評同樣 30 個頁面（不看本檔），計算 κ |
| 2 | 依 §3 修訂評分準則：0 分佐證規則、創作型作品、創業取向的證據來源 |
| 3 | 逐字核對引文（人工打開網址比對），通過才把 `quote_verified` 改為 true |
| 4 | V-C0-2：擷取 vs 黃金標準一致率 ≥ 90% 才擴充到全目錄，並經 W-22 融合進真實目錄 |
