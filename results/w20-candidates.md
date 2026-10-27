# W-20／W-76 候選學程（全領域、跨領域分散）

- 產生日期：2026-10-28 · 指令：`python scripts/select_w20_candidates.py`
- 合格 CIP：243（非學術剖面、碩士 ≥ 500 或專業博士 ≥ 200）；選出 30 個；涵蓋 10 / 12 個領域群
- 41 領域向量的兩兩距離：中位數 1.650、最小 0.871（同數量隨機抽樣：中位數 1.446、最小 0.220）
- 每個 CIP 列出學位數最多的 3 所學校，供評分者挑一個學程頁面；規則見腳本檔頭

| CIP | 名稱 | 群 | 學位層級 | 學位數 | 前 3 領域 | 學校選項 |
|---|---|---|---|---|---|---|
| 52.0201 | Business Administration and Management, General | 1 經營、銷售、金融與行政 | 7 | 99,764 | management_administration human_resources business_initiatives | Western Governors University (UT, 3858) | Southern New Hampshire University (NH, 2955) | Louisiana State University-Shreveport (LA, 2658) |
| 23.1302 | Creative Writing | 2 藝術、設計、媒體與人文 | 7 | 3,121 | creative_writing media humanities | Southern New Hampshire University (NH, 138) | Lindenwood University (MO, 109) | Columbia University in the City of New York (NY, 106) |
| 44.0701 | Social Work | 3 教育、社會服務與社會科學 | 7 | 34,898 | social_service professional_advising social_science | Walden University (MN, 1180) | University of Kentucky (KY, 840) | Rutgers University-New Brunswick (NJ, 663) |
| 01.8001 | Veterinary Medicine | 4 農業、自然與動物 | 18 | 3,818 | animal_service life_science medical_science | Ohio State University-Main Campus (OH, 164) | Texas A&M University-College Station (TX, 162) | Iowa State University (IA, 149) |
| 51.1201 | Medicine | 6 生命科學、醫學與醫療照護 | 18 | 21,394 | health_care_service medical_science life_science | Indiana University-Indianapolis (IN, 354) | Rutgers University-New Brunswick (NJ, 328) | Wayne State University (MI, 309) |
| 11.0701 | Computer Science | 7 物理科學、數學統計與資訊科技 | 7 | 27,199 | information_technology mathematics_statistics engineering | University of Central Missouri (MO, 1535) | University of Illinois Urbana-Champaign (IL, 1131) | University of Southern California (CA, 1126) |
| 14.1001 | Electrical and Electronics Engineering | 8 機電與工程 | 7 | 8,538 | engineering mechanics_electronics information_technology | Georgia Institute of Technology-Main Campus (GA, 454) | Carnegie Mellon University (PA, 374) | Purdue University-Main Campus (IN, 346) |
| 22.0101 | Law | 9 法律與公共安全 | 18 | 36,895 | law public_speaking management_administration | Georgetown University (DC, 703) | Harvard University (MA, 603) | George Washington University (DC, 569) |
| 31.0504 | Sport and Fitness Administration/Management | 10 運動 | 7 | 5,027 | athletics teaching_education personal_service | Concordia University-Irvine (CA, 232) | Georgetown University (DC, 161) | Southeastern Oklahoma State University (OK, 159) |
| 39.0602 | Divinity/Ministry | 12 宗教活動 | 7 | 5,065 | religious_activities social_service public_speaking | Liberty University (VA, 531) | The Southern Baptist Theological Seminary (KY, 341) | Southeastern Baptist Theological Seminary (NC, 196) |
| 50.0702 | Fine/Studio Arts, General | 2 藝術、設計、媒體與人文 | 7 | 1,180 | visual_arts applied_arts_and_design marketing_advertising | School of the Art Institute of Chicago (IL, 142) | Kennesaw State University (GA, 55) | Pratt Institute-Main (NY, 53) |
| 03.0101 | Natural Resources/Conservation, General | 4 農業、自然與動物 | 7 | 861 | nature_outdoors agriculture life_science | University of Michigan-Ann Arbor (MI, 227) | Duke University (NC, 116) | Colorado State University-Fort Collins (CO, 54) |
| 50.0903 | Music Performance, General | 2 藝術、設計、媒體與人文 | 7 | 2,206 | music performing_arts media | Indiana University-Bloomington (IN, 174) | Johns Hopkins University (MD, 108) | University of Cincinnati-Main Campus (OH, 102) |
| 30.2001 | International/Globalization Studies | 3 教育、社會服務與社會科學 | 7 | 753 | social_science politics humanities | American Public University System (WV, 130) | University of Pennsylvania (PA, 89) | Northeastern University Professional Programs (MA, 73) |
| 09.0900 | Public Relations, Advertising, and Applied Communication | 1 經營、銷售、金融與行政 | 7 | 754 | marketing_advertising public_speaking management_administration | Georgetown University (DC, 96) | University of Southern California (CA, 91) | Florida State University (FL, 82) |
| 13.1315 | Reading Teacher Education | 3 教育、社會服務與社會科學 | 7 | 4,958 | teaching_education social_service public_speaking | American College of Education (IN, 639) | Grand Canyon University (AZ, 507) | SUNY Oneonta (NY, 144) |
| 01.1001 | Food Science | 6 生命科學、醫學與醫療照護 | 7 | 531 | life_science physical_science mathematics_statistics | University of Wisconsin-Stout (WI, 54) | University of Illinois Urbana-Champaign (IL, 51) | Illinois Institute of Technology (IL, 47) |
| 51.0913 | Athletic Training/Trainer | 10 運動 | 7 | 2,300 | athletics health_care_service teaching_education | University of South Carolina-Columbia (SC, 46) | Northern Arizona University (AZ, 32) | University of Delaware (DE, 27) |
| 30.1401 | Museology/Museum Studies | 2 藝術、設計、媒體與人文 | 7 | 542 | humanities management_administration office_work | Johns Hopkins University (MD, 75) | University of Oklahoma-Norman Campus (OK, 72) | Harvard University (MA, 48) |
| 50.0602 | Cinematography and Film/Video Production | 2 藝術、設計、媒體與人文 | 7 | 986 | media applied_arts_and_design visual_arts | University of Southern California (CA, 143) | New York Film Academy (CA, 118) | New York University (NY, 93) |
| 52.1005 | Human Resources Development | 3 教育、社會服務與社會科學 | 7 | 750 | professional_advising teaching_education human_resources | Louisiana State University and Agricultural & Mechanical College (LA, 161) | Villanova University (PA, 81) | Columbia Southern University (AL, 61) |
| 52.0301 | Accounting | 1 經營、銷售、金融與行政 | 7 | 11,486 | accounting finance office_work | Western Governors University (UT, 490) | Southern New Hampshire University (NH, 308) | CUNY Bernard M Baruch College (NY, 196) |
| 13.1099 | Special Education and Teaching, Other | 3 教育、社會服務與社會科學 | 7 | 628 | teaching_education social_service athletics | University of St Thomas (MN, 98) | Arkansas State University (AR, 38) | Molloy University (NY, 34) |
| 40.0801 | Physics, General | 7 物理科學、數學統計與資訊科技 | 7 | 2,076 | mathematics_statistics physical_science teaching_education | Yale University (CT, 63) | Brown University (RI, 56) | University of Washington-Seattle Campus (WA, 46) |
| 50.0901 | Music, General | 3 教育、社會服務與社會科學 | 7 | 1,628 | public_speaking teaching_education music | Central Washington University (WA, 49) | The University of Texas at Austin (TX, 37) | University of Maryland-College Park (MD, 36) |
| 51.2001 | Pharmacy | 6 生命科學、醫學與醫療照護 | 18 | 11,098 | health_care_service management_administration life_science | Massachusetts College of Pharmacy and Health Sciences (MA, 297) | Lake Erie College of Osteopathic Medicine (PA, 233) | Nova Southeastern University (FL, 222) |
| 04.0201 | Architecture | 8 機電與工程 | 7 | 572 | engineering management_administration business_initiatives | University of South Florida (FL, 53) | North Dakota State University-Main Campus (ND, 49) | University of Puerto Rico-Rio Piedras (PR, 47) |
| 51.3102 | Clinical Nutrition/Nutritionist | 6 生命科學、醫學與醫療照護 | 7 | 711 | health_care_service professional_advising medical_science | University of Western States (OR, 104) | Logan University (MO, 66) | Sonoran University of Health Sciences (AZ, 65) |
| 43.0104 | Criminal Justice/Safety Studies | 9 法律與公共安全 | 7 | 3,041 | protective_service management_administration law | Liberty University (VA, 419) | American Public University System (WV, 158) | Grand Canyon University (AZ, 121) |
| 27.0305 | Financial Mathematics | 7 物理科學、數學統計與資訊科技 | 7 | 4,896 | mathematics_statistics finance information_technology | Washington University in St Louis (MO, 269) | University of Illinois Urbana-Champaign (IL, 177) | Northeastern University (MA, 164) |
