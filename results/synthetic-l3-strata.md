# L3 分層：C2 v1 對轉換領域的學生是否成立（W-68，D12）

- 產生日期：2026-10-27 · 指令：`python scripts/run_l3_strata.py --catalog-seed 0 --pop-seed 41 --seed 42`
- 每條件 400 位學生；完整陳述需求；分區（短名單 25 → 熟悉度 → 發現區）vs 只依契合
- 分層：continuing（大學背景 = 需求指向的族群）／switching（其他）；方括號 = 人數 < 30，不納入判定
- 預先登記：見 `scripts/run_l3_strata.py` 檔頭

## 判定：H-S1 ✅ 成立

## 分區 − 只依契合（配對差）

### novel_relevant_rate

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | +0.103±0.004 (n=400) | +0.114±0.006 (n=214) | +0.090±0.006 (n=186) | switching |
| archetype / distance / 10/5 | +0.079±0.005 (n=400) | +0.095±0.007 (n=214) | +0.061±0.007 (n=186) | switching |
| archetype / cosine / perfect | +0.101±0.005 (n=400) | +0.120±0.007 (n=183) | +0.085±0.006 (n=217) | switching |
| archetype / cosine / 10/5 | +0.069±0.005 (n=400) | +0.095±0.007 (n=183) | +0.046±0.007 (n=217) | switching |
| continuous / distance / perfect | +0.106±0.005 (n=400) | +0.110±0.007 (n=210) | +0.101±0.007 (n=190) | switching |
| continuous / distance / 10/5 | +0.079±0.005 (n=400) | +0.083±0.007 (n=210) | +0.074±0.007 (n=190) | switching |
| continuous / cosine / perfect | +0.110±0.005 (n=400) | +0.110±0.007 (n=199) | +0.109±0.007 (n=201) | switching |
| continuous / cosine / 10/5 | +0.081±0.005 (n=400) | +0.082±0.007 (n=199) | +0.079±0.008 (n=201) | switching |

### relevant_rate

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | -0.222±0.011 (n=400) | -0.297±0.016 (n=214) | -0.135±0.011 (n=186) | continuing |
| archetype / distance / 10/5 | -0.206±0.010 (n=400) | -0.257±0.015 (n=214) | -0.147±0.011 (n=186) | continuing |
| archetype / cosine / perfect | -0.207±0.010 (n=400) | -0.297±0.018 (n=183) | -0.131±0.009 (n=217) | continuing |
| archetype / cosine / 10/5 | -0.201±0.010 (n=400) | -0.272±0.017 (n=183) | -0.141±0.009 (n=217) | continuing |
| continuous / distance / perfect | -0.116±0.008 (n=400) | -0.120±0.011 (n=210) | -0.111±0.012 (n=190) | continuing |
| continuous / distance / 10/5 | -0.113±0.008 (n=400) | -0.112±0.011 (n=210) | -0.113±0.012 (n=190) | switching |
| continuous / cosine / perfect | -0.128±0.009 (n=400) | -0.143±0.012 (n=199) | -0.113±0.012 (n=201) | continuing |
| continuous / cosine / 10/5 | -0.125±0.008 (n=400) | -0.132±0.011 (n=199) | -0.118±0.012 (n=201) | continuing |

### best_fit_retained

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | +0.033±0.010 (n=400) | +0.033±0.014 (n=214) | +0.032±0.013 (n=186) | switching |
| archetype / distance / 10/5 | +0.033±0.010 (n=400) | +0.028±0.015 (n=214) | +0.038±0.014 (n=186) | continuing |
| archetype / cosine / perfect | +0.010±0.008 (n=400) | +0.016±0.014 (n=183) | +0.005±0.008 (n=217) | switching |
| archetype / cosine / 10/5 | +0.020±0.008 (n=400) | +0.027±0.014 (n=183) | +0.014±0.008 (n=217) | switching |
| continuous / distance / perfect | +0.025±0.010 (n=400) | +0.014±0.013 (n=210) | +0.037±0.016 (n=190) | continuing |
| continuous / distance / 10/5 | +0.030±0.011 (n=400) | +0.029±0.013 (n=210) | +0.032±0.017 (n=190) | continuing |
| continuous / cosine / perfect | +0.040±0.010 (n=400) | +0.030±0.014 (n=199) | +0.050±0.015 (n=201) | continuing |
| continuous / cosine / 10/5 | +0.040±0.011 (n=400) | +0.035±0.017 (n=199) | +0.045±0.015 (n=201) | continuing |

## 各方法的水準（只報告）

### novel_relevant_rate

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.575±0.010 (n=400) | 0.476±0.012 (n=214) | 0.689±0.013 (n=186) |
| archetype / distance / perfect | fit-only | 0.472±0.011 (n=400) | 0.362±0.012 (n=214) | 0.598±0.015 (n=186) |
| archetype / distance / 10/5 | sections | 0.551±0.010 (n=400) | 0.457±0.012 (n=214) | 0.659±0.013 (n=186) |
| archetype / distance / 10/5 | fit-only | 0.472±0.011 (n=400) | 0.362±0.012 (n=214) | 0.598±0.015 (n=186) |
| archetype / cosine / perfect | sections | 0.617±0.011 (n=400) | 0.498±0.015 (n=183) | 0.717±0.014 (n=217) |
| archetype / cosine / perfect | fit-only | 0.516±0.012 (n=400) | 0.379±0.014 (n=183) | 0.631±0.015 (n=217) |
| archetype / cosine / 10/5 | sections | 0.584±0.011 (n=400) | 0.474±0.014 (n=183) | 0.677±0.014 (n=217) |
| archetype / cosine / 10/5 | fit-only | 0.516±0.012 (n=400) | 0.379±0.014 (n=183) | 0.631±0.015 (n=217) |
| continuous / distance / perfect | sections | 0.484±0.009 (n=400) | 0.472±0.012 (n=210) | 0.498±0.014 (n=190) |
| continuous / distance / perfect | fit-only | 0.379±0.010 (n=400) | 0.362±0.013 (n=210) | 0.397±0.015 (n=190) |
| continuous / distance / 10/5 | sections | 0.458±0.009 (n=400) | 0.446±0.012 (n=210) | 0.471±0.013 (n=190) |
| continuous / distance / 10/5 | fit-only | 0.379±0.010 (n=400) | 0.362±0.013 (n=210) | 0.397±0.015 (n=190) |
| continuous / cosine / perfect | sections | 0.503±0.009 (n=400) | 0.483±0.013 (n=199) | 0.523±0.013 (n=201) |
| continuous / cosine / perfect | fit-only | 0.393±0.010 (n=400) | 0.373±0.014 (n=199) | 0.413±0.014 (n=201) |
| continuous / cosine / 10/5 | sections | 0.474±0.009 (n=400) | 0.455±0.013 (n=199) | 0.493±0.012 (n=201) |
| continuous / cosine / 10/5 | fit-only | 0.393±0.010 (n=400) | 0.373±0.014 (n=199) | 0.413±0.014 (n=201) |

### relevant_rate

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.575±0.010 (n=400) | 0.477±0.012 (n=214) | 0.689±0.013 (n=186) |
| archetype / distance / perfect | fit-only | 0.797±0.010 (n=400) | 0.773±0.014 (n=214) | 0.824±0.015 (n=186) |
| archetype / distance / 10/5 | sections | 0.591±0.010 (n=400) | 0.516±0.012 (n=214) | 0.677±0.013 (n=186) |
| archetype / distance / 10/5 | fit-only | 0.797±0.010 (n=400) | 0.773±0.014 (n=214) | 0.824±0.015 (n=186) |
| archetype / cosine / perfect | sections | 0.617±0.011 (n=400) | 0.498±0.015 (n=183) | 0.717±0.014 (n=217) |
| archetype / cosine / perfect | fit-only | 0.824±0.010 (n=400) | 0.796±0.015 (n=183) | 0.847±0.012 (n=217) |
| archetype / cosine / 10/5 | sections | 0.623±0.011 (n=400) | 0.523±0.014 (n=183) | 0.706±0.013 (n=217) |
| archetype / cosine / 10/5 | fit-only | 0.824±0.010 (n=400) | 0.796±0.015 (n=183) | 0.847±0.012 (n=217) |
| continuous / distance / perfect | sections | 0.484±0.009 (n=400) | 0.472±0.012 (n=210) | 0.498±0.014 (n=190) |
| continuous / distance / perfect | fit-only | 0.600±0.010 (n=400) | 0.592±0.013 (n=210) | 0.609±0.014 (n=190) |
| continuous / distance / 10/5 | sections | 0.487±0.009 (n=400) | 0.480±0.012 (n=210) | 0.496±0.013 (n=190) |
| continuous / distance / 10/5 | fit-only | 0.600±0.010 (n=400) | 0.592±0.013 (n=210) | 0.609±0.014 (n=190) |
| continuous / cosine / perfect | sections | 0.504±0.009 (n=400) | 0.484±0.013 (n=199) | 0.523±0.013 (n=201) |
| continuous / cosine / perfect | fit-only | 0.632±0.010 (n=400) | 0.627±0.014 (n=199) | 0.636±0.015 (n=201) |
| continuous / cosine / 10/5 | sections | 0.507±0.009 (n=400) | 0.495±0.013 (n=199) | 0.518±0.012 (n=201) |
| continuous / cosine / 10/5 | fit-only | 0.632±0.010 (n=400) | 0.627±0.014 (n=199) | 0.636±0.015 (n=201) |

### best_fit_retained

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.940±0.012 (n=400) | 0.949±0.015 (n=214) | 0.930±0.019 (n=186) |
| archetype / distance / perfect | fit-only | 0.907±0.015 (n=400) | 0.916±0.019 (n=214) | 0.898±0.022 (n=186) |
| archetype / distance / 10/5 | sections | 0.940±0.012 (n=400) | 0.944±0.016 (n=214) | 0.935±0.018 (n=186) |
| archetype / distance / 10/5 | fit-only | 0.907±0.015 (n=400) | 0.916±0.019 (n=214) | 0.898±0.022 (n=186) |
| archetype / cosine / perfect | sections | 0.945±0.011 (n=400) | 0.929±0.019 (n=183) | 0.959±0.014 (n=217) |
| archetype / cosine / perfect | fit-only | 0.935±0.012 (n=400) | 0.913±0.021 (n=183) | 0.954±0.014 (n=217) |
| archetype / cosine / 10/5 | sections | 0.955±0.010 (n=400) | 0.940±0.018 (n=183) | 0.968±0.012 (n=217) |
| archetype / cosine / 10/5 | fit-only | 0.935±0.012 (n=400) | 0.913±0.021 (n=183) | 0.954±0.014 (n=217) |
| continuous / distance / perfect | sections | 0.877±0.016 (n=400) | 0.890±0.022 (n=210) | 0.863±0.025 (n=190) |
| continuous / distance / perfect | fit-only | 0.853±0.018 (n=400) | 0.876±0.023 (n=210) | 0.826±0.028 (n=190) |
| continuous / distance / 10/5 | sections | 0.882±0.016 (n=400) | 0.905±0.020 (n=210) | 0.858±0.025 (n=190) |
| continuous / distance / 10/5 | fit-only | 0.853±0.018 (n=400) | 0.876±0.023 (n=210) | 0.826±0.028 (n=190) |
| continuous / cosine / perfect | sections | 0.912±0.014 (n=400) | 0.915±0.020 (n=199) | 0.910±0.020 (n=201) |
| continuous / cosine / perfect | fit-only | 0.873±0.017 (n=400) | 0.884±0.023 (n=199) | 0.861±0.024 (n=201) |
| continuous / cosine / 10/5 | sections | 0.912±0.014 (n=400) | 0.920±0.019 (n=199) | 0.905±0.021 (n=201) |
| continuous / cosine / 10/5 | fit-only | 0.873±0.017 (n=400) | 0.884±0.023 (n=199) | 0.861±0.024 (n=201) |

