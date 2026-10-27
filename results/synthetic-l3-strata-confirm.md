# L3 分層：C2 v1 對轉換領域的學生是否成立（W-68，D12）

- 產生日期：2026-10-27 · 指令：`python scripts/run_l3_strata.py --catalog-seed 600 --pop-seed 601 --seed 602`
- 每條件 400 位學生；完整陳述需求；分區（短名單 25 → 熟悉度 → 發現區）vs 只依契合
- 分層：continuing（大學背景 = 需求指向的族群）／switching（其他）；方括號 = 人數 < 30，不納入判定
- 預先登記：見 `scripts/run_l3_strata.py` 檔頭

## 判定：H-S1 ✅ 成立

## 分區 − 只依契合（配對差）

### novel_relevant_rate

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | +0.097±0.004 (n=400) | +0.100±0.006 (n=209) | +0.093±0.007 (n=191) | switching |
| archetype / distance / 10/5 | +0.067±0.005 (n=400) | +0.076±0.006 (n=209) | +0.057±0.008 (n=191) | switching |
| archetype / cosine / perfect | +0.094±0.004 (n=400) | +0.097±0.006 (n=210) | +0.090±0.006 (n=190) | switching |
| archetype / cosine / 10/5 | +0.062±0.005 (n=400) | +0.073±0.006 (n=210) | +0.049±0.007 (n=190) | switching |
| continuous / distance / perfect | +0.098±0.004 (n=400) | +0.099±0.006 (n=202) | +0.096±0.006 (n=198) | switching |
| continuous / distance / 10/5 | +0.074±0.005 (n=400) | +0.073±0.006 (n=202) | +0.075±0.007 (n=198) | continuing |
| continuous / cosine / perfect | +0.103±0.005 (n=400) | +0.096±0.006 (n=207) | +0.111±0.007 (n=193) | continuing |
| continuous / cosine / 10/5 | +0.074±0.005 (n=400) | +0.071±0.007 (n=207) | +0.079±0.008 (n=193) | continuing |

### relevant_rate

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | -0.235±0.011 (n=400) | -0.313±0.017 (n=209) | -0.149±0.012 (n=191) | continuing |
| archetype / distance / 10/5 | -0.223±0.011 (n=400) | -0.285±0.016 (n=209) | -0.154±0.012 (n=191) | continuing |
| archetype / cosine / perfect | -0.226±0.010 (n=400) | -0.311±0.015 (n=210) | -0.133±0.011 (n=190) | continuing |
| archetype / cosine / 10/5 | -0.218±0.010 (n=400) | -0.280±0.015 (n=210) | -0.151±0.010 (n=190) | continuing |
| continuous / distance / perfect | -0.103±0.008 (n=400) | -0.112±0.011 (n=202) | -0.094±0.011 (n=198) | continuing |
| continuous / distance / 10/5 | -0.106±0.008 (n=400) | -0.114±0.011 (n=202) | -0.097±0.012 (n=198) | continuing |
| continuous / cosine / perfect | -0.129±0.009 (n=400) | -0.142±0.012 (n=207) | -0.116±0.013 (n=193) | continuing |
| continuous / cosine / 10/5 | -0.129±0.009 (n=400) | -0.137±0.012 (n=207) | -0.119±0.012 (n=193) | continuing |

### best_fit_retained

| 條件 | all | continuing | switching | 最差層 |
|---|---|---|---|---|
| archetype / distance / perfect | +0.015±0.009 (n=400) | +0.010±0.015 (n=209) | +0.021±0.010 (n=191) | continuing |
| archetype / distance / 10/5 | +0.007±0.009 (n=400) | -0.005±0.014 (n=209) | +0.021±0.010 (n=191) | continuing |
| archetype / cosine / perfect | +0.025±0.011 (n=400) | +0.029±0.018 (n=210) | +0.021±0.010 (n=190) | switching |
| archetype / cosine / 10/5 | +0.025±0.009 (n=400) | +0.033±0.014 (n=210) | +0.016±0.009 (n=190) | switching |
| continuous / distance / perfect | +0.052±0.014 (n=400) | +0.035±0.016 (n=202) | +0.071±0.022 (n=198) | continuing |
| continuous / distance / 10/5 | +0.052±0.014 (n=400) | +0.050±0.017 (n=202) | +0.056±0.023 (n=198) | continuing |
| continuous / cosine / perfect | +0.060±0.014 (n=400) | +0.072±0.023 (n=207) | +0.047±0.017 (n=193) | switching |
| continuous / cosine / 10/5 | +0.060±0.013 (n=400) | +0.077±0.021 (n=207) | +0.041±0.016 (n=193) | switching |

## 各方法的水準（只報告）

### novel_relevant_rate

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.578±0.011 (n=400) | 0.476±0.014 (n=209) | 0.690±0.014 (n=191) |
| archetype / distance / perfect | fit-only | 0.481±0.012 (n=400) | 0.376±0.014 (n=209) | 0.597±0.015 (n=191) |
| archetype / distance / 10/5 | sections | 0.548±0.011 (n=400) | 0.451±0.014 (n=209) | 0.654±0.013 (n=191) |
| archetype / distance / 10/5 | fit-only | 0.481±0.012 (n=400) | 0.376±0.014 (n=209) | 0.597±0.015 (n=191) |
| archetype / cosine / perfect | sections | 0.585±0.012 (n=400) | 0.474±0.014 (n=210) | 0.708±0.014 (n=190) |
| archetype / cosine / perfect | fit-only | 0.491±0.012 (n=400) | 0.377±0.014 (n=210) | 0.618±0.015 (n=190) |
| archetype / cosine / 10/5 | sections | 0.553±0.011 (n=400) | 0.450±0.014 (n=210) | 0.667±0.014 (n=190) |
| archetype / cosine / 10/5 | fit-only | 0.491±0.012 (n=400) | 0.377±0.014 (n=210) | 0.618±0.015 (n=190) |
| continuous / distance / perfect | sections | 0.440±0.009 (n=400) | 0.435±0.012 (n=202) | 0.444±0.014 (n=198) |
| continuous / distance / perfect | fit-only | 0.342±0.009 (n=400) | 0.336±0.011 (n=202) | 0.348±0.013 (n=198) |
| continuous / distance / 10/5 | sections | 0.416±0.009 (n=400) | 0.409±0.012 (n=202) | 0.423±0.013 (n=198) |
| continuous / distance / 10/5 | fit-only | 0.342±0.009 (n=400) | 0.336±0.011 (n=202) | 0.348±0.013 (n=198) |
| continuous / cosine / perfect | sections | 0.503±0.009 (n=400) | 0.478±0.013 (n=207) | 0.530±0.013 (n=193) |
| continuous / cosine / perfect | fit-only | 0.400±0.009 (n=400) | 0.382±0.013 (n=207) | 0.419±0.014 (n=193) |
| continuous / cosine / 10/5 | sections | 0.474±0.009 (n=400) | 0.453±0.013 (n=207) | 0.497±0.013 (n=193) |
| continuous / cosine / 10/5 | fit-only | 0.400±0.009 (n=400) | 0.382±0.013 (n=207) | 0.419±0.014 (n=193) |

### relevant_rate

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.578±0.011 (n=400) | 0.476±0.014 (n=209) | 0.690±0.014 (n=191) |
| archetype / distance / perfect | fit-only | 0.813±0.010 (n=400) | 0.789±0.014 (n=209) | 0.839±0.014 (n=191) |
| archetype / distance / 10/5 | sections | 0.590±0.010 (n=400) | 0.504±0.014 (n=209) | 0.684±0.013 (n=191) |
| archetype / distance / 10/5 | fit-only | 0.813±0.010 (n=400) | 0.789±0.014 (n=209) | 0.839±0.014 (n=191) |
| archetype / cosine / perfect | sections | 0.585±0.012 (n=400) | 0.474±0.014 (n=210) | 0.708±0.014 (n=190) |
| archetype / cosine / perfect | fit-only | 0.812±0.010 (n=400) | 0.785±0.014 (n=210) | 0.841±0.013 (n=190) |
| archetype / cosine / 10/5 | sections | 0.593±0.011 (n=400) | 0.506±0.013 (n=210) | 0.690±0.014 (n=190) |
| archetype / cosine / 10/5 | fit-only | 0.812±0.010 (n=400) | 0.785±0.014 (n=210) | 0.841±0.013 (n=190) |
| continuous / distance / perfect | sections | 0.440±0.009 (n=400) | 0.435±0.012 (n=202) | 0.444±0.014 (n=198) |
| continuous / distance / perfect | fit-only | 0.543±0.011 (n=400) | 0.547±0.014 (n=202) | 0.538±0.016 (n=198) |
| continuous / distance / 10/5 | sections | 0.437±0.009 (n=400) | 0.433±0.012 (n=202) | 0.441±0.013 (n=198) |
| continuous / distance / 10/5 | fit-only | 0.543±0.011 (n=400) | 0.547±0.014 (n=202) | 0.538±0.016 (n=198) |
| continuous / cosine / perfect | sections | 0.503±0.009 (n=400) | 0.478±0.013 (n=207) | 0.530±0.013 (n=193) |
| continuous / cosine / perfect | fit-only | 0.632±0.010 (n=400) | 0.620±0.014 (n=207) | 0.645±0.015 (n=193) |
| continuous / cosine / 10/5 | sections | 0.503±0.009 (n=400) | 0.483±0.013 (n=207) | 0.526±0.013 (n=193) |
| continuous / cosine / 10/5 | fit-only | 0.632±0.010 (n=400) | 0.620±0.014 (n=207) | 0.645±0.015 (n=193) |

### best_fit_retained

| 條件 | 方法 | all | continuing | switching |
|---|---|---|---|---|
| archetype / distance / perfect | sections | 0.943±0.012 (n=400) | 0.923±0.018 (n=209) | 0.963±0.014 (n=191) |
| archetype / distance / perfect | fit-only | 0.927±0.013 (n=400) | 0.914±0.019 (n=209) | 0.942±0.017 (n=191) |
| archetype / distance / 10/5 | sections | 0.935±0.012 (n=400) | 0.909±0.020 (n=209) | 0.963±0.014 (n=191) |
| archetype / distance / 10/5 | fit-only | 0.927±0.013 (n=400) | 0.914±0.019 (n=209) | 0.942±0.017 (n=191) |
| archetype / cosine / perfect | sections | 0.960±0.010 (n=400) | 0.938±0.017 (n=210) | 0.984±0.009 (n=190) |
| archetype / cosine / perfect | fit-only | 0.935±0.012 (n=400) | 0.910±0.020 (n=210) | 0.963±0.014 (n=190) |
| archetype / cosine / 10/5 | sections | 0.960±0.010 (n=400) | 0.943±0.016 (n=210) | 0.979±0.010 (n=190) |
| archetype / cosine / 10/5 | fit-only | 0.935±0.012 (n=400) | 0.910±0.020 (n=210) | 0.963±0.014 (n=190) |
| continuous / distance / perfect | sections | 0.825±0.019 (n=400) | 0.842±0.026 (n=202) | 0.808±0.028 (n=198) |
| continuous / distance / perfect | fit-only | 0.772±0.021 (n=400) | 0.807±0.028 (n=202) | 0.737±0.031 (n=198) |
| continuous / distance / 10/5 | sections | 0.825±0.019 (n=400) | 0.856±0.025 (n=202) | 0.793±0.029 (n=198) |
| continuous / distance / 10/5 | fit-only | 0.772±0.021 (n=400) | 0.807±0.028 (n=202) | 0.737±0.031 (n=198) |
| continuous / cosine / perfect | sections | 0.912±0.014 (n=400) | 0.899±0.021 (n=207) | 0.927±0.019 (n=193) |
| continuous / cosine / perfect | fit-only | 0.853±0.018 (n=400) | 0.826±0.026 (n=207) | 0.881±0.023 (n=193) |
| continuous / cosine / 10/5 | sections | 0.912±0.014 (n=400) | 0.903±0.021 (n=207) | 0.922±0.019 (n=193) |
| continuous / cosine / 10/5 | fit-only | 0.853±0.018 (n=400) | 0.826±0.026 (n=207) | 0.881±0.023 (n=193) |

