"""Year effects in the admission model (B-C3-3; THEORY "年度效果"; A-C3-8).

Worlds as in run_l3_admission_bias.py (4 CIP families, 30 programmes, 3 features,
self-reports with q1 / q0 = 0.8 / 0.2, C3.2 v0.2 default with the LATEST year's
official rates as anchors). Records come from three application years (2023, 2024,
2025); per programme and year the number of self-reports is uniform on {5, ..., 30}.
Generators:
  drift   2023 and 2024 were easier: logit +0.8 relative to 2025 (e.g. an applicant
          surge in 2025)
  stable  no year differences
Data: self-reports only, or plus complete advisor records for 10 programmes (15 per
programme, from 2025). 30 replicate worlds per setting.
Compared: Y (records carry their year) vs P (years ignored, all pooled). Metric: RMSE
of predicted P(admit) for 200 fresh 2025 applicants per programme; 90% coverage.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  Labelling years is RECOMMENDED if
    (a) drift: Y's RMSE is below P's by more than 2 SE in both data settings; and
    (b) stable: Y's RMSE is not above P's by more than 2 SE in either data setting.
  Report only: coverage, the estimated year shifts.
  Disclosure: smoke-tested with 2 replicates (exit status only).

    python scripts/run_l3_admission_years.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

from aisc.feasibility import AdmissionRecord, fit_admission_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_l3_admission_bias import BETA, MU_F, N_PROG, N_TEST, evaluate, sig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENS = ("drift", "stable")
DATA = ("self_only", "with_advisor")
YEARS = (2023, 2024, 2025)
DRIFT = 0.8
N_REP = 30
SEED = 2034


def world(gen, data, rng):
    fam = {f"p{i:02d}": f"f{i % 4}" for i in range(N_PROG)}
    recs, rates, tests = [], {}, {}
    for i in range(N_PROG):
        pid = f"p{i:02d}"
        a = MU_F[i % 4] + rng.normal(0, 0.6)
        for year in YEARS:
            shift = DRIFT if (gen == "drift" and year < 2025) else 0.0
            pool = rng.normal(0, 1, (3000, 3))
            y = rng.random(3000) < sig(a + shift + pool @ BETA)
            if year == 2025:
                rates[pid] = float(y.mean())
            n = int(rng.integers(5, 31))
            idx = np.flatnonzero(rng.random(3000) < np.where(y, 0.8, 0.2))[:n]
            recs += [AdmissionRecord(pid, fam[pid], tuple(pool[j]), int(y[j]), "self_report",
                                     year) for j in idx]  # fmt: skip
        if data == "with_advisor" and i < 10:
            Xa = rng.normal(0, 1, (15, 3))
            ya = rng.random(15) < sig(a + Xa @ BETA)
            recs += [AdmissionRecord(pid, fam[pid], tuple(Xa[j]), int(ya[j]), "complete", 2025)
                     for j in range(15)]  # fmt: skip
        Xt = rng.normal(0, 1, (N_TEST, 3))
        tests[pid] = (Xt, sig(a + Xt @ BETA))
    return recs, rates, tests, fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    out = ROOT / "results" / "synthetic-l3-admission-years.md"
    ap.add_argument("--out", type=Path, default=out)
    args = ap.parse_args()
    res = {}
    for gen in GENS:
        for data in DATA:
            rows = {"Y": [], "P": [], "shift": []}
            for r in range(args.reps):
                rng = np.random.default_rng([SEED, GENS.index(gen), DATA.index(data), r])
                recs, rates, tests, fam = world(gen, data, rng)
                my = fit_admission_model(recs, families=fam, official_rates=rates)
                pooled = [dataclasses.replace(x, year=None) for x in recs]
                mp = fit_admission_model(pooled, families=fam, official_rates=rates)
                rows["Y"].append(evaluate(my, tests))
                rows["P"].append(evaluate(mp, tests))
                rows["shift"].append(np.asarray(my.theta[-1 - len(my.years) : -1]).mean())
            res[(gen, data)] = rows
            print(gen, data, flush=True)

    def paired(r):
        d = np.asarray([x[0] for x in r["Y"]]) - np.asarray([x[0] for x in r["P"]])
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    a_ok = all(paired(res[("drift", d)])[0] < -2 * paired(res[("drift", d)])[1] for d in DATA)
    b_ok = all(not paired(res[("stable", d)])[0] > 2 * paired(res[("stable", d)])[1]
               for d in DATA)  # fmt: skip
    rec = a_ok and b_ok

    def ms(v):
        v = np.asarray([x[0] for x in v])
        return f"{v.mean():.3f}±{v.std(ddof=1) / math.sqrt(len(v)):.3f}"

    L = [
        "# 錄取模型的年度效果（B-C3-3）",
        "",
        f"- 產生日期：{date.today().isoformat()} · "
        "指令：`python scripts/run_l3_admission_years.py`",
        f"- {N_PROG} 個學程、3 個申請年度；每個設定 {args.reps} 個世界；預測最新年度（2025）",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        f"## 判定：(a) 有年度波動時較準 {'✅' if a_ok else '❌'}；(b) 沒有波動時不變差 "
        f"{'✅' if b_ok else '❌'} → {'**建議標記年度**' if rec else '不建議'}",
        "",
        "| 生成 | 資料 | RMSE 標記年度 | RMSE 不分年度 | 差 | 90% 涵蓋（標記／不分） "
        "| 舊年度平移估計（平均） |",
        "|---|---|---|---|---|---|---|",
    ]
    for (gen, data), r in res.items():
        d = paired(r)
        L.append(f"| {gen} | {data} | {ms(r['Y'])} | {ms(r['P'])} | {d[0]:+.4f}±{d[1]:.4f} | "
                 f"{np.mean([x[2] for x in r['Y']]):.3f} / {np.mean([x[2] for x in r['P']]):.3f} | "
                 f"{np.mean(r['shift']):+.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"a={a_ok} b={b_ok} recommend={rec} wrote {args.out}")


if __name__ == "__main__":
    main()
