"""Does a programme-type level help the admission model? (W-74, B-C3-6; THEORY
"學程類型層"; A-C3-7.)

DESIGN v2 (2026-10-28, written after the first registered run and committed before
this one): in v1 the type (i % 2) was completely confounded with the family (i % 4),
so a type effect could not be separated from family means and the run could not
answer the question (results/synthetic-l3-admission-types-confounded.md). v2 assigns
the type by (i // 4) % 2, so every family has both types, holds out p00-p07 (four of
each type, one per family and type) and uses a fresh seed (2033). The decision rule
below is unchanged.

Worlds as in run_l3_admission_bias.py (4 CIP families, 30 programmes, 3 features,
self-reports with q1 / q0 = 0.8 / 0.2, official rates as anchors, C3.2 v0.2 default),
plus a programme type: programmes alternate between "standard" and "interview".
Generators:
  typed     interview programmes have a logit shift of -1.0 (in every family)
  untyped   no type effect (the type label carries no information)
Data: self-reports only, or plus complete advisor records for 10 programmes (15 each).
Eight programmes (v2: 4 per type, one per family x type) are held out entirely and
predicted as unseen programmes from their family (and type). 30 replicate worlds per setting.
Compared: T (types given) vs N (no type level), both with official rates for the
observed programmes only (the held-out programmes have none, as for a new programme).
Metrics per replicate: RMSE of predicted P vs true P over 200 fresh applicants per
programme, separately for held-out and observed programmes; 90% coverage.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  Using types is RECOMMENDED when type information exists if
    (a) typed generator: T's held-out RMSE is below N's by more than 2 SE in both data
        settings; and
    (b) untyped generator: T's RMSE (held-out and observed) is not above N's by more
        than 2 SE in any data setting (no harm when types carry no signal).
  Otherwise types stay an option without a recommendation.
  Report only: observed-programme RMSE under the typed generator, coverage.
  Disclosure: smoke-tested with 2 replicates (exit status only).

    python scripts/run_l3_admission_types.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

from aisc.feasibility import AdmissionRecord, fit_admission_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_l3_admission_bias import BETA, MU_F, N_PROG, N_TEST, sig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENS = ("typed", "untyped")
DATA = ("self_only", "with_advisor")
SHIFT = -1.0
HELD = tuple(f"p{i:02d}" for i in range(8))  # v2: one per family x type
N_REP = 30
SEED = 2033  # v2 (v1 used 2032)


def world(gen, data, rng):
    fam = {f"p{i:02d}": f"f{i % 4}" for i in range(N_PROG)}
    ptype = {f"p{i:02d}": ("interview" if (i // 4) % 2 else "standard") for i in range(N_PROG)}
    a = {}
    for i in range(N_PROG):
        pid = f"p{i:02d}"
        a[pid] = MU_F[i % 4] + rng.normal(0, 0.6)
        if gen == "typed" and ptype[pid] == "interview":
            a[pid] += SHIFT
    recs, rates, tests = [], {}, {}
    for i in range(N_PROG):
        pid = f"p{i:02d}"
        pool = rng.normal(0, 1, (5000, 3))
        y = rng.random(5000) < sig(a[pid] + pool @ BETA)
        Xt = rng.normal(0, 1, (N_TEST, 3))
        tests[pid] = (Xt, sig(a[pid] + Xt @ BETA))
        if pid in HELD:
            continue
        rates[pid] = float(y.mean())
        idx = np.flatnonzero(rng.random(5000) < np.where(y, 0.8, 0.2))[: int(rng.integers(10, 81))]
        recs += [AdmissionRecord(pid, fam[pid], tuple(pool[j]), int(y[j]), "self_report")
                 for j in idx]  # fmt: skip
        if data == "with_advisor" and 8 <= i < 18:
            Xa = rng.normal(0, 1, (15, 3))
            ya = rng.random(15) < sig(a[pid] + Xa @ BETA)
            recs += [AdmissionRecord(pid, fam[pid], tuple(Xa[j]), int(ya[j]), "complete")
                     for j in range(15)]  # fmt: skip
    return recs, rates, tests, fam, ptype


def evaluate(model, tests, fam, ptype, held):
    err, cov = [], []
    for pid in held:
        Xt, pt = tests[pid]
        for x, p in zip(Xt, pt, strict=True):
            est = model.predict(pid, tuple(x), family=fam[pid], ptype=ptype[pid])
            err.append(est.mean - p)
            cov.append(est.low <= p <= est.high)
    err = np.asarray(err)
    return math.sqrt(float(np.mean(err**2))), float(np.mean(cov))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    out = ROOT / "results" / "synthetic-l3-admission-types.md"
    ap.add_argument("--out", type=Path, default=out)
    args = ap.parse_args()
    res = {}
    for gen in GENS:
        for data in DATA:
            rows = {k: [] for k in ("T_held", "N_held", "T_obs", "N_obs")}
            for r in range(args.reps):
                rng = np.random.default_rng([SEED, GENS.index(gen), DATA.index(data), r])
                recs, rates, tests, fam, ptype = world(gen, data, rng)
                observed = [p for p in tests if p not in HELD]
                mt = fit_admission_model(recs, families=fam, official_rates=rates,
                                         types={p: ptype[p] for p in observed})  # fmt: skip
                mn = fit_admission_model(recs, families=fam, official_rates=rates)
                rows["T_held"].append(evaluate(mt, tests, fam, ptype, HELD))
                rows["N_held"].append(evaluate(mn, tests, fam, ptype, HELD))
                rows["T_obs"].append(evaluate(mt, tests, fam, ptype, observed))
                rows["N_obs"].append(evaluate(mn, tests, fam, ptype, observed))
            res[(gen, data)] = rows
            print(gen, data, flush=True)

    def paired(a, b):
        d = np.asarray([x[0] for x in a]) - np.asarray([x[0] for x in b])
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    a_ok = all(paired(res[("typed", d)]["T_held"], res[("typed", d)]["N_held"])[0]
               < -2 * paired(res[("typed", d)]["T_held"], res[("typed", d)]["N_held"])[1]
               for d in DATA)  # fmt: skip
    b_ok = all(not (lambda x: x[0] > 2 * x[1])(paired(res[("untyped", d)][f"T_{w}"],
                                                       res[("untyped", d)][f"N_{w}"]))
               for d in DATA for w in ("held", "obs"))  # fmt: skip
    rec = a_ok and b_ok

    def ms(v):
        v = np.asarray([x[0] for x in v])
        return f"{v.mean():.3f}±{v.std(ddof=1) / math.sqrt(len(v)):.3f}"

    L = [
        "# 錄取模型的學程類型層（W-74，B-C3-6）",
        "",
        f"- 產生日期：{date.today().isoformat()} · "
        "指令：`python scripts/run_l3_admission_types.py`",
        f"- {N_PROG} 個學程（{len(HELD)} 個完全未觀測、當作新學程預測）；"
        f"每個設定 {args.reps} 個世界；設計 v2（類型與族群不重疊）",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        f"## 判定：(a) 有類型效果時新學程顯著較準 {'✅' if a_ok else '❌'}；"
        f"(b) 沒有類型效果時沒有變差 {'✅' if b_ok else '❌'} → "
        f"{'**有類型資訊時建議使用**' if rec else '類型層維持可選用、不建議'}",
        "",
        "| 生成 | 資料 | 新學程 RMSE（有類型） | （無類型） | 差 | 已觀測學程 RMSE（有類型） "
        "| （無類型） | 差 | 新學程 90% 涵蓋（有／無） |",
        "|---" * 9 + "|",
    ]
    for (gen, data), r in res.items():
        dh, do = paired(r["T_held"], r["N_held"]), paired(r["T_obs"], r["N_obs"])
        cov_t = np.mean([x[1] for x in r["T_held"]])
        cov_n = np.mean([x[1] for x in r["N_held"]])
        L.append(f"| {gen} | {data} | {ms(r['T_held'])} | {ms(r['N_held'])} | "
                 f"{dh[0]:+.4f}±{dh[1]:.4f} | {ms(r['T_obs'])} | {ms(r['N_obs'])} | "
                 f"{do[0]:+.4f}±{do[1]:.4f} | {cov_t:.3f} / {cov_n:.3f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"a={a_ok} b={b_ok} recommend={rec} wrote {args.out}")


if __name__ == "__main__":
    main()
