"""V-C3-1: parameter recovery under a known self-report mechanism (W-44, B-C3-1;
C3.2 v0, A-C3-3 ... A-C3-5).

Synthetic admission worlds: 4 CIP families with mean logit intercepts
(-2.2, -1.2, -0.4, 0.4), 30 programmes (a_p = mu_f + N(0, 0.6^2)), 3 standardised
applicant features with beta = (0.8, -0.4, 0.3). Applicant pools are N(0, I). Each
programme's official rate is its pool's true admission rate. Self-reports come from
applicants with probability q1 if admitted and q0 if rejected; per programme the
number of self-reports is uniform on {10, ..., 80}.
Generators (both must agree before a claim is made):
  standard  reporting depends only on the outcome (A-C3-3 and A-C3-5 hold)
  stress    reporting also rises with feature 1 (q times 2 * sigmoid(0.7 x1), capped
            at 1) and the official pool's feature 1 has mean +0.3 while the model
            assumes 0 (A-C3-3 and A-C3-5 violated)
Settings: (q1, q0) in {(0.6, 0.3), (0.8, 0.2)} x data in {self-report only;
self-report + complete advisor records for 10 programmes, 15 each} = 4 per generator.
30 replicate worlds per setting.
Methods (C3.2 v0, tau chosen by marginal likelihood):
  U  uncorrected: no official rates (self-report shift not identified -> fixed 0)
  C  corrected: official rates as anchors (kappa 0.3) + self-report shift
Evaluation per world: 200 fresh pool applicants per programme; error = predicted
P(admit) (posterior median) - true P. Per replicate: RMSE and mean error over all
test applicants, coverage of the 90% interval.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  V-C3-1 holds for a generator if, in all 4 settings, C's RMSE is below U's by more
  than 2 SE (paired over the 30 replicates). The B-C3-1 correction is reported as
  confirmed only if V-C3-1 holds under both generators; holding under standard only
  is reported as "holds under its assumptions".
  Report only: mean error (bias), 90% interval coverage (V-C3-3 is an L5 target,
  85-95%), the estimated shift vs log(q1 / q0).
  Disclosure: before this run the model was only exercised by its unit tests
  (tests/test_c3_feasibility.py) and a smoke run of this script with 1 replicate
  (exit status only).

    python scripts/run_l3_admission_bias.py
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np

from aisc.feasibility import AdmissionRecord, fit_admission_model

ROOT = Path(__file__).resolve().parents[1]
MU_F = (-2.2, -1.2, -0.4, 0.4)
BETA = np.array([0.8, -0.4, 0.3])
N_PROG = 30
N_TEST = 200
QS = ((0.6, 0.3), (0.8, 0.2))
DATA = ("self_only", "with_advisor")
GENS = ("standard", "stress")
N_REP = 30
SEED = 2028


def sig(x):
    return 1 / (1 + np.exp(-x))


def world(gen, q, data, rng):
    fam = [f"f{i % 4}" for i in range(N_PROG)]
    a = np.array([MU_F[i % 4] + rng.normal(0, 0.6) for i in range(N_PROG)])
    shift = np.array([0.3, 0.0, 0.0]) if gen == "stress" else np.zeros(3)
    recs, rates, tests = [], {}, {}
    for i in range(N_PROG):
        pid = f"p{i:02d}"
        pool = rng.normal(0, 1, (5000, 3)) + shift
        p = sig(a[i] + pool @ BETA)
        y = rng.random(5000) < p
        rates[pid] = float(y.mean())
        rep = np.where(y, q[0], q[1])
        if gen == "stress":
            rep = np.minimum(1.0, rep * 2 * sig(0.7 * pool[:, 0]))
        idx = np.flatnonzero(rng.random(5000) < rep)[: int(rng.integers(10, 81))]
        recs += [AdmissionRecord(pid, fam[i], tuple(pool[j]), int(y[j]), "self_report")
                 for j in idx]  # fmt: skip
        Xt = rng.normal(0, 1, (N_TEST, 3)) + shift
        tests[pid] = (Xt, sig(a[i] + Xt @ BETA))
        if data == "with_advisor" and i < 10:
            Xa = rng.normal(0, 1, (15, 3)) + shift
            ya = rng.random(15) < sig(a[i] + Xa @ BETA)
            recs += [AdmissionRecord(pid, fam[i], tuple(Xa[j]), int(ya[j]), "complete")
                     for j in range(15)]  # fmt: skip
    return recs, rates, tests, {f"p{i:02d}": fam[i] for i in range(N_PROG)}


def evaluate(model, tests):
    err, cov = [], []
    for pid, (Xt, pt) in tests.items():
        for x, p in zip(Xt, pt, strict=True):
            est = model.predict(pid, tuple(x))
            err.append(est.mean - p)
            cov.append(est.low <= p <= est.high)
    err = np.array(err)
    return math.sqrt(float(np.mean(err**2))), float(err.mean()), float(np.mean(cov))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-admission-bias.md")
    args = ap.parse_args()
    res = {}
    for gen in GENS:
        for q in QS:
            for data in DATA:
                rows = {"U": [], "C": [], "shift": []}
                for r in range(args.reps):
                    rng = np.random.default_rng([SEED, GENS.index(gen), QS.index(q),
                                                 DATA.index(data), r])  # fmt: skip
                    recs, rates, tests, fams = world(gen, q, data, rng)
                    self_only = [x for x in recs if x.source == "self_report"]
                    u_recs = self_only if data == "self_only" else recs
                    mu = fit_admission_model(
                        [AdmissionRecord(x.program_id, x.family, x.features, x.admitted,
                                         "self_report") for x in u_recs],
                        families=fams,
                        tau_average=False,  # pinned to v0 (default changed later, F136)
                    )  # fmt: skip
                    mc = fit_admission_model(
                        recs, families=fams, official_rates=rates, tau_average=False
                    )  # v0 pinned
                    rows["U"].append(evaluate(mu, tests))
                    rows["C"].append(evaluate(mc, tests))
                    rows["shift"].append(mc.shift)
                res[(gen, q, data)] = rows
                print(gen, q, data, flush=True)

    def paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    holds = {}
    for gen in GENS:
        ok = True
        for (g, _, _), rows in res.items():
            if g != gen:
                continue
            d, se = paired([x[0] for x in rows["C"]], [x[0] for x in rows["U"]])
            if not d < -2 * se:
                ok = False
        holds[gen] = ok
    verdict = ("confirmed (both generators)" if all(holds.values())
               else "holds under its assumptions (standard only)" if holds["standard"]
               else "not supported")  # fmt: skip

    def ms(v):
        v = np.asarray(v)
        return f"{v.mean():.3f}±{v.std(ddof=1) / math.sqrt(len(v)):.3f}"

    L = [
        "# V-C3-1：已知自陳機制下的錄取模型參數還原（C3.2 v0，B-C3-1）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_admission_bias.py`",
        f"- {N_PROG} 個學程、4 個族群、3 個特徵；每個設定 {args.reps} 個模擬世界，"
        f"每學程 {N_TEST} 位新申請者評估",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        f"## 判定：standard {'✅' if holds['standard'] else '❌'} · stress "
        f"{'✅' if holds['stress'] else '❌'} → **{verdict}**",
        "",
        "U = 未校正（不用官方錄取率）；C = 校正（官方錄取率錨點 ＋ 自陳平移）。"
        "誤差 = 預測 − 真實機率。",
        "",
        "| 生成 | q₁ / q₀ | 資料 | RMSE U | RMSE C | C − U | 平均誤差 U | 平均誤差 C "
        "| 90% 涵蓋 U | 90% 涵蓋 C | 平移估計 | log(q₁/q₀) |",
        "|---" * 12 + "|",
    ]
    for (gen, q, data), rows in res.items():
        d = paired([x[0] for x in rows["C"]], [x[0] for x in rows["U"]])
        L.append(
            f"| {gen} | {q[0]} / {q[1]} | {data} | {ms([x[0] for x in rows['U']])} | "
            f"{ms([x[0] for x in rows['C']])} | {d[0]:+.3f}±{d[1]:.3f} | "
            f"{ms([x[1] for x in rows['U']])} | {ms([x[1] for x in rows['C']])} | "
            f"{np.mean([x[2] for x in rows['U']]):.2f} | "
            f"{np.mean([x[2] for x in rows['C']]):.2f} | "
            f"{ms(rows['shift'])} | {math.log(q[0] / q[1]):.3f} |"
        )
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"standard={holds['standard']} stress={holds['stress']} -> {verdict}; wrote {args.out}")


if __name__ == "__main__":
    main()
