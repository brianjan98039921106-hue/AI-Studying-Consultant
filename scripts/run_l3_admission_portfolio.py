"""C3 -> C4: does the self-report correction change which schools a student applies to?
(W-44 x W-29/W-2A; B-C3-1, B-C3-8; uses the V-C3-1 worlds of run_l3_admission_bias.py.)

For each V-C3-1 setting (2 generators x 2 reporting strengths x 2 data sources) and
30 replicate worlds, fit C3.2 v0 uncorrected (U) and corrected (C) exactly as in
run_l3_admission_bias.py. Then 20 test applicants per world, each considering 10 of
the 30 programmes (random), with utilities that favour selective programmes:
u_j = 0.4 + 0.6 * (rank of -true intercept among the 10) / 9 + N(0, 0.05), clipped to
[0.05, 1]; budget 4 applications, equal cost. Portfolios (independent admissions):
  U   C4.1 on U's posterior medians
  C   C4.1 on C's posterior medians
  CR  C4.3 minimax regret on C's 90% intervals (point = median)
Metric: regret under the TRUE probabilities, (v* - v(S)) / v*, per applicant;
averaged per replicate.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  H-AP: in every setting of both generators, C's mean regret is below U's by more
        than 2 SE (paired over the 30 replicates).
  Report only: CR vs C (does acting on C's intervals help further?), overconfidence:
        U's claimed expected utility minus the true one.
  This is a report on decision impact; it changes no default (C3 is not yet used by
  the pipeline). Disclosure: smoke-tested with 1 replicate (exit status only).

    python scripts/run_l3_admission_portfolio.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

import aisc.portfolio as pf
from aisc.feasibility import AdmissionRecord, fit_admission_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_l3_admission_bias import BETA, DATA, GENS, N_PROG, QS, sig, world  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_REP = 30
N_APPL = 20
M, H = 10, 4
SEED = 2029


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "synthetic-l3-admission-portfolio.md")  # fmt: skip
    args = ap.parse_args()
    res = {}
    for gen in GENS:
        for q in QS:
            for data in DATA:
                rows = {"U": [], "C": [], "CR": [], "over_U": [], "over_C": []}
                for r in range(args.reps):
                    rng = np.random.default_rng([SEED, GENS.index(gen), QS.index(q),
                                                 DATA.index(data), r])  # fmt: skip
                    recs, rates, tests, fams = world(gen, q, data, rng)
                    u_recs = (
                        [x for x in recs if x.source == "self_report"]
                        if data == "self_only"
                        else recs
                    )
                    mu = fit_admission_model(
                        [AdmissionRecord(x.program_id, x.family, x.features, x.admitted,
                                         "self_report") for x in u_recs],
                        families=fams,
                        tau_average=False,  # pinned to v0 (default changed later, F136)
                        anchor="mean",
                    )  # fmt: skip
                    mc = fit_admission_model(
                        recs, families=fams, official_rates=rates, tau_average=False, anchor="mean"
                    )  # v0 pinned
                    # true intercepts are recoverable from the test sets: logit p - x.beta
                    a_true = {pid: float(np.mean(np.log(pt / (1 - pt)) - Xt @ BETA))
                              for pid, (Xt, pt) in tests.items()}  # fmt: skip
                    shift = np.array([0.3, 0.0, 0.0]) if gen == "stress" else np.zeros(3)
                    reg = {"U": [], "C": [], "CR": [], "over_U": [], "over_C": []}
                    for _ in range(N_APPL):
                        x = rng.normal(0, 1, 3) + shift
                        pids = list(rng.choice(sorted(tests), M, replace=False))
                        order = np.argsort(np.argsort([-a_true[p] for p in pids]))
                        u = np.clip(0.4 + 0.6 * (M - 1 - order) / (M - 1)
                                    + rng.normal(0, 0.05, M), 0.05, 1.0).tolist()  # fmt: skip
                        p_true = [float(sig(a_true[p] + x @ BETA)) for p in pids]
                        eu = [mu.predict(p, tuple(x)) for p in pids]
                        ec = [mc.predict(p, tuple(x)) for p in pids]
                        cost = [1] * M
                        _, v_star = pf.dp_varied_cost(p_true, u, cost, H)
                        s_u, claim_u = pf.dp_varied_cost([e.mean for e in eu], u, cost, H)
                        s_c, claim_c = pf.dp_varied_cost([e.mean for e in ec], u, cost, H)
                        s_cr = pf.minimax_regret_portfolio(
                            [e.low for e in ec], [e.high for e in ec], [e.mean for e in ec],
                            u, cost, H, n_scenarios=100, rng=rng,
                        )  # fmt: skip
                        for k, s in (("U", s_u), ("C", s_c), ("CR", s_cr)):
                            reg[k].append(
                                (v_star - pf.portfolio_value(p_true, u, list(s))) / v_star
                            )
                        reg["over_U"].append(claim_u - pf.portfolio_value(p_true, u, list(s_u)))
                        reg["over_C"].append(claim_c - pf.portfolio_value(p_true, u, list(s_c)))
                    for k in rows:
                        rows[k].append(float(np.mean(reg[k])))
                res[(gen, q, data)] = rows
                print(gen, q, data, flush=True)

    def paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    holds = {g: all(paired(r["C"], r["U"])[0] < -2 * paired(r["C"], r["U"])[1]
                    for (gg, _, _), r in res.items() if gg == g) for g in GENS}  # fmt: skip

    def ms(v, scale=100):
        v = np.asarray(v) * scale
        return f"{v.mean():.2f}±{v.std(ddof=1) / math.sqrt(len(v)):.2f}"

    L = [
        "# C3 → C4：自陳偏誤校正對申請組合的影響",
        "",
        f"- 產生日期：{date.today().isoformat()} · "
        "指令：`python scripts/run_l3_admission_portfolio.py`",
        f"- V-C3-1 的模擬世界（{N_PROG} 個學程）；每個設定 {args.reps} 個世界 × {N_APPL} 位申請者，"
        f"每人從 {M} 個學程中選 {H} 個申請；效用偏好較難錄取的學程",
        "- 預先登記見腳本檔頭；全部是合成資料",
        "",
        f"## 判定（H-AP）：standard {'✅' if holds['standard'] else '❌'} · stress "
        f"{'✅' if holds['stress'] else '❌'}",
        "",
        "遺憾 = 比真正最佳組合少多少（%）；"
        "宣稱高估 = 模型宣稱的期望效用 − 真實期望效用（效用單位 × 100）。",
        "",
        "| 生成 | q₁ / q₀ | 資料 | 遺憾 U | 遺憾 C | C − U | 遺憾 CR | CR − C "
        "| 宣稱高估 U | 宣稱高估 C |",
        "|---" * 10 + "|",
    ]
    for (gen, q, data), r in res.items():
        d, dcr = paired(r["C"], r["U"]), paired(r["CR"], r["C"])
        L.append(f"| {gen} | {q[0]} / {q[1]} | {data} | {ms(r['U'])} | {ms(r['C'])} | "
                 f"{100 * d[0]:+.2f}±{100 * d[1]:.2f} | {ms(r['CR'])} | "
                 f"{100 * dcr[0]:+.2f}±{100 * dcr[1]:.2f} | {ms(r['over_U'])} | "
                 f"{ms(r['over_C'])} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"H-AP standard={holds['standard']} stress={holds['stress']} wrote {args.out}")


if __name__ == "__main__":
    main()
