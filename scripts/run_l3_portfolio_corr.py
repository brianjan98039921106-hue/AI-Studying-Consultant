"""V-C4-2: how wrong is the independence model when admissions are correlated?
(W-29, B-C4-1; C4.2 one-factor Gaussian copula, A-C4-5.)

No C3 yet, so scenarios are synthetic (two generator families, as for every L3 claim):
  tiered   40% reach p~U[0.05,0.25] u~0.90, 40% target p~U[0.30,0.60] u~0.70,
           20% safety p~U[0.70,0.95] u~0.45; u = base + N(0, 0.08), clipped [0.05, 1]
  uniform  p~U[0.05,0.95], u~U[0.2,1] independent of p
m in {10, 20} schools, budget H in {3, 6} applications (equal cost), 100 scenarios per
(generator, m, H), the same scenarios for every rho. True rho in {0, 0.2, 0.4, 0.6, 0.8}.
  S_ind   C4.1 exact optimum under independence (dp_varied_cost); its claimed value
          v_ind = the analytic independence value C4.1 reports
  S_r     recommend_portfolio(rho = r) on 10,000 optimisation draws (seed A); exact
          enumeration for m = 10, local search from S_ind for m = 20
  Every portfolio is evaluated on 50,000 fresh draws (seed B) at each true rho; at
  rho = 0 the analytic value is used.
Metrics per cell (mean over scenarios, SE):
  over(rho)        (v_ind(S_ind) - v_rho(S_ind)) / v_rho(S_ind)
  regret(S, rho)   (v_rho(S_rho) - v_rho(S)) / v_rho(S_rho)

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  D1 value reporting (V-C4-2 as written: "高估 > 10% 時預設改用 C4.2"). Moderate
     correlation is fixed at rho = 0.4 in advance, since the real rho is unknown. If
     mean over(0.4) > 10% in any cell, C4's reported expected utility must come from
     C4.2 as a range over rho in {0, 0.4, 0.8}, not the C4.1 value alone.
  D2 choice. The default choice becomes recommend_portfolio(rho = 0.4) only if all of
     (a) mean regret(S_ind, 0.4) > 1% and > 2 SE in at least one cell of each generator;
     (b) at true rho = 0, mean regret(S_0.4, 0) <= 0.5% in every cell (little harm if
         admissions are in fact independent);
     (c) at true rho = 0.8, mean regret(S_0.4, 0.8) < mean regret(S_ind, 0.8) in every
         cell.
     Otherwise the C4.1 choice stays the default and C4.2 is an option.
  Disclosure: before this run the code was only exercised by unit tests and by a random
  search for a single 6-school example where the two choices differ (used in the test).

    python scripts/run_l3_portfolio_corr.py
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np

import aisc.portfolio as pf

ROOT = Path(__file__).resolve().parents[1]
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
MS = (10, 20)
HS = (3, 6)
GENS = ("tiered", "uniform")
N_SCEN = 100
N_OPT, N_EVAL = 10_000, 50_000
SEED_SCEN, SEED_A, SEED_B = 2026, 11, 22
RHO_MID = 0.4


def scenario(gen: str, m: int, rng: np.random.Generator) -> tuple[list[float], list[float]]:
    if gen == "uniform":
        return rng.uniform(0.05, 0.95, m).tolist(), rng.uniform(0.2, 1.0, m).tolist()
    n_r, n_t = round(0.4 * m), round(0.4 * m)
    n_s = m - n_r - n_t
    p = np.concatenate([rng.uniform(0.05, 0.25, n_r), rng.uniform(0.30, 0.60, n_t),
                        rng.uniform(0.70, 0.95, n_s)])  # fmt: skip
    base = np.repeat([0.90, 0.70, 0.45], [n_r, n_t, n_s])
    u = np.clip(base + rng.normal(0, 0.08, m), 0.05, 1.0)
    return p.tolist(), u.tolist()


def value(p, u, s, rho, draws):
    return pf.portfolio_value(p, u, list(s)) if rho == 0 else pf.portfolio_value_mc(u, draws, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=N_SCEN)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-portfolio-corr.md")
    args = ap.parse_args()
    opt_rhos = [r for r in RHOS if r > 0]
    rows = {}
    for gen in GENS:
        for m in MS:
            for h in HS:
                srng = np.random.default_rng([SEED_SCEN, GENS.index(gen), m, h])
                acc = {k: [] for k in ("over", "reg_ind", "reg_mid", "changed")}
                acc = {r: {k: [] for k in acc} for r in RHOS}
                for k in range(args.n_scen):
                    p, u = scenario(gen, m, srng)
                    cost = [1] * m
                    s_ind, v_ind = pf.dp_varied_cost(p, u, cost, h)
                    chosen = {0.0: sorted(s_ind)}
                    for r in opt_rhos:
                        d = pf.admission_draws(
                            p, r, N_OPT, np.random.default_rng([SEED_A, k, int(r * 10)])
                        )
                        chosen[r] = pf.recommend_portfolio(p, u, cost, h, rho=r, draws=d)
                    for r in RHOS:
                        erng = np.random.default_rng([SEED_B, k, int(r * 10)])
                        ev = None if r == 0 else pf.admission_draws(p, r, N_EVAL, erng)
                        v_best = value(p, u, chosen[r], r, ev)
                        v_s_ind = value(p, u, s_ind, r, ev)
                        v_s_mid = value(p, u, chosen[RHO_MID], r, ev)
                        acc[r]["over"].append((v_ind - v_s_ind) / v_s_ind)
                        acc[r]["reg_ind"].append((v_best - v_s_ind) / v_best)
                        acc[r]["reg_mid"].append((v_best - v_s_mid) / v_best)
                        acc[r]["changed"].append(chosen[r] != sorted(s_ind))
                rows[(gen, m, h)] = acc
                print(gen, m, h, "done", flush=True)

    def ms(v):
        v = np.asarray(v, float)
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))

    # decisions
    d1 = any(ms(a[RHO_MID]["over"])[0] > 0.10 for a in rows.values())
    d2a = all(
        any((lambda x: x[0] > 0.01 and x[0] > 2 * x[1])(ms(a[RHO_MID]["reg_ind"]))
            for (g, _, _), a in rows.items() if g == gen)
        for gen in GENS
    )  # fmt: skip
    d2b = all(ms(a[0.0]["reg_mid"])[0] <= 0.005 for a in rows.values())
    d2c = all(ms(a[0.8]["reg_mid"])[0] < ms(a[0.8]["reg_ind"])[0] for a in rows.values())
    d2 = d2a and d2b and d2c

    def pct(x):
        return f"{100 * x[0]:+.1f}±{100 * x[1]:.1f}"

    L = [
        "# V-C4-2：錄取相關時，獨立模型錯多少（C4.2，W-29）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_portfolio_corr.py`",
        f"- 合成情境：2 種生成方式 × 學校數 {MS} × 申請數 {HS}，每格 {args.n_scen} 個情境；"
        f"最佳化 {N_OPT:,} 次模擬、評估另用 {N_EVAL:,} 次新模擬",
        "- 預先登記見腳本檔頭；錄取機率與效用都是合成值（C3 未完成）",
        "",
        "## 判定",
        "",
        f"- D1 回報值（ρ = 0.4 時高估 > 10%）：{'✅ 觸發' if d1 else '❌ 未觸發'}",
        f"- D2 選擇（a 各生成方式至少一格遺憾 > 1%：{'✅' if d2a else '❌'}；"
        f"b ρ = 0 時誤設代價 ≤ 0.5%：{'✅' if d2b else '❌'}；"
        f"c ρ = 0.8 時優於獨立：{'✅' if d2c else '❌'}）→ "
        f"{'預設改用 recommend_portfolio(rho=0.4)' if d2 else '預設仍為 C4.1 選擇'}",
        "",
        "## 高估與遺憾（%，平均 ± 標準誤）",
        "",
        "高估 = 獨立模型宣稱的期望效用比真實值高多少；遺憾 = 比該 ρ 下最佳組合少多少；"
        "改變 = C4.2 選出的組合與 C4.1 不同的比例。",
        "",
        "| 生成 | 學校數 | 申請數 | 真實 ρ | 高估 | 遺憾（C4.1 組合） "
        "| 遺憾（ρ=0.4 組合） | 改變 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (gen, m, h), a in rows.items():
        for r in RHOS:
            L.append(f"| {gen} | {m} | {h} | {r} | {pct(ms(a[r]['over']))} | "
                     f"{pct(ms(a[r]['reg_ind']))} | {pct(ms(a[r]['reg_mid']))} | "
                     f"{np.mean(a[r]['changed']):.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"D1={d1} D2={d2} (a={d2a} b={d2b} c={d2c}) wrote {args.out}")


if __name__ == "__main__":
    main()
