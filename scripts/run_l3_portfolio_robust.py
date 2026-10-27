"""V-C4-3: how much do errors in P(admit) cost, and does a robust criterion help?
(W-2A, B-C4-2; C4.3, A-C4-6, A-C4-7.)

No C3 yet, so everything is synthetic. True probabilities and utilities come from the
two scenario generators of run_l3_portfolio_corr.py (tiered, uniform); m = 10
schools, budget H in {3, 6} (equal cost), 200 cases per cell. The system only sees
an estimate and an interval, from one of two error models:
  mult   p_hat = clip(p * (1 + d), 0.01, 0.99), d ~ U[-0.2, 0.2] (V-C4-3's "+-20%");
         interval [p_hat / 1.2, min(1, p_hat / 0.8)], which contains p
  logit  logit p_hat = logit p + N(0, 0.5^2); interval expit(logit p_hat +- 1.645 * 0.5),
         a 90% interval (p is outside it in ~10% of schools)
Criteria (independent admissions):
  E  C4.1 optimum at p_hat (current default)
  W  worst case: C4.1 optimum at the interval's lower bounds
  R  minimax regret over the box, 200 scenarios + corners (seeded per case)
Metrics under the TRUE p: regret = (v*(p) - v(S, p)) / v*(p); miss = share of cases
where S is not the true optimum; 95th percentile of regret (tail).

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  A criterion X in {R, W} becomes the default instead of E if
    (a) its mean regret is below E's by more than 2 SE (paired) in at least one cell
        of each error model, and
    (b) its mean regret is not above E's by more than 2 SE in any cell.
  If both qualify, the one with the lower mean regret averaged over all cells.
  Otherwise E stays the default and C4.3 remains an option.
  Report only: miss rate (V-C4-3's "change rate"), 95th-percentile regret.
  Disclosure: before this run the code was only exercised by unit tests and a smoke
  run with 2 cases per cell (exit status only).

    python scripts/run_l3_portfolio_robust.py
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np

import aisc.portfolio as pf

ROOT = Path(__file__).resolve().parents[1]
GENS = ("tiered", "uniform")
ERRORS = ("mult", "logit")
HS = (3, 6)
M = 10
N_CASES = 200
N_SCEN = 200
SEED = 2027
CRITERIA = ("E", "W", "R")


def scenario(gen: str, m: int, rng: np.random.Generator):
    if gen == "uniform":
        return rng.uniform(0.05, 0.95, m), rng.uniform(0.2, 1.0, m)
    n_r, n_t = round(0.4 * m), round(0.4 * m)
    n_s = m - n_r - n_t
    p = np.concatenate([rng.uniform(0.05, 0.25, n_r), rng.uniform(0.30, 0.60, n_t),
                        rng.uniform(0.70, 0.95, n_s)])  # fmt: skip
    base = np.repeat([0.90, 0.70, 0.45], [n_r, n_t, n_s])
    return p, np.clip(base + rng.normal(0, 0.08, m), 0.05, 1.0)


def estimate(err: str, p: np.ndarray, rng: np.random.Generator):
    if err == "mult":
        ph = np.clip(p * (1 + rng.uniform(-0.2, 0.2, len(p))), 0.01, 0.99)
        lo, hi = ph / 1.2, np.minimum(1.0, ph / 0.8)
        return ph, lo, hi
    z = np.log(p / (1 - p)) + rng.normal(0, 0.5, len(p))

    def expit(x):
        return 1 / (1 + np.exp(-x))

    return expit(z), expit(z - 1.645 * 0.5), expit(z + 1.645 * 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=N_CASES)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results" / "synthetic-l3-portfolio-robust.md"
    )
    args = ap.parse_args()
    res = {}
    for err in ERRORS:
        for gen in GENS:
            for h in HS:
                rng = np.random.default_rng([SEED, ERRORS.index(err), GENS.index(gen), h])
                acc = {c: {"regret": [], "miss": []} for c in CRITERIA}
                cost = [1] * M
                for k in range(args.n_cases):
                    p, u = scenario(gen, M, rng)
                    ph, lo, hi = estimate(err, p, rng)
                    s_true, v_true = pf.dp_varied_cost(list(p), list(u), cost, h)
                    chosen = {
                        "E": pf.dp_varied_cost(list(ph), list(u), cost, h)[0],
                        "W": pf.worst_case_portfolio(list(lo), list(u), cost, h),
                        "R": pf.minimax_regret_portfolio(
                            list(lo), list(hi), list(ph), list(u), cost, h,
                            n_scenarios=N_SCEN, rng=np.random.default_rng([SEED, k, h]),
                        ),
                    }  # fmt: skip
                    for c, s in chosen.items():
                        v = pf.portfolio_value(list(p), list(u), list(s))
                        acc[c]["regret"].append((v_true - v) / v_true)
                        acc[c]["miss"].append(float(sorted(s) != sorted(s_true)))
                res[(err, gen, h)] = acc
                print(err, gen, h, flush=True)

    def paired(a, b):
        d = np.asarray(a) - np.asarray(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    qual = {}
    for x in ("R", "W"):
        better = {e: False for e in ERRORS}
        worse = False
        for (err, _, _), acc in res.items():
            d, se = paired(acc[x]["regret"], acc["E"]["regret"])
            if d < -2 * se:
                better[err] = True
            if d > 2 * se:
                worse = True
        qual[x] = all(better.values()) and not worse
    mean_reg = {
        x: float(np.mean([np.mean(a[x]["regret"]) for a in res.values()])) for x in CRITERIA
    }
    ok = [x for x in ("R", "W") if qual[x]]
    adopted = min(ok, key=lambda x: mean_reg[x]) if ok else "E"

    def pct(v):
        return f"{100 * np.mean(v):.2f}±{100 * np.std(v, ddof=1) / math.sqrt(len(v)):.2f}"

    L = [
        "# V-C4-3：錄取機率估計誤差的代價與穩健選擇（C4.3，W-2A）",
        "",
        f"- 產生日期：{date.today().isoformat()} · "
        "指令：`python scripts/run_l3_portfolio_robust.py`",
        f"- {M} 間學校、申請數 {HS}、每格 {args.n_cases} 個情境；2 種估計誤差 × 2 種情境生成",
        "- 預先登記見腳本檔頭；機率、區間與效用都是合成值（C3 未完成）",
        "",
        "## 判定",
        "",
        f"- R（最小化最大遺憾）：{'✅' if qual['R'] else '❌'}；W（最壞情況）："
        f"{'✅' if qual['W'] else '❌'} → 預設 **{adopted}**",
        "",
        "## 真實機率下的遺憾（%，平均 ± 標準誤）、95 百分位、選錯率",
        "",
        "| 誤差 | 生成 | 申請數 | E 遺憾 | W 遺憾 | R 遺憾 | W − E | R − E | E p95 | W p95 "
        "| R p95 | E 選錯 | W 選錯 | R 選錯 |",
        "|---" * 14 + "|",
    ]
    for (err, gen, h), a in res.items():
        dw, dr = (
            paired(a["W"]["regret"], a["E"]["regret"]),
            paired(a["R"]["regret"], a["E"]["regret"]),
        )
        L.append(
            f"| {err} | {gen} | {h} | "
            + " | ".join(pct(a[c]["regret"]) for c in CRITERIA)
            + f" | {100 * dw[0]:+.2f}±{100 * dw[1]:.2f} | {100 * dr[0]:+.2f}±{100 * dr[1]:.2f} | "
            + " | ".join(f"{100 * np.percentile(a[c]['regret'], 95):.2f}" for c in CRITERIA)
            + " | "
            + " | ".join(f"{np.mean(a[c]['miss']):.2f}" for c in CRITERIA)
            + " |"
        )
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"R={qual['R']} W={qual['W']} adopted={adopted} wrote {args.out}")


if __name__ == "__main__":
    main()
