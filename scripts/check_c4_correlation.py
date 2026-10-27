"""C4.2 instance study (W-29): how much does independence (C4.1) misjudge a portfolio?

NOT an L3 result. The C5 simulator has no admission layer (no applicant strength,
no P(admit) generated from a known truth), so V-C4-2 cannot be run yet. This
script only evaluates the closed-form one-factor model on stated instances:

  A. The published Kapur & Hong / OptimalApplication.jl example (h = 1..5).
  B. Random instances: m = 10, p_j ~ U(0.02, 0.95), u_j ~ U{1..100}, h in {3, 5},
     N = 200 per h (seed 0). For each rho in {0.2, 0.4, 0.6, 0.8}:
     - overestimate of the C4.1 greedy portfolio's value: (v_indep - v_rho) / v_rho
     - overstatement of P(at least one admit) in percentage points
     - regret of the C4.1 choice: (v_rho(best under rho) - v_rho(C4.1 choice))
       / v_rho(best under rho), and the share of instances where it is > 1e-9.

Rules fixed before the first run (2026-10-28): the instance distribution, seeds
and the rho grid above; report median, 90th percentile and max; no tuning.

    python scripts/check_c4_correlation.py
"""

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

import numpy as np

from aisc.portfolio import (
    brute_force_correlated,
    correlation_sweep,
    factor_portfolio_value,
    greedy_same_cost,
)

ROOT = Path(__file__).resolve().parents[1]
RHOS = (0.2, 0.4, 0.6, 0.8)
HS = (3, 5)


def kapur_rows():
    p = [0.2, 0.5, 0.1, 0.6, 0.1]
    u = [1, 4, 9, 1, 8]
    order, _ = greedy_same_cost(p, u)
    rows = []
    for h in range(1, 6):
        sweep = correlation_sweep(p, u, order[:h], rhos=RHOS)
        rows.append((h, sweep))
    return rows


def random_study(n: int, seed: int):
    rng = random.Random(seed)
    out = {(h, r): {"over": [], "pany": [], "regret": []} for h in HS for r in RHOS}
    for _ in range(n):
        m = 10
        p = [round(rng.uniform(0.02, 0.95), 3) for _ in range(m)]
        u = [rng.randint(1, 100) for _ in range(m)]
        order, _ = greedy_same_cost(p, u)
        for h in HS:
            chosen = order[:h]
            for row in correlation_sweep(p, u, chosen, rhos=RHOS):
                rec = out[(h, row.rho)]
                rec["over"].append(row.overestimate)
                rec["pany"].append(100 * (row.p_any_independent - row.p_any_correlated))
                _, best = brute_force_correlated(p, u, rho=row.rho, h=h)
                got = factor_portfolio_value(p, u, chosen, rho=row.rho).expected_utility
                rec["regret"].append((best - got) / best if best > 0 else 0.0)
    return out


def _q(xs, q):
    return float(np.quantile(np.asarray(xs), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "c4-correlation-check.md")
    args = ap.parse_args()

    lines = [
        "# C4.2 相關錄取：實例檢查（W-29）",
        "",
        f"產生日期：{date.today().isoformat()} · `python scripts/check_c4_correlation.py`"
        f"（instances={args.instances}, seed={args.seed}）",
        "",
        "> **不是 L3**：C5 模擬器沒有錄取層，V-C4-2 尚未執行。以下是單因子 Gaussian copula"
        "（`factor_portfolio_value`，確定性積分）在指定實例上的計算結果。",
        "",
        "## A. Kapur & Hong 範例（p = 0.2, 0.5, 0.1, 0.6, 0.1；u = 1, 4, 9, 1, 8）",
        "",
        "C4.1 greedy 前 h 個的組合；高估 = (v_indep − v_ρ) / v_ρ。",
        "",
        "| h | v_indep | " + " | ".join(f"v(ρ={r})" for r in RHOS) + " | "
        + " | ".join(f"高估 ρ={r}" for r in RHOS) + " |",
        "|---|---|" + "---|" * (2 * len(RHOS)),
    ]  # fmt: skip
    for h, sweep in kapur_rows():
        vals = " | ".join(f"{s.correlated:.4f}" for s in sweep)
        over = " | ".join(f"{100 * s.overestimate:.1f}%" for s in sweep)
        lines.append(f"| {h} | {sweep[0].independent:.4f} | {vals} | {over} |")

    res = random_study(args.instances, args.seed)
    lines += [
        "",
        f"## B. 隨機實例（m = 10，N = {args.instances}／h）",
        "",
        "p ~ U(0.02, 0.95)、u ~ U{1..100}；組合 = C4.1 greedy 前 h 個。"
        "遺憾 = 相關下最佳組合（暴力法）與 C4.1 組合在相關下的價值差，除以前者。",
        "",
        "| h | ρ | 高估 中位數 | 高估 P90 | 高估 最大 | P(≥1 錄取) 高估（百分點）中位數 | 最大 "
        "| C4.1 非最佳比例 | 遺憾 中位數 | 遺憾 P90 | 遺憾 最大 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h in HS:
        for r in RHOS:
            d = res[(h, r)]
            o, pa, rg = d["over"], d["pany"], d["regret"]
            share = sum(x > 1e-9 for x in rg) / len(rg)
            lines.append(
                f"| {h} | {r} | {100 * _q(o, .5):.1f}% | {100 * _q(o, .9):.1f}% | "
                f"{100 * max(o):.1f}% | {_q(pa, .5):.1f} | {max(pa):.1f} | {100 * share:.1f}% | "
                f"{100 * _q(rg, .5):.2f}% | {100 * _q(rg, .9):.2f}% | {100 * max(rg):.2f}% |"
            )  # fmt: skip
    lines += [
        "",
        "## 讀法",
        "- ρ 是扣除 C3 已用到的申請者特徵後的殘差相關（A-C4-6）；目前無資料可估計（B-C4-8），"
        "所以只報告掃描範圍。",
        "- 高估與 P(≥1 錄取) 只描述「C4.1 給的數字錯多少」；遺憾描述「C4.1 選錯組合的代價」。",
        "- 檢查：ρ=0 時本模型與 C4.1 解析值一致、ρ=1 時等於 comonotone 封閉解"
        "（`tests/test_c4_correlation.py`）。",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
