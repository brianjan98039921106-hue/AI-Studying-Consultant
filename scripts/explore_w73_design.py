"""Which study design would let the W-73 invariance / DIF analysis work? EXPLORATORY,
report only (results/synthetic-w73-plan-check.md failed its pre-registered check:
F151-F153). Synthetic data only; same worlds and conditions as check_w73_plan.py
(impact, intercept, loading; noise 0.10 and 0.18), 15 replicates per cell, seed 2030.

Designs (each run through configural / metric / scalar and binding-tier DIF):
  E1  2 items per dimension, the planned 200 people analysed as 2 groups (near/far)
  E2  4 items per dimension, 4 strata x 50
  E3  2 items per dimension, 4 strata x 100 (400 people)
  E4  2 items per dimension, 4 strata x 50, tau-equivalent model (all loadings 1:
      no metric step, scalar step only)
  E5  2 items per dimension, 4 strata x 100 analysed as 2 groups (near/far, 200 each)
DIF is reported as planned (BH p < 0.05 and dR2 >= 0.035) and with p only.
NOTHING IS SELECTED HERE: the design is the study owner's decision (items per
dimension and sample size, study-protocol §7). The chosen design's rules are then
pre-registered and confirmed on fresh seeds before any data.

    python scripts/explore_w73_design.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_w41_study import STRATA  # noqa: E402
from analyze_w73_invariance import (  # noqa: E402
    CFI_CONFIGURAL,
    DELTA_CFI,
    FDR,
    NEAR_FAR,
    MultiGroupCFA,
    dif_tier,
)
from check_w73_plan import DIMS, IMPACT, NOISES, latent_corr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = {"E1": (2, 50, "near_far", False), "E2": (4, 50, "strata", False),
           "E3": (2, 100, "strata", False), "E4": (2, 50, "strata", True),
           "E5": (2, 100, "near_far", False)}  # fmt: skip
CONDS = ("impact", "intercept", "loading")
REPS, SEED = 15, 2030


def world(cond, noise, k_items, n_per, rng):
    items = [f"{d}#{j + 1}" for d in DIMS for j in range(k_items)]
    fac = [DIMS.index(d) for d in DIMS for _ in range(k_items)]
    shift_item, load_item = "riasec_a#2", "riasec_s#2"
    C = 0.2**2 * latent_corr()
    X, groups = [], []
    for s in STRATA:
        mu = np.full(10, 0.5)
        for d, v in IMPACT.get(s, {}).items():
            mu[DIMS.index(d)] += v
        lat = rng.multivariate_normal(mu, C, n_per)
        load, shift = np.ones(len(items)), np.zeros(len(items))
        if s == "distant" and cond == "intercept":
            shift[items.index(shift_item)] = 0.15
        if s == "distant" and cond == "loading":
            j = items.index(load_item)
            load[j], shift[j] = 0.4, 0.3
        y = lat[:, fac] * load + shift + rng.normal(0, noise, (n_per, len(items)))
        X.append(np.round(np.clip(y, 0, 1) * 4) / 4)
        groups += [s] * n_per
    return items, fac, X, np.array(groups)


def replicate(args):
    design, cond, noise, r = args
    k_items, n_per, grouping, tau_eq = DESIGNS[design]
    rng = np.random.default_rng([SEED, list(DESIGNS).index(design), CONDS.index(cond),
                                 NOISES.index(noise), r])  # fmt: skip
    items, fac, X, groups = world(cond, noise, k_items, n_per, rng)
    A = np.vstack(X)
    if grouping == "near_far":
        lab = np.array([NEAR_FAR[g] for g in groups])
        X = [A[lab == v] for v in ("near", "far")]
    out = {"conf_bad": True, "metric": False, "scalar": False, "heywood": False}
    try:
        cfa = MultiGroupCFA([x.mean(0) for x in X], [np.cov(x.T, ddof=0) for x in X],
                            [len(x) for x in X], fac, tau_equivalent=tau_eq)  # fmt: skip
        nf, p = len(cfa.free_items), len(items)
        c = cfa.fit([False] * nf, [False] * p, [False] * 10)
        ci = cfa._last_idx
        m = cfa.fit([True] * nf, [False] * p, [False] * 10, c.x, ci)
        mi = cfa._last_idx
        s = cfa.fit([True] * nf, [True] * p, [True] * 10, m.x, mi)
        out["conf_bad"] = not (c.converged and c.cfi >= CFI_CONFIGURAL)
        out["metric"] = (not tau_eq) and c.cfi - m.cfi > DELTA_CFI
        out["scalar"] = m.cfi - s.cfi > DELTA_CFI
        out["heywood"] = c.boundary or m.boundary or s.boundary
    except (ValueError, np.linalg.LinAlgError):
        pass
    ans = {it: np.round(A[:, j] * 4).astype(int) + 1 for j, it in enumerate(items)}
    match = {}
    for d in range(10):
        its = [items[j] for j in range(len(items)) if fac[j] == d]
        tot = sum(ans[i] for i in its).astype(float)
        for i in its:
            match[i] = tot
    rows = dif_tier(items, ans, match, groups)
    out["dif"] = {it for it, _, _, flag, _ in rows if flag}
    out["dif_p"] = {it for it, _, pa, _, _ in rows if pa < FDR}
    out["n_items"] = len(items)
    return design, cond, noise, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-w73-design.md")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(d, c, n, r) for d in DESIGNS for n in NOISES for c in CONDS
            for r in range(args.reps)]  # fmt: skip
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for d, c, n, out in ex.map(replicate, jobs):
            res.setdefault((d, c, n), []).append(out)
            print(d, c, n, len(res[(d, c, n)]), flush=True)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    L = ["# W-73 研究設計探索（合成資料，只報告）", "",
         f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/explore_w73_design.py`",
         f"- 每格 {args.reps} 次；**探索性，不做選擇**：設計由研究負責人決定後，"
         "再預先登記並以新種子確認",
         "", "| 設計 | 雜訊 | 情境 | 形態不可用 | 量尺 ΔCFI > 0.01 | 截距 ΔCFI > 0.01 | "
         "DIF 偽陽性（計畫判準） | 被操弄題目 DIF（計畫判準） | 被操弄題目 DIF（只看 p） "
         "| Heywood |",
         "|---" * 10 + "|"]  # fmt: skip
    for d in DESIGNS:
        for n in NOISES:
            for c in CONDS:
                rows = res[(d, c, n)]
                tgt = {"intercept": "riasec_a#2", "loading": "riasec_s#2"}.get(c, "")
                fp = float(np.mean([len(r["dif"] - {tgt}) / (r["n_items"] - bool(tgt))
                                    for r in rows]))  # fmt: skip
                det = "—" if not tgt else f"{np.mean([tgt in r['dif'] for r in rows]):.2f}"
                detp = "—" if not tgt else f"{np.mean([tgt in r['dif_p'] for r in rows]):.2f}"
                met = "—" if DESIGNS[d][3] else f"{mean(rows, 'metric'):.2f}"
                L.append(f"| {d} | {n} | {c} | {mean(rows, 'conf_bad'):.2f} | {met} | "
                         f"{mean(rows, 'scalar'):.2f} | {fp:.3f} | {det} | {detp} | "
                         f"{mean(rows, 'heywood'):.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
