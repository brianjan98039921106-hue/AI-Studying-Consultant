"""Does the W-73 analysis plan behave on data like the planned study? (A-C1-30,
A-C1-31; scripts/analyze_w73_invariance.py.) Synthetic data only.

Worlds: 4 strata x 50 respondents (protocol §2 minimum), the 10 two-item dimensions.
Latent dimension scores on the unit scale, mean 0.5, SD 0.2, correlated: RIASEC
circumplex (adjacent 0.45, alternate 0.20, opposite 0.05), quant-I 0.40,
programming-I 0.30, quant-programming 0.50, research-I 0.40, entrepreneurship-E 0.50
(other pairs 0.10). Item = latent x loading + intercept shift + N(0, noise), clipped to
[0, 1] and rounded to the 5-point scale. Two generator settings: noise 0.10 and 0.18.
Conditions:
  invariant   all strata identical
  impact      stratum latent means differ (distant: A +0.10, C -0.10; multi_degree:
              I +0.10; adjacent: E +0.05); still invariant measurement
  intercept   impact + item riasec_a#2 shifted +0.15 in stratum distant (scalar violation)
  loading     impact + item riasec_s#2 loading 0.4 instead of 1 in stratum distant
              (metric violation)
Each replicate runs the plan's main CFA steps (configural, metric, scalar) and the
binding-tier DIF test exactly as in analyze_w73_invariance.py.

PRE-REGISTERED (written and committed before the first run, 2026-10-29):
  The plan is KEPT if, in both noise settings and both invariant and impact
  conditions: metric false alarms (delta CFI > 0.01) <= 10% of replicates, scalar
  false alarms <= 10%, configural unusable (not converged or CFI < 0.90) <= 10%, and
  the binding-tier DIF false-flag rate over items without DIF <= 5%.
  Otherwise the plan is revised before any study data and the revision re-checked on
  fresh seeds. Report only: detection rates in the intercept and loading conditions
  (scalar / metric delta CFI > 0.01; DIF flag on the shifted item), Heywood cases.
  25 replicates per cell, seed 2029. Disclosure: smoke-tested with 1 replicate.

    python scripts/check_w73_plan.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_w41_study import STRATA, dim_items  # noqa: E402
from analyze_w73_invariance import (  # noqa: E402
    CFI_CONFIGURAL,
    DELTA_CFI,
    MultiGroupCFA,
    dif_tier,
)

ROOT = Path(__file__).resolve().parents[1]
DIMS = [k for k, v in dim_items().items() if len(v) >= 2]
ITEMS = [i for k in DIMS for i in dim_items()[k]]
FAC = [DIMS.index(k) for k in DIMS for _ in dim_items()[k]]
NOISES = (0.10, 0.18)
CONDS = ("invariant", "impact", "intercept", "loading")
N_PER, REPS, SEED = 50, 25, 2029
SHIFT_ITEM, LOAD_ITEM = "riasec_a#2", "riasec_s#2"


def latent_corr() -> np.ndarray:
    R = np.full((10, 10), 0.10)
    ring = ["riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c"]
    for a in range(6):
        for b in range(6):
            dist = min(abs(a - b), 6 - abs(a - b))
            R[DIMS.index(ring[a]), DIMS.index(ring[b])] = {0: 1, 1: 0.45, 2: 0.20, 3: 0.05}[dist]
    pairs = {("quant_intensity", "riasec_i"): 0.40, ("programming_intensity", "riasec_i"): 0.30,
             ("quant_intensity", "programming_intensity"): 0.50,
             ("research_orientation", "riasec_i"): 0.40,
             ("entrepreneurship", "riasec_e"): 0.50}  # fmt: skip
    for (a, b), r in pairs.items():
        R[DIMS.index(a), DIMS.index(b)] = R[DIMS.index(b), DIMS.index(a)] = r
    np.fill_diagonal(R, 1)
    return R


IMPACT = {"distant": {"riasec_a": 0.10, "riasec_c": -0.10}, "multi_degree": {"riasec_i": 0.10},
          "adjacent": {"riasec_e": 0.05}}  # fmt: skip


def world(cond: str, noise: float, rng: np.random.Generator):
    C = 0.2**2 * latent_corr()
    X, groups = [], []
    for s in STRATA:
        mu = np.full(10, 0.5)
        if cond != "invariant":
            for k, d in IMPACT.get(s, {}).items():
                mu[DIMS.index(k)] += d
        lat = rng.multivariate_normal(mu, C, N_PER)
        load = np.ones(len(ITEMS))
        shift = np.zeros(len(ITEMS))
        if s == "distant" and cond == "intercept":
            shift[ITEMS.index(SHIFT_ITEM)] = 0.15
        if s == "distant" and cond == "loading":
            j = ITEMS.index(LOAD_ITEM)
            load[j] = 0.4  # around the mean: 0.5 + 0.4 (latent - 0.5)
            shift[j] = 0.5 * (1 - 0.4)
        y = lat[:, FAC] * load + shift + rng.normal(0, noise, (N_PER, len(ITEMS)))
        X.append(np.round(np.clip(y, 0, 1) * 4) / 4)
        groups += [s] * N_PER
    return X, np.array(groups)


def replicate(args):
    cond, noise, r = args
    rng = np.random.default_rng([SEED, CONDS.index(cond), NOISES.index(noise), r])
    X, groups = world(cond, noise, rng)
    out = {"conf_bad": True, "metric": False, "scalar": False, "heywood": False}
    try:
        cfa = MultiGroupCFA([x.mean(0) for x in X], [np.cov(x.T, ddof=0) for x in X],
                            [len(x) for x in X], FAC)  # fmt: skip
        nf = len(cfa.free_items)
        c = cfa.fit([False] * nf, [False] * 20, [False] * 10)
        ci = cfa._last_idx
        m = cfa.fit([True] * nf, [False] * 20, [False] * 10, c.x, ci)
        mi = cfa._last_idx
        s = cfa.fit([True] * nf, [True] * 20, [True] * 10, m.x, mi)
        out["conf_bad"] = not (c.converged and c.cfi >= CFI_CONFIGURAL)
        out["metric"] = c.cfi - m.cfi > DELTA_CFI
        out["scalar"] = m.cfi - s.cfi > DELTA_CFI
        out["heywood"] = c.boundary or m.boundary or s.boundary
        out["cfi"] = (c.cfi, m.cfi, s.cfi)
    except (ValueError, np.linalg.LinAlgError):
        pass
    A = np.vstack(X)
    ans = {it: np.round(A[:, j] * 4).astype(int) + 1 for j, it in enumerate(ITEMS)}
    match = {}
    for k in range(10):
        tot = (ans[ITEMS[2 * k]] + ans[ITEMS[2 * k + 1]]).astype(float)
        match[ITEMS[2 * k]] = match[ITEMS[2 * k + 1]] = tot
    rows = dif_tier(ITEMS, ans, match, groups)
    out["dif_flags"] = {it for it, _, _, flag, _ in rows if flag}
    return cond, noise, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-w73-plan-check.md")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(c, n, r) for n in NOISES for c in CONDS for r in range(args.reps)]
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for cond, noise, out in ex.map(replicate, jobs):
            res.setdefault((cond, noise), []).append(out)
            print(cond, noise, len(res[(cond, noise)]), flush=True)

    def rate(rows, key):
        return float(np.mean([r[key] for r in rows]))

    def dif_fp(rows, skip):
        clean = [i for i in ITEMS if i != skip]
        return float(np.mean([len(r["dif_flags"] & set(clean)) / len(clean) for r in rows]))

    kept = True
    L = ["# W-73 分析計畫的模擬檢查（合成資料）", "",
         f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/check_w73_plan.py`",
         f"- 4 層 × {N_PER} 人、10 個兩題維度；每格 {args.reps} 次；預先登記見腳本檔頭",
         "", "| 雜訊 | 情境 | 形態不可用 | 量尺 ΔCFI > 0.01 | 截距 ΔCFI > 0.01 | "
         "DIF 偽陽性（每題） | 被操弄題目的 DIF 偵測 | Heywood |",
         "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for noise in NOISES:
        for cond in CONDS:
            rows = res[(cond, noise)]
            skip = SHIFT_ITEM if cond == "intercept" else LOAD_ITEM if cond == "loading" else ""
            cb, mt, sc = rate(rows, "conf_bad"), rate(rows, "metric"), rate(rows, "scalar")
            fp = dif_fp(rows, skip)
            det = float(np.mean([skip in r["dif_flags"] for r in rows])) if skip else None
            if cond in ("invariant", "impact") and (cb > 0.1 or mt > 0.1 or sc > 0.1 or fp > 0.05):
                kept = False
            dets = "—" if det is None else f"{det:.2f}"
            L.append(f"| {noise} | {cond} | {cb:.2f} | {mt:.2f} | {sc:.2f} | {fp:.3f} | "
                     f"{dets} | {rate(rows, 'heywood'):.2f} |")  # fmt: skip
    L[4:4] = [f"- **判定：{'保留計畫' if kept else '計畫須在資料之前修改'}**"]
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"kept={kept} wrote {args.out}")


if __name__ == "__main__":
    main()
