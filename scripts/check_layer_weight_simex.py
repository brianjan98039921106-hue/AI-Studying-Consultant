"""Parameter recovery for the SIMEX-corrected layer-weight estimator (W-76, B-C2-12).

Same world and choice model as scripts/check_layer_weight_recovery.py (F85-F87):
v2 world, programme pairs drawn uniformly, choices from the pairwise logit with TRUE
needs and known layer shares (residual dims 10% of the weight, consistency 0.75);
the estimator sees only the STATED needs (51 items, response noise + 0.05
acquiescence, clipped to [0, 1]).

Estimator: discovery.layer_weights.estimate_layer_weights_simex with lambdas
(0.5, 1, 1.5, 2), 20 simulations each, quadratic extrapolation to lambda = -1;
interval = naive 100-draw respondent bootstrap shifted by the correction.
The answer error SD given to SIMEX ("sigma") is the true noise SD, or deliberately
wrong by x0.7 / x1.3 (a retest in W-41 will only estimate it).

Grid: generator {archetype, continuous} x shares {dim, equal, mid} x noise
{0.15, 0.30} x sigma factor {1.0, 0.7, 1.3}; N = 200 respondents, K = 20 pairs;
40 replications each.

PRE-REGISTERED (written and committed before the first run, 2026-10-27), with the
true sigma and noise 0.15, for every generator x shares (the same criteria as the
naive estimator in F85-F87):
  R1 bias: |mean SIMEX estimate - true| <= 0.05 for every layer share.
  R2 coverage: the shifted 95% interval covers the true domain share in >= 85%.
  R3 discrimination: the interval excludes the other hypothesis value in >= 80%.
  If R1-R3 hold: SIMEX is the W-76 analysis method (with sigma from the W-41
  retest); noise 0.30 and the wrong-sigma rows are reported as robustness only.
  If they fail: W-76 needs a different correction (e.g. a latent-variable model)
  before real data are analysed.
  Seeds: --seed 1600. Disclosure: a smoke run on seed 12 only checks that the
  script runs.

    python scripts/check_layer_weight_simex.py
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.dimensions import LAYERS, NEED_KEYS_V2, keys_in_layer
from aisc.discovery.layer_weights import estimate_layer_weights_simex, layer_differences
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.world_v2 import stated, v2_catalog, v2_students

ROOT = Path(__file__).resolve().parents[1]
SIZES = np.array([len(keys_in_layer(lay, "v2")) for lay in LAYERS], float)
SHARES = {
    "dim": SIZES / SIZES.sum(),
    "equal": np.full(4, 0.25),
    "mid": np.array([1 / 6, 1 / 6, 0.5, 1 / 6]),
}
DOMAIN = LAYERS.index("domain")
RESIDUAL_SHARE = 0.10
CONSISTENCY = 0.75
BIAS = 0.05
N, K = 200, 20


def scale_for(G: np.ndarray, wts: np.ndarray) -> float:
    lo, hi = 0.0, 1e5
    for _ in range(80):
        c = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(G @ (c * wts))))
        lo, hi = (c, hi) if np.mean(np.maximum(p, 1 - p)) < CONSISTENCY else (lo, c)
    return (lo + hi) / 2


def one_rep(world, students, shares, noise, sig_factor, rng):
    X, Xres = world
    pick = rng.choice(len(students), N, replace=False)
    G, st_rows, pairs = [], [], []
    for j, i in enumerate(pick):
        _, w2, r = students[i]
        a = rng.integers(0, len(X), K)
        b = (a + rng.integers(1, len(X), K)) % len(X)
        d = np.column_stack([layer_differences(w2, X, "v2"), ((Xres - r) ** 2).mean(axis=1)])
        G.append(d[b] - d[a])
        ws = stated(w2, NEED_KEYS_V2, rng, noise, BIAS)
        st_rows.append([ws[k] for k in NEED_KEYS_V2])
        pairs += [(j, ai, bi) for ai, bi in zip(a, b, strict=True)]
    G = np.vstack(G)
    wts = np.append(shares * (1 - RESIDUAL_SHARE), RESIDUAL_SHARE)
    p = 1 / (1 + np.exp(-(G @ (scale_for(G, wts) * wts))))
    y = (rng.random(len(p)) < p).astype(float)
    est = estimate_layer_weights_simex(np.array(st_rows), X, np.array(pairs), y,
                                       noise * sig_factor, "v2", n_sim=20, n_boot=100,
                                       rng=rng)  # fmt: skip
    c, nv = est.corrected, est.naive
    return (np.array([c.shares[lay] for lay in LAYERS]), c.ci["domain"],
            np.array([nv.shares[lay] for lay in LAYERS]))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1600)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--noises", type=float, nargs="+", default=[0.15, 0.30])
    ap.add_argument("--factors", type=float, nargs="+", default=[1.0, 0.7, 1.3])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "layer-weight-simex.md")  # fmt: skip
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, _, residual = v2_catalog(src, hidden, rng)
    world = (np.array([p.vector for p in cat_v2]),
             np.array([residual[p.program_id] for p in cat_v2]))  # fmt: skip

    rows = []
    for gen in ("archetype", "continuous"):
        cfg = SimConfig(n_students=400, prestige_weight=0.5, needs_model=gen,
                        field_model="all_fields")  # fmt: skip
        pop = generate_population(src, cfg, np.random.default_rng(args.seed + 1), hidden=hidden)
        students = v2_students(pop, np.random.default_rng([args.seed + 1, 1]))
        for (name, sh), noise, fac in product(SHARES.items(), args.noises, args.factors):
            rr = np.random.default_rng([args.seed + 2, len(rows)])
            out = [one_rep(world, students, sh, noise, fac, rr) for _ in range(args.reps)]
            E = np.array([o[0] for o in out])
            C = np.array([o[1] for o in out])
            Nv = np.array([o[2] for o in out])
            bias = E.mean(axis=0) - sh
            cover = float(np.mean((C[:, 0] <= sh[DOMAIN]) & (sh[DOMAIN] <= C[:, 1])))
            excl = {o: float(np.mean((C[:, 0] > v) | (C[:, 1] < v)))
                    for o, v in (("dim", SHARES["dim"][DOMAIN]), ("equal", 0.25))}  # fmt: skip
            rows.append(dict(gen=gen, shares=name, noise=noise, fac=fac, bias=bias,
                             naive=Nv.mean(axis=0) - sh, cover=cover,
                             width=float(np.mean(C[:, 1] - C[:, 0])), excl=excl))  # fmt: skip
            nd = rows[-1]["naive"][DOMAIN]
            print(f"  {gen} {name} noise={noise} sigma x{fac} bias_dom={bias[DOMAIN]:+.3f} "
                  f"naive={nd:+.3f} cover={cover:.2f}", flush=True)  # fmt: skip

    other = {"dim": ["equal"], "equal": ["dim"], "mid": ["dim", "equal"]}
    verdict = []
    for r in rows:
        if r["noise"] == 0.15 and r["fac"] == 1.0:
            verdict.append((r["gen"], r["shares"], bool(np.all(np.abs(r["bias"]) <= 0.05)),
                            r["cover"] >= 0.85,
                            all(r["excl"][o] >= 0.80 for o in other[r["shares"]])))  # fmt: skip
    yn = {True: "✅", False: "❌"}
    L = [
        "# SIMEX 校正後的層權重參數還原（W-76）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令："
        f"`python scripts/check_layer_weight_simex.py --seed {args.seed}`",
        f"- N = {N} 人、K = {K} 組；每格 {args.reps} 次重複；λ = 0.5, 1, 1.5, 2，各 20 次模擬",
        "- 區間 = 未校正自助法區間平移校正量；sigma × 係數 = 給 SIMEX 的作答誤差（1.0 為正確值）",
        "- 預先登記見 `scripts/check_layer_weight_simex.py` 檔頭",
        "",
        "## 判定（雜訊 0.15、正確 sigma）",
        "",
        "| 生成模型 | 真實份量 | R1 偏誤 ≤ 0.05 | R2 涵蓋率 ≥ 0.85 | R3 區分另一假設 ≥ 0.80 |",
        "|---|---|---|---|---|",
    ]
    L += [f"| {g} | {s} | {yn[a]} | {yn[b]} | {yn[c]} |" for g, s, a, b, c in verdict]
    L += ["", "## 全部設計", "",
          "| 生成模型 | 份量 | 雜訊 | sigma 係數 | SIMEX 偏誤（興趣、強度、領域、取向） "
          "| 未校正領域偏誤 | 領域涵蓋率 | 區間寬度 | 排除 dim 值 | 排除 0.25 |",
          "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        b = ", ".join(f"{v:+.3f}" for v in r["bias"])
        L.append(f"| {r['gen']} | {r['shares']} | {r['noise']} | {r['fac']} | {b} | "
                 f"{r['naive'][DOMAIN]:+.3f} | {r['cover']:.2f} | {r['width']:.3f} | "
                 f"{r['excl']['dim']:.2f} | {r['excl']['equal']:.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    ok = bool(verdict) and all(a and b and c for *_, a, b, c in verdict)
    print(f"all_pass={ok} wrote {args.out}")


if __name__ == "__main__":
    main()
