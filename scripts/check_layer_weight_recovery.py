"""Parameter recovery for the layer-weight estimator (W-76, docs/layer-weights.md §6).

Can the W-76 study tell the two simulation truths apart, and how many students does
it need? World: the v1/v2 comparison world (simulation/world_v2.py). For each
respondent: true v2 needs w (51 dims) and 3 residual dims; K programme pairs drawn
uniformly from the 180-programme catalogue; choices from the pairwise logit with
TRUE needs and known layer shares; the estimator only sees the STATED needs
(51 items, response noise + 0.05 acquiescence) and the programme vectors.

Known shares (interest, intensity, domain, orientation):
    dim     proportional to layer size (6, 2, 41, 2) / 51  -> domain 0.804
    equal   0.25 each
    mid     domain 0.5, the other three 1/6 each
The residual dims carry 10% of the total weight in the generating utility (unmeasured
fit, acts as noise). Scale: beta = c * shares, with c set by bisection so that the
mean probability of choosing the truly better programme (noise-free) is 0.75
(assumed consistency of real respondents; reported, not tuned).

Grid: generator {archetype, continuous} x shares {dim, equal, mid} x answer noise
{0.15, 0.30} x respondents N {100, 200, 400} x pairs K {10, 20}; 40 replications
each; 95% interval from a 100-draw respondent bootstrap.

PRE-REGISTERED (written and committed before the first run, 2026-10-27), evaluated
at the reference design N = 200, K = 20, noise 0.15, for every generator x shares:
  R1 bias: |mean estimate - true| <= 0.05 for every layer share.
  R2 coverage: the 95% interval covers the true domain share in >= 85% of
     replications.
  R3 discrimination: the interval for the domain share excludes the other
     hypothesis (truth dim -> excludes 0.25; truth equal -> excludes 0.804) in
     >= 80% of replications (reported for 'mid' against both).
  If R1 fails: B-C2-12 is confirmed and SIMEX correction must be added before W-76
  data are analysed. Smallest N meeting R1-R3 per design is reported.
  Seeds: --seed 1500 (catalogue 1500, population 1501, replications 1502...).
  Disclosure: a smoke run on seed 11 (tiny grid) only checks that the script runs.

    python scripts/check_layer_weight_recovery.py
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.dimensions import LAYERS, NEED_KEYS_V2, keys_in_layer
from aisc.discovery.layer_weights import estimate_layer_weights, layer_differences
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
REF = (200, 20, 0.15)


def generating_features(w2, r, X, Xres, a, b):
    """True (noise-free) feature rows: 4 layers + residual, D(B) - D(A)."""
    d = layer_differences(w2, X, "v2")
    dres = ((Xres - r) ** 2).mean(axis=1)
    full = np.column_stack([d, dres])
    return full[b] - full[a]


def scale_for(G: np.ndarray, shares: np.ndarray) -> float:
    """c such that mean max(p, 1 - p) = CONSISTENCY for beta = c * weights."""
    wts = np.append(shares * (1 - RESIDUAL_SHARE), RESIDUAL_SHARE)
    lo, hi = 0.0, 1e5
    for _ in range(80):
        c = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(G @ (c * wts))))
        if np.mean(np.maximum(p, 1 - p)) < CONSISTENCY:
            lo = c
        else:
            hi = c
    return (lo + hi) / 2


def one_rep(world, students, shares, n, k, noise, rng):
    cat_v2, X, Xres = world
    pick = rng.choice(len(students), n, replace=False)
    G, Fm, resp = [], [], []
    for i in pick:
        st, w2, r = students[i]
        a = rng.integers(0, len(cat_v2), k)
        b = (a + rng.integers(1, len(cat_v2), k)) % len(cat_v2)  # b != a
        G.append(generating_features(w2, r, X, Xres, a, b))
        ws = stated(w2, NEED_KEYS_V2, rng, noise, BIAS)
        wm = np.array([ws[key] for key in NEED_KEYS_V2])
        dm = layer_differences(wm, X, "v2")
        Fm.append(dm[b] - dm[a])
        resp += [st.student_id] * k
    G, Fm = np.vstack(G), np.vstack(Fm)
    c = scale_for(G, shares)
    wts = np.append(shares * (1 - RESIDUAL_SHARE), RESIDUAL_SHARE)
    p = 1 / (1 + np.exp(-(G @ (c * wts))))
    y = (rng.random(len(p)) < p).astype(float)
    est = estimate_layer_weights(Fm, y, resp, n_boot=100, rng=rng)
    return np.array([est.shares[lay] for lay in LAYERS]), est.ci["domain"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1500)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.15, 0.30])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "layer-weight-recovery.md")  # fmt: skip
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, _, residual = v2_catalog(src, hidden, rng)
    X = np.array([p.vector for p in cat_v2])
    Xres = np.array([residual[p.program_id] for p in cat_v2])
    world = (cat_v2, X, Xres)

    rows = []
    for gen in ("archetype", "continuous"):
        cfg = SimConfig(n_students=max(args.ns), prestige_weight=0.5, needs_model=gen,
                        field_model="all_fields")  # fmt: skip
        pop = generate_population(src, cfg, np.random.default_rng(args.seed + 1), hidden=hidden)
        students = v2_students(pop, np.random.default_rng([args.seed + 1, 1]))
        for (name, sh), noise, n, k in product(SHARES.items(), args.noises, args.ns, args.ks):
            rr = np.random.default_rng([args.seed + 2, len(rows)])
            ests, cis = zip(*(one_rep(world, students, sh, n, k, noise, rr)
                              for _ in range(args.reps)), strict=True)  # fmt: skip
            E, C = np.array(ests), np.array(cis)
            bias = E.mean(axis=0) - sh
            cover = np.mean((C[:, 0] <= sh[DOMAIN]) & (sh[DOMAIN] <= C[:, 1]))
            excl = {o: float(np.mean((C[:, 0] > v) | (C[:, 1] < v)))
                    for o, v in (("dim", SHARES["dim"][DOMAIN]), ("equal", 0.25))}  # fmt: skip
            rows.append(dict(gen=gen, shares=name, noise=noise, n=n, k=k, bias=bias,
                             rmse=float(np.sqrt(np.mean((E[:, DOMAIN] - sh[DOMAIN]) ** 2))),
                             cover=float(cover), width=float(np.mean(C[:, 1] - C[:, 0])),
                             excl=excl))  # fmt: skip
            print(f"  {gen} {name} noise={noise} n={n} k={k} bias_dom={bias[DOMAIN]:+.3f} "
                  f"cover={cover:.2f}", flush=True)  # fmt: skip

    def other(name):
        return {"dim": ["equal"], "equal": ["dim"], "mid": ["dim", "equal"]}[name]

    verdict = []
    for r in rows:
        if (r["n"], r["k"], r["noise"]) != REF:
            continue
        r1 = bool(np.all(np.abs(r["bias"]) <= 0.05))
        r2 = r["cover"] >= 0.85
        r3 = all(r["excl"][o] >= 0.80 for o in other(r["shares"]))
        verdict.append((r["gen"], r["shares"], r1, r2, r3))

    yn = {True: "✅", False: "❌"}
    L = [
        "# 層權重估計器的參數還原（W-76）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令："
        f"`python scripts/check_layer_weight_recovery.py --seed {args.seed}`",
        f"- 每格 {args.reps} 次重複；區間 = 以受訪者為單位的 100 次自助法 95% 區間",
        f"- 真實份量（興趣、強度、領域、取向）：dim = {np.round(SHARES['dim'], 3).tolist()}；"
        "equal = 0.25；mid = 領域 0.5、其他 1/6",
        f"- 殘差維度佔生成效用 {RESIDUAL_SHARE:.0%}；選擇一致性 {CONSISTENCY}；學程成對隨機抽取",
        "- 預先登記見 `scripts/check_layer_weight_recovery.py` 檔頭",
        "",
        f"## 判定（參考設計 N = {REF[0]}、K = {REF[1]}、雜訊 {REF[2]}）",
        "",
        "| 生成模型 | 真實份量 | R1 偏誤 ≤ 0.05 | R2 涵蓋率 ≥ 0.85 | R3 區分另一假設 ≥ 0.80 |",
        "|---|---|---|---|---|",
    ]
    L += [f"| {g} | {s} | {yn[a]} | {yn[b]} | {yn[c]} |" for g, s, a, b, c in verdict]
    L += ["", "## 全部設計", "",
          "| 生成模型 | 份量 | 雜訊 | N | K | 偏誤（興趣、強度、領域、取向） | 領域 RMSE "
          "| 領域涵蓋率 | 區間寬度 | 排除 dim 值 | 排除 0.25 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        b = ", ".join(f"{v:+.3f}" for v in r["bias"])
        L.append(f"| {r['gen']} | {r['shares']} | {r['noise']} | {r['n']} | {r['k']} | {b} | "
                 f"{r['rmse']:.3f} | {r['cover']:.2f} | {r['width']:.3f} | "
                 f"{r['excl']['dim']:.2f} | {r['excl']['equal']:.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    ok = bool(verdict) and all(a and b and c for *_, a, b, c in verdict)
    print(f"all_pass={ok} wrote {args.out}")


if __name__ == "__main__":
    main()
