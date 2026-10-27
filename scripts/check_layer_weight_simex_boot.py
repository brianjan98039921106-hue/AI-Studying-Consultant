"""Parameter recovery for SIMEX with a bootstrap-around-SIMEX interval (W-76, F88).

Same world, choice model and estimator settings as scripts/check_layer_weight_simex.py
(F88-F90): v2 world, N = 200 respondents x K = 20 uniformly drawn programme pairs,
choices from the pairwise logit with TRUE needs and known layer shares (residual
dims 10% of the weight, consistency 0.75); the estimator sees STATED needs (51 items,
response noise + 0.05 acquiescence, clipped). SIMEX point estimate: lambdas
(0.5, 1, 1.5, 2) x 20 simulations, quadratic extrapolation to -1, true sigma.

Change under test: the interval. F88 used the naive bootstrap interval shifted by
the correction (coverage 0.82 in one setting). Here: respondent bootstrap around
the WHOLE SIMEX procedure (100 resamples, 5 simulations per lambda inside each),
95% percentile interval.

PRE-REGISTERED (written and committed before the first run, 2026-10-27):
  Primary: noise 0.15, true sigma, every generator {archetype, continuous} x shares
  {dim, equal, mid}; 100 replications each (coverage SE about 0.02 near 0.95).
  R1 bias: |mean SIMEX estimate - true| <= 0.05 for every layer share.
  R2 coverage: the interval covers the true domain share in >= 85% of replications.
  R3 discrimination: the interval excludes the other hypothesis value in >= 80%.
  If R1-R3 hold: SIMEX with this interval is the W-76 analysis method, given sigma
  from the W-41 retest. Otherwise W-76 needs another approach before real data.
  Report only: noise 0.30 with true sigma (40 replications each).
  Seeds: --seed 1700 (none used before). Disclosure: a timing run on synthetic
  random data (no catalogue, no seeds from this script) was made; no smoke run of
  this script's output was looked at beyond checking that it runs (seed 13).

    python scripts/check_layer_weight_simex_boot.py
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from multiprocessing import Pool
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
N_BOOT, N_SIM_BOOT = 100, 5
_STATE: dict = {}


def scale_for(G: np.ndarray, wts: np.ndarray) -> float:
    lo, hi = 0.0, 1e5
    for _ in range(80):
        c = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(G @ (c * wts))))
        lo, hi = (c, hi) if np.mean(np.maximum(p, 1 - p)) < CONSISTENCY else (lo, c)
    return (lo + hi) / 2


def one_rep(X, Xres, students, shares, noise, rng):
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
    est = estimate_layer_weights_simex(np.array(st_rows), X, np.array(pairs), y, noise, "v2",
                                       n_sim=20, n_boot=N_BOOT, rng=rng,
                                       interval="bootstrap", n_sim_boot=N_SIM_BOOT)  # fmt: skip
    c = est.corrected
    return np.array([c.shares[lay] for lay in LAYERS]), c.ci["domain"]


def _init(seed: int) -> None:
    rng = np.random.default_rng(seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, _, residual = v2_catalog(src, hidden, rng)
    _STATE["X"] = np.array([p.vector for p in cat_v2])
    _STATE["Xres"] = np.array([residual[p.program_id] for p in cat_v2])
    for gen in ("archetype", "continuous"):
        cfg = SimConfig(n_students=400, prestige_weight=0.5, needs_model=gen,
                        field_model="all_fields")  # fmt: skip
        pop = generate_population(src, cfg, np.random.default_rng(seed + 1), hidden=hidden)
        _STATE[gen] = v2_students(pop, np.random.default_rng([seed + 1, 1]))


def _job(task):
    seed, idx, gen, name, noise, rep = task
    rng = np.random.default_rng([seed + 2, idx, rep])
    return idx, one_rep(_STATE["X"], _STATE["Xres"], _STATE[gen], SHARES[name], noise, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1700)
    ap.add_argument("--reps", type=int, default=100, help="replications, primary rows")
    ap.add_argument("--reps-robust", type=int, default=40, help="replications, noise 0.30")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "layer-weight-simex-boot.md")  # fmt: skip
    args = ap.parse_args()
    configs = [(gen, name, noise) for noise in (0.15, 0.30)
               for gen, name in product(("archetype", "continuous"), SHARES)]  # fmt: skip
    tasks = [(args.seed, i, gen, name, noise, rep)
             for i, (gen, name, noise) in enumerate(configs)
             for rep in range(args.reps if noise == 0.15 else args.reps_robust)]  # fmt: skip
    results: dict[int, list] = {i: [] for i in range(len(configs))}
    with Pool(args.workers, initializer=_init, initargs=(args.seed,)) as pool:
        for done, (i, out) in enumerate(pool.imap_unordered(_job, tasks, chunksize=1), 1):
            results[i].append(out)
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    rows = []
    for i, (gen, name, noise) in enumerate(configs):
        sh = SHARES[name]
        E = np.array([o[0] for o in results[i]])
        C = np.array([o[1] for o in results[i]])
        excl = {o: float(np.mean((C[:, 0] > v) | (C[:, 1] < v)))
                for o, v in (("dim", SHARES["dim"][DOMAIN]), ("equal", 0.25))}  # fmt: skip
        rows.append(dict(gen=gen, shares=name, noise=noise, reps=len(E),
                         bias=E.mean(axis=0) - sh,
                         cover=float(np.mean((C[:, 0] <= sh[DOMAIN]) & (sh[DOMAIN] <= C[:, 1]))),
                         width=float(np.mean(C[:, 1] - C[:, 0])), excl=excl))  # fmt: skip

    other = {"dim": ["equal"], "equal": ["dim"], "mid": ["dim", "equal"]}
    verdict = [(r["gen"], r["shares"], bool(np.all(np.abs(r["bias"]) <= 0.05)),
                r["cover"] >= 0.85, all(r["excl"][o] >= 0.80 for o in other[r["shares"]]),
                r["cover"])
               for r in rows if r["noise"] == 0.15]  # fmt: skip
    yn = {True: "✅", False: "❌"}
    L = [
        "# SIMEX 自助法區間的層權重參數還原（W-76）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令："
        f"`python scripts/check_layer_weight_simex_boot.py --seed {args.seed}`",
        f"- N = {N} 人、K = {K} 組；SIMEX 點估計 λ = 0.5, 1, 1.5, 2 各 20 次模擬；"
        f"區間 = 受訪者自助法 {N_BOOT} 次 × 每次完整 SIMEX（每個 λ {N_SIM_BOOT} 次模擬）",
        "- 預先登記見 `scripts/check_layer_weight_simex_boot.py` 檔頭",
        "",
        "## 判定（雜訊 0.15、正確 sigma）",
        "",
        "| 生成模型 | 真實份量 | R1 偏誤 ≤ 0.05 | R2 涵蓋率 ≥ 0.85 | R3 區分另一假設 ≥ 0.80 |",
        "|---|---|---|---|---|",
    ]
    L += [f"| {g} | {s} | {yn[a]} | {yn[b]}（{cv:.2f}） | {yn[c]} |"
          for g, s, a, b, c, cv in verdict]  # fmt: skip
    L += ["", "## 全部設計", "",
          "| 生成模型 | 份量 | 雜訊 | 重複 | SIMEX 偏誤（興趣、強度、領域、取向） | 領域涵蓋率 "
          "| 區間寬度 | 排除 dim 值 | 排除 0.25 |",
          "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        b = ", ".join(f"{v:+.3f}" for v in r["bias"])
        L.append(f"| {r['gen']} | {r['shares']} | {r['noise']} | {r['reps']} | {b} | "
                 f"{r['cover']:.2f} | {r['width']:.3f} | {r['excl']['dim']:.2f} | "
                 f"{r['excl']['equal']:.2f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    ok = bool(verdict) and all(a and b and c for _, _, a, b, c, _ in verdict)
    print(f"all_pass={ok} wrote {args.out}")


if __name__ == "__main__":
    main()
