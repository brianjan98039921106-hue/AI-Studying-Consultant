"""EXPLORATORY (not pre-registered): why is v2 below v1 in the W-16c L3?

The pre-registered L3 (scripts/run_l3_dims_v2.py, F70-F73) found v2 below v1 on
novel-relevant rate, even with all 51 items answered. This script separates three
candidate causes on seeds that no registered run uses (4242/4243/4244):

    scorer     v2 with the provisional equal-layer scorer (DEFAULT_SCORER_V2) vs the
               unlayered blend(0.25) vs layer weights proportional to dimension count
    noise      every method also run with noise-free, unbiased answers ("-nf")
    truth      the registered truth weights every one of the 54 dims equally, so the
               41 domain dims are 76% of it; an unlayered scorer mirrors that. Each
               list is also scored against a layer-equal truth (each of interest,
               intensity, domain, orientation and residual = 1/5)

Same pipeline settings as the registered run. History: a first, simplified version
(default pipeline settings, means only) was run once on these seeds and motivated
this script; its output is in results/exploratory-dims-v2-scorer.md §0. Nothing here
may be reported as confirmed; it only shapes the next pre-registered run.

    python scripts/explore_dims_v2_scorer.py
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import LAYERS, NEED_KEYS, NEED_KEYS_V2, keys_in_layer
from aisc.discovery import DEFAULT_SCORER, DEFAULT_SCORER_V2, blend, layer_balanced
from aisc.pipeline import interactive_discovery
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.respondent import make_ask
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified
from aisc.simulation.truth import relevant_ids
from aisc.simulation.world_v2 import N_RESIDUAL, stated, to_v1, v2_catalog, v2_students

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
KINDS = ("distance", "cosine")
WEIGHTINGS = ("dim", "layer")
MISS, FALSE_ALARM = 0.10, 0.05
NOISE_SD, BIAS = 0.15, 0.05
DIM_PROP = layer_balanced({lay: len(keys_in_layer(lay, "v2")) for lay in LAYERS})
# name -> (view, noisy answers, scorer)
METHODS = {
    "v1": ("v1", True, DEFAULT_SCORER),
    "v1-nf": ("v1", False, DEFAULT_SCORER),
    "v2-bal": ("v2", True, DEFAULT_SCORER_V2),
    "v2-unbal": ("v2", True, blend(0.25)),
    "v2-dimprop": ("v2", True, DIM_PROP),
    "v2-bal-nf": ("v2", False, DEFAULT_SCORER_V2),
    "v2-unbal-nf": ("v2", False, blend(0.25)),
}


def layer_scale() -> np.ndarray:
    """Per-coordinate scale (v2 dims + residual) giving each layer equal weight."""
    s = [1 / np.sqrt(len(keys_in_layer(lay, "v2")))
         for k in NEED_KEYS_V2 for lay in LAYERS if k in keys_in_layer(lay, "v2")]  # fmt: skip
    return np.array(s + [1 / np.sqrt(N_RESIDUAL)] * N_RESIDUAL)


def layer_relevant(w2, r, cat_v2, residual, kind, scale, top_frac=0.1) -> set[str]:
    """Top programs under the layer-equal truth (rank-equivalent to scaling coordinates)."""
    w = np.concatenate([w2, r]) * scale
    X = np.array([np.concatenate([p.vector, residual[p.program_id]]) for p in cat_v2]) * scale
    if kind == "distance":
        u = -np.linalg.norm(X - w, axis=1)
    else:
        u = X @ w / (np.linalg.norm(X, axis=1) * np.linalg.norm(w))
    n = max(1, int(np.ceil(top_frac * len(cat_v2))))
    return {cat_v2[i].program_id for i in np.argsort(-u, kind="stable")[:n]}


def run(world, gen, kind, n, pop_seed, seed):
    src, hidden, cat_v2, cat_v1, residual = world
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=kind, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(src, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    students = v2_students(pop, np.random.default_rng([pop_seed, 1]))
    scale = layer_scale()
    rows = {w: {m: [] for m in METHODS} for w in WEIGHTINGS}
    groups = []
    for idx, (st, w2, r) in enumerate(students):
        groups.append(macro_group(st.natural_family))
        true2 = dict(zip(NEED_KEYS_V2, map(float, w2), strict=True))
        w1 = np.array([to_v1(true2)[k] for k in NEED_KEYS])
        rel = {"dim": relevant_ids(w2, cat_v2, kind, 0.1, residual, r),
               "layer": layer_relevant(w2, r, cat_v2, residual, kind, scale)}  # fmt: skip
        for code, (name, (view, noisy, scorer)) in enumerate(METHODS.items()):
            rng = np.random.default_rng([seed, idx, code])
            keys, w, cat = (NEED_KEYS, w1, cat_v1) if view == "v1" else (NEED_KEYS_V2, w2, cat_v2)
            needs = stated(w, keys, rng, NOISE_SD if noisy else 0.0, BIAS if noisy else 0.0)
            prof = StudentProfile(st.student_id, needs, {k: 0.3 for k in needs},
                                  anchors=list(st.anchors), dims=view)  # fmt: skip
            ask = make_ask(st, np.random.default_rng([seed, idx]), MISS, FALSE_ALARM)
            res = interactive_discovery(prof, cat, ask, n_shortlist=25, k_new=10, k_known=5,
                                        scorer=scorer)  # fmt: skip
            ids = [x.program.program_id for x in res.discoveries]
            for wt in WEIGHTINGS:
                q = M.list_quality(ids, rel[wt], set(st.familiar))
                rows[wt][name].append(q["novel_relevant_rate"])
    return rows, groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=300)
    ap.add_argument("--catalog-seed", type=int, default=4242)
    ap.add_argument("--pop-seed", type=int, default=4243)
    ap.add_argument("--seed", type=int, default=4244)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "exploratory-dims-v2-scorer.md")  # fmt: skip
    args = ap.parse_args()
    rng = np.random.default_rng(args.catalog_seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, cat_v1, residual = v2_catalog(src, hidden, rng)
    world = (src, hidden, cat_v2, cat_v1, residual)

    cmd = (f"python scripts/explore_dims_v2_scorer.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}")  # fmt: skip
    L = [
        "# 探索性診斷：v2 為何低於 v1（W-16c，**未預先登記**）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(cat_v2)} 個學程；每條件 {args.students} 位學生；"
        "管線設定與預先登記的 run 相同",
        "- 指標：相關且不知道（novel-relevant）；配對差 = 方法 − v1（同一批學生）",
        "- 真相加權：dim = 54 維等權（預先登記 run 所用；領域層佔 41/54）；"
        "layer = 興趣、強度、領域、取向、殘差 5 層等權",
        "- 方法：-nf = 無作答雜訊、無偏誤；bal = 現行 DEFAULT_SCORER_V2（四層等權）；"
        "unbal = 不分層 blend(0.25)；dimprop = 分層、層權重 ∝ 維度數",
        "- 只用來設計下一次預先登記；不可當成已確認結論",
        "",
        "## 0. 第一版簡化診斷（預設管線設定、只有平均；同一組種子）",
        "",
        "| 生成 / 真相 | v1 | v1-nf | v2-bal | v2-unbal | v2-dimprop | v2-bal-nf | v2-unbal-nf |",
        "|---|---|---|---|---|---|---|---|",
        "| archetype / distance | 0.652 | 0.713 | 0.551 | 0.739 | 0.674 | 0.646 | 0.784 |",
        "| archetype / cosine | 0.640 | 0.704 | 0.571 | 0.747 | 0.667 | 0.652 | 0.795 |",
        "| continuous / distance | 0.460 | 0.545 | 0.379 | 0.512 | 0.373 | 0.456 | 0.626 |",
        "| continuous / cosine | 0.470 | 0.566 | 0.359 | 0.519 | 0.322 | 0.438 | 0.643 |",
        "",
    ]
    runs = {}
    for gen, kind in product(GENERATORS, KINDS):
        runs[(gen, kind)] = run(world, gen, kind, args.students, args.pop_seed, args.seed)
        print(f"  {gen} {kind}", flush=True)
    others = [m for m in METHODS if m != "v1"]
    for wt in WEIGHTINGS:
        L += [f"## 真相加權 = {wt}", "",
              "### 水準與配對差（全體）", "",
              "| 生成 / 真相 | v1 水準 | " + " | ".join(f"{m} − v1" for m in others) + " |",
              "|---|---|" + "---|" * len(others)]  # fmt: skip
        L_g = [f"### v2-unbal − v1，依需求大領域群（{wt}）", ""]
        for gen, kind in product(GENERATORS, KINDS):
            rows, groups = runs[(gen, kind)]
            r = rows[wt]
            lvl = stratified(r["v1"], groups)[ALL]
            cells = [format_cell(paired_stratified(r[m], r["v1"], groups)[ALL], signed=True)
                     for m in others]  # fmt: skip
            L.append(f"| {gen} / {kind} | {format_cell(lvl)} | " + " | ".join(cells) + " |")
            by_g = paired_stratified(r["v2-unbal"], r["v1"], groups)
            names = [g for g in by_g if g != ALL]
            if len(L_g) == 2:
                L_g += ["| 生成 / 真相 | " + " | ".join(names) + " |",
                        "|---|" + "---|" * len(names)]  # fmt: skip
            L_g.append(
                f"| {gen} / {kind} | "
                + " | ".join(format_cell(by_g[g], signed=True) for g in names)
                + " |"
            )
        L += ["", *L_g, ""]
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
