"""All-fields L3 (W-63, W-67): far-field students, hidden domains, field switchers.

World: 12 business-type families + 6 far-field families (dentistry, biology,
materials, letters, education, law), 10 programs each. True needs and program
attributes include 6 hidden domains that no component can see (A-C5-10); stated
needs cover the 15 measured dimensions only. Transition types: continuing,
adjacent, far, multi_degree (mix 0.40 / 0.25 / 0.25 / 0.10, A-C5-11).
Per student: complete stated needs -> C2 v1 sections vs fit-only (same profile).
Conditions per generator: 2 truths x familiarity noise {perfect, 10/5} = 4;
500 students per condition. Relevance and best fit use the FULL truth.

PRE-REGISTERED (written before the first run, 2026-10-27):
  H-F1 (D12): in every condition and every supported transition stratum
       (n >= 30), sections - fit-only in novel-relevant rate is not
       significantly negative (> -2 SE).
  H-F2 (B-C2-8 materialises): in every condition, fit-only's hidden-mismatch
       rate (share of the top-10 that is relevant on the measured dimensions but
       not on the full truth) is significantly above 0 (mean > 2 SE).
  Report only: all metrics by transition stratum and by the macro group the
       student's true needs point to; worst stratum per metric.
  Replication: rerun with --catalog-seed 700 --pop-seed 701 --seed 702; both
       hypotheses must hold in both runs to be reported as confirmed.

    python scripts/run_l3_fields.py
    python scripts/run_l3_fields.py --catalog-seed 700 --pop-seed 701 --seed 702 \\
        --out results/synthetic-l3-fields-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.discovery import shortlist
from aisc.profiling import complete_neutral
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import c2_v1_pipeline, evaluate_lists_per_student
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import (
    ALL,
    format_cell,
    paired_stratified,
    stratified,
    worst_stratum,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
FAM_NOISE = {"perfect": (0.0, 0.0), "10/5": (0.10, 0.05)}
METRICS = ("novel_relevant_rate", "relevant_rate", "best_fit_retained", "hidden_mismatch_rate")
TRANS = ("continuing", "adjacent", "far", "multi_degree")
GROUPS = ("business_data", "engineering_design", "health_life", "humanities_social")
K = 10


def fit_only(profile, catalog, k):
    return shortlist(complete_neutral(profile), catalog, n=k)


def run_condition(catalog, hidden, gen, truth, noise, n, pop_seed, seed):
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    by_trans = [st.transition for st in pop]
    by_group = [macro_group(st.natural_family) for st in pop]
    recs = {"sections": c2_v1_pipeline(), "fit-only": fit_only}
    per = evaluate_lists_per_student(
        pop, catalog, recs, k=K, truth=truth,
        familiarity_noise={"sections": FAM_NOISE[noise]}, seed=seed, hidden=hidden,
    )  # fmt: skip
    out = {"trans": {}, "group": {}, "diff": {}}
    for m in METRICS:
        out["trans"][m] = {r: stratified(per[r][m], by_trans) for r in recs}
        out["group"][m] = {r: stratified(per[r][m], by_group) for r in recs}
        out["diff"][m] = paired_stratified(per["sections"][m], per["fit-only"][m], by_trans)
    return out


def _row(stats, keys, signed=False):
    return " | ".join(format_cell(stats[s], signed=signed) if s in stats else "—" for s in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=500)
    ap.add_argument("--catalog-seed", type=int, default=0)
    ap.add_argument("--pop-seed", type=int, default=51)
    ap.add_argument("--seed", type=int, default=52)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-fields.md")
    args = ap.parse_args()
    rng = np.random.default_rng(args.catalog_seed)
    catalog, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)

    res = {}
    for gen, truth, noise in product(GENERATORS, TRUTHS, FAM_NOISE):
        res[(gen, truth, noise)] = run_condition(
            catalog, hidden, gen, truth, noise, args.students, args.pop_seed, args.seed
        )
        print(f"  {gen} {truth} {noise}", flush=True)

    f1 = [(c, s) for c, r in res.items() for s, st in r["diff"]["novel_relevant_rate"].items()
          if s != ALL and st.significantly_below()]  # fmt: skip
    f2 = [c for c, r in res.items()
          if not (r["trans"]["hidden_mismatch_rate"]["fit-only"][ALL].mean
                  > 2 * r["trans"]["hidden_mismatch_rate"]["fit-only"][ALL].se)]  # fmt: skip
    cmd = (f"python scripts/run_l3_fields.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}")  # fmt: skip
    yn = lambda ok: "✅ 成立" if ok else "❌ 不成立"  # noqa: E731
    L = [
        "# L3 全領域：遠領域學生、隱藏領域與轉換者（W-63、W-67）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(catalog)} 個學程（12 商管型 + 6 遠領域族群 × 10）",
        f"- 每條件 {args.students} 位學生；真相含 6 個隱藏領域",
        "- 方括號 = 人數 < 30，不納入判定；預先登記見 `scripts/run_l3_fields.py` 檔頭",
        "",
        f"## 判定：H-F1 {yn(not f1)} · H-F2 {yn(not f2)}",
        "",
    ]
    L += [f"- H-F1 顯著較差：{c} 的 {s}" for c, s in f1]
    L += [f"- H-F2 未顯著大於 0：{c}" for c in f2]
    L += ["", "## 分區 − 只依契合（配對差，依轉換類型）", ""]
    for m in METRICS:
        L += [f"### {m}", "", "| 條件 | all | " + " | ".join(TRANS) + " | 最差層 |",
              "|---" * (len(TRANS) + 3) + "|"]  # fmt: skip
        for c, r in res.items():
            d = r["diff"][m]
            better = m != "hidden_mismatch_rate"
            L.append(f"| {' / '.join(c)} | {_row(d, (ALL, *TRANS), True)} | "
                     f"{worst_stratum(d, better)} |")  # fmt: skip
        L.append("")
    for title, key, cols in (
        ("依轉換類型", "trans", TRANS),
        ("依需求指向的大領域群", "group", GROUPS),
    ):
        L += [f"## 水準（{title}，只報告）", ""]
        for m in METRICS:
            L += [f"### {m}", "", "| 條件 | 方法 | all | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 3) + "|"]  # fmt: skip
            for c, r in res.items():
                for rec, st in r[key][m].items():
                    L.append(f"| {' / '.join(c)} | {rec} | {_row(st, (ALL, *cols))} |")
            L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    status = f"H-F1 {'holds' if not f1 else 'FAILS'}; H-F2 {'holds' if not f2 else 'FAILS'}"
    print(f"{status} -> {args.out}")


if __name__ == "__main__":
    main()
