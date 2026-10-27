"""L3 for need dimensions v2 (W-16c, docs/dimensions-v2.md §5): v1 vs v2 on one truth.

World (simulation/world_v2.py): all-fields catalogue (12 business-type + 6 far-field
families x 10) and all-fields population (W-63, W-67), converted to v2; truth =
51 v2 dims + 3 residual dims that no view can measure. Familiarity reports with
10% misses / 5% false alarms; 500 students per condition; 2 generators x 2 truths.
Every method runs the full pipeline (C1.2 v1 prior with anchors -> shortlist 25 ->
familiarity check -> sections) on its own view:
    v1           15 v1 items (v1 projection of the truth), v1 catalogue view
    v2-full      51 v2 items, v2 view                      (upper reference)
    v2-2stage(m) two-stage questionnaire: 10 non-domain items, 12 group items, then
                 every area of the top-m groups; unrated areas get the hierarchy prior
Scorers: DEFAULT_SCORER for v1, DEFAULT_SCORER_V2 (provisional equal layer weights) for v2.
Metrics against the FULL truth; `domain_mismatch_rate` = share of the discoveries that
are relevant under the v1-view truth but not under the full truth (B-C2-8 proxy).

PRE-REGISTERED (written and committed before the first run, 2026-10-27):
  A (select m, archetype generator): candidates m in {1, 2, 3}. m is eligible if
    v2-2stage(m) - v2-full in novel-relevant rate is never significantly negative
    (> -2 SE) in any condition and any supported transition stratum (n >= 30).
    Choose the smallest eligible m. If none is eligible, report it and use v2-full
    as the v2 representative in H-V1 and H-V2.
  B (validate, continuous generator): the chosen m must also be eligible there.
  H-V1: the chosen v2 method has a lower domain-mismatch rate than v1 (paired
    difference < -2 SE) in all 4 conditions.
  H-V2: the chosen v2 method's novel-relevant rate is not significantly below v1
    (> -2 SE) in any condition, in any supported transition stratum or need group.
  Report only: relevant rate, best-fit retention, items asked, by stratum.
  Replication: rerun with --catalog-seed 800 --pop-seed 801 --seed 802; A, B,
    H-V1 and H-V2 must all hold in both runs to be reported as confirmed.
  Disclosure: a 30-student smoke run on seeds 0/61/62 was made only to check that
  the script runs; its output was seen, so those seeds are not used. Run 1 uses
  900/901/902.

    python scripts/run_l3_dims_v2.py
    python scripts/run_l3_dims_v2.py --catalog-seed 800 --pop-seed 801 --seed 802 \\
        --out results/synthetic-l3-dims-v2-confirm.md
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.dimensions import NEED_KEYS, NEED_KEYS_V2
from aisc.pipeline import interactive_discovery
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.fields import all_fields_catalog, macro_group
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.respondent import make_ask
from aisc.simulation.strata import ALL, format_cell, paired_stratified, stratified
from aisc.simulation.truth import best_fit_id, relevant_ids
from aisc.simulation.world_v2 import stated, to_v1, two_stage_profile, v2_catalog, v2_students

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
MS = (1, 2, 3)
METHODS = ("v1", "v2-full", *[f"v2-2stage({m})" for m in MS])
METRICS = ("novel_relevant_rate", "relevant_rate", "best_fit_retained",
           "domain_mismatch_rate", "items")  # fmt: skip
TRANS = ("continuing", "adjacent", "far", "multi_degree")
MISS, FALSE_ALARM = 0.10, 0.05
NOISE_SD, BIAS = 0.15, 0.05
K = 10


def run_condition(world, gen, truth, n, pop_seed, seed):
    cat_v1src, hidden, cat_v2, cat_v1, residual = world
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(cat_v1src, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    students = v2_students(pop, np.random.default_rng([pop_seed, 1]))
    rows = {m: {k: [] for k in METRICS} for m in METHODS}
    trans, groups = [], []
    for idx, (st, w2, r) in enumerate(students):
        trans.append(st.transition)
        groups.append(macro_group(st.natural_family))
        true2 = dict(zip(NEED_KEYS_V2, map(float, w2), strict=True))
        w1 = np.array([to_v1(true2)[k] for k in NEED_KEYS])
        rel = relevant_ids(w2, cat_v2, truth, 0.1, residual, r)
        rel_v1view = relevant_ids(w1, cat_v1, truth, 0.1, residual, r)
        best = best_fit_id(w2, cat_v2, set(st.anchors), truth, residual, r)
        for code, method in enumerate(METHODS):
            rng = np.random.default_rng([seed, idx, code])
            if method == "v1":
                needs = stated(w1, NEED_KEYS, rng, NOISE_SD, BIAS)
                prof = StudentProfile(st.student_id, needs, {k: 0.3 for k in needs},
                                      anchors=list(st.anchors))  # fmt: skip
                catalog, items = cat_v1, len(NEED_KEYS)
            elif method == "v2-full":
                needs = stated(w2, NEED_KEYS_V2, rng, NOISE_SD, BIAS)
                prof = StudentProfile(st.student_id, needs, {k: 0.3 for k in needs},
                                      anchors=list(st.anchors), dims="v2")  # fmt: skip
                catalog, items = cat_v2, len(NEED_KEYS_V2)
            else:
                m = int(method.split("(")[1].rstrip(")"))
                prof, items = two_stage_profile(st, w2, m, rng, NOISE_SD, BIAS)
                catalog = cat_v2
            ask = make_ask(st, np.random.default_rng([seed, idx]), MISS, FALSE_ALARM)
            res = interactive_discovery(prof, catalog, ask, n_shortlist=25, k_new=K, k_known=5)
            ids = [x.program.program_id for x in res.discoveries]
            q = M.list_quality(ids, rel, set(st.familiar))
            rows[method]["novel_relevant_rate"].append(q["novel_relevant_rate"])
            rows[method]["relevant_rate"].append(q["relevant_rate"])
            rows[method]["best_fit_retained"].append(M.best_fit_retained(res.all_ids, best))
            rows[method]["domain_mismatch_rate"].append(
                float("nan") if not ids
                else sum(i in rel_v1view and i not in rel for i in ids) / len(ids)
            )  # fmt: skip
            rows[method]["items"].append(float(items))
    return rows, trans, groups


def summarise(rows, trans, groups):
    out = {"level_t": {}, "level_g": {}, "diff_full": {}, "diff_v1": {}, "diff_v1_g": {}}
    for m in METHODS:
        out["level_t"][m] = {k: stratified(rows[m][k], trans) for k in METRICS}
        out["level_g"][m] = {k: stratified(rows[m][k], groups) for k in METRICS}
        out["diff_full"][m] = {k: paired_stratified(rows[m][k], rows["v2-full"][k], trans)
                               for k in METRICS}  # fmt: skip
        out["diff_v1"][m] = {k: paired_stratified(rows[m][k], rows["v1"][k], trans)
                             for k in METRICS}  # fmt: skip
        out["diff_v1_g"][m] = {k: paired_stratified(rows[m][k], rows["v1"][k], groups)
                               for k in METRICS}  # fmt: skip
    return out


def eligible(res_gen, m):
    meth = f"v2-2stage({m})"
    return not any(
        st.significantly_below()
        for s in res_gen.values()
        for name, st in s["diff_full"][meth]["novel_relevant_rate"].items()
        if name != ALL
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=500)
    ap.add_argument("--catalog-seed", type=int, default=900)
    ap.add_argument("--pop-seed", type=int, default=901)
    ap.add_argument("--seed", type=int, default=902)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-dims-v2.md")
    args = ap.parse_args()
    rng = np.random.default_rng(args.catalog_seed)
    src, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=10), 10)
    cat_v2, cat_v1, residual = v2_catalog(src, hidden, rng)
    world = (src, hidden, cat_v2, cat_v1, residual)

    res: dict[str, dict] = {g: {} for g in GENERATORS}
    for gen, truth in product(GENERATORS, TRUTHS):
        rows, trans, groups = run_condition(world, gen, truth, args.students,
                                            args.pop_seed, args.seed)  # fmt: skip
        res[gen][truth] = summarise(rows, trans, groups)
        print(f"  {gen} {truth}", flush=True)

    ok_a = [m for m in MS if eligible(res["archetype"], m)]
    chosen_m = min(ok_a) if ok_a else None
    chosen = f"v2-2stage({chosen_m})" if chosen_m else "v2-full"
    ok_b = chosen_m is not None and eligible(res["continuous"], chosen_m)
    every = [(g, t) for g in GENERATORS for t in TRUTHS]
    hv1_fail = [c for c in every
                if not res[c[0]][c[1]]["diff_v1"][chosen]["domain_mismatch_rate"][ALL]
                .significantly_below()]  # fmt: skip
    hv2_fail = [(c, key, s) for c in every for key in ("diff_v1", "diff_v1_g")
                for s, st in res[c[0]][c[1]][key][chosen]["novel_relevant_rate"].items()
                if s != ALL and st.significantly_below()]  # fmt: skip

    cmd = (f"python scripts/run_l3_dims_v2.py --catalog-seed {args.catalog_seed} "
           f"--pop-seed {args.pop_seed} --seed {args.seed}")  # fmt: skip
    yn = {True: "✅", False: "❌"}
    L = [
        "# L3：需求維度 v1 vs v2（W-16c）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`{cmd}`",
        f"- 目錄 {len(cat_v2)} 個學程；每條件 {args.students} 位學生",
        "- 真相 = v2 51 維 + 3 個殘差維度；熟悉度回報 10%／5%",
        "- 方括號 = 人數 < 30，不納入判定；預先登記見 `scripts/run_l3_dims_v2.py` 檔頭",
        "",
        "## 判定",
        "",
        f"- A（原型模型挑 m）：合格 {ok_a or '無'} → 選出 **{chosen}**",
        f"- B（連續模型驗證）：{yn[ok_b]}",
        f"- H-V1（領域不符率 v2 < v1，4 條件）：{yn[not hv1_fail]}"
        + (f"（未成立：{hv1_fail}）" if hv1_fail else ""),
        f"- H-V2（相關且不知道 v2 不劣於 v1，所有層）：{yn[not hv2_fail]}"
        + (f"（未成立：{hv2_fail}）" if hv2_fail else ""),
        "",
    ]
    for gen in GENERATORS:
        for truth in TRUTHS:
            s = res[gen][truth]
            L += [f"## {gen} / {truth}", ""]
            L += ["### 水準（依轉換類型）", "",
                  "| 方法 | 指標 | all | " + " | ".join(TRANS) + " |",
                  "|---" * (len(TRANS) + 3) + "|"]  # fmt: skip
            for m in METHODS:
                for k in METRICS:
                    st = s["level_t"][m][k]
                    cells = " | ".join(
                        format_cell(st[t]) if t in st else "—" for t in (ALL, *TRANS)
                    )
                    L.append(f"| {m} | {k} | {cells} |")
            L += ["", "### 與 v1 的配對差（依轉換類型）", "",
                  "| 方法 | 指標 | all | " + " | ".join(TRANS) + " |",
                  "|---" * (len(TRANS) + 3) + "|"]  # fmt: skip
            for m in METHODS[1:]:
                for k in METRICS[:4]:
                    st = s["diff_v1"][m][k]
                    cells = " | ".join(format_cell(st[t], signed=True) if t in st else "—"
                                       for t in (ALL, *TRANS))  # fmt: skip
                    L.append(f"| {m} | {k} | {cells} |")
            gnames = sorted(k for k in s["diff_v1_g"][chosen]["novel_relevant_rate"] if k != ALL)
            L += [
                "",
                f"### {chosen} 與 v1 的配對差（依需求大領域群）",
                "",
                "| 指標 | " + " | ".join(gnames) + " |",
                "|---" * (len(gnames) + 1) + "|",
            ]
            for k in METRICS[:4]:
                st = s["diff_v1_g"][chosen][k]
                L.append(f"| {k} | " + " | ".join(format_cell(st[g], signed=True) for g in gnames)
                         + " |")  # fmt: skip
            L += [
                "",
                "### 兩階段 − 全部 51 題（相關且不知道，依轉換類型）",
                "",
                "| m | all | " + " | ".join(TRANS) + " |",
                "|---" * (len(TRANS) + 2) + "|",
            ]
            for m in MS:
                st = s["diff_full"][f"v2-2stage({m})"]["novel_relevant_rate"]
                cells = " | ".join(format_cell(st[t], signed=True) if t in st else "—"
                                   for t in (ALL, *TRANS))  # fmt: skip
                L.append(f"| {m} | {cells} |")
            L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"chosen={chosen} A={ok_a} B={ok_b} H-V1={not hv1_fail} H-V2={not hv2_fail}")


if __name__ == "__main__":
    main()
