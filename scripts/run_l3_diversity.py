"""L3 for CIP-family diversity of the shortlist and discoveries (B-C2-9; THEORY
"短名單族群多樣化"; A-C2-15, A-C2-16).

World: the all-fields world of run_l3_fields.py (12 business-type + 6 far-field
families, 6 hidden domains, transition strata). Catalogue size is a condition:
10 or 30 programmes per family (B-C2-9 is about large catalogues). Per condition:
2 truths x familiarity noise {perfect, 10/5} x catalogue size {10, 30} = 8 conditions,
500 students each. Recommenders: the C2 v1 pipeline (shortlist 25 -> familiarity
check -> 10 discoveries) with diversity lambda in {0 (current), 0.005, 0.01, 0.02,
0.04}; the grid was set from the default scorer's score scale only (median gap
between the 1st and 10th programme 0.025, 1st and 25th 0.045), before any outcome.
Paired differences candidate - lambda 0 on the same students and respondent seeds.

Metrics
  best_family_hit      the CIP family (4 digits) of the student's true best programme
                       (full truth, anchors excluded) appears among the discoveries
  novel_relevant_rate  as in every C2 L3 (guard; relevance by the full truth)
  report only          relevant_rate, best_fit_retained, hidden_mismatch_rate,
                       number of families among the discoveries

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  A candidate PASSES a condition if
    B    best_family_hit difference (all students) > 2 SE;
    G    novel_relevant_rate difference has mean > -0.02 for all students and in every
         transition stratum with n >= 30;
    D12  no transition stratum (n >= 30) has a best_family_hit difference more than
         2 SE below 0.
  Selection (generator "archetype", 8 conditions): among the candidates passing every
  condition, the largest mean best_family_hit gain averaged over conditions; ties ->
  smaller lambda. None -> stop, lambda stays 0.
  Validation: the selected lambda must pass all 8 conditions with generator
  "continuous".
  Confirmation: rerun with fresh seeds (catalogue 700, population 701, respondent
  702) under both generators; the selected lambda must pass all 16 conditions.
  If all three hold, the selected lambda becomes the default diversity of the C2
  pipeline; otherwise the default stays 0 and diversity remains an option.
  Disclosure: before writing this, an exploratory run (200 students, distance truth,
  perfect familiarity, lambda 0 only) showed that with 30 programmes per family the
  discoveries come from ~1.9 (archetype) / 3.3 (continuous) families and the true best
  family is missing for 6.5% / 25.5% of students (2.5% / 19.0% with 10 per family).
  The finished script was smoke-tested with 3 students per condition (output to a
  scratch file, checked only for exit status) before the commit.

    python scripts/run_l3_diversity.py
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.discovery.sectioned import cip_family
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import c2_v1_pipeline, evaluate_lists_per_student
from aisc.simulation.fields import all_fields_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import ALL, paired_stratified
from aisc.simulation.truth import best_fit_id

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
FAM_NOISE = {"perfect": (0.0, 0.0), "10/5": (0.10, 0.05)}
SIZES = (10, 30)
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.04)
CANDS = LAMBDAS[1:]
K = 10
MARGIN = 0.02
SEEDS = {"main": (0, 51, 52), "fresh": (700, 701, 702)}


def recorder(rec, store):
    def run(profile, catalog, k, ask):
        res = rec(profile, catalog, k, ask)
        store.append([r.program.program_id for r in res.discoveries])
        return res

    run.interactive = True  # type: ignore[attr-defined]
    return run


def run_condition(gen, truth, noise, size, seeds, n):
    cat_seed, pop_seed, seed = seeds
    rng = np.random.default_rng([cat_seed, size])
    catalog, hidden = all_fields_catalog(rng, synthetic_catalog(rng, n_per_family=size), size)
    fam = {p.program_id: cip_family(p) for p in catalog}
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen,
                    field_model="all_fields")  # fmt: skip
    pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed), hidden=hidden)
    lists = {lam: [] for lam in LAMBDAS}
    recs = {lam: recorder(c2_v1_pipeline(diversity=lam), lists[lam]) for lam in LAMBDAS}
    per = evaluate_lists_per_student(
        pop, catalog, recs, k=K, truth=truth,
        familiarity_noise={lam: FAM_NOISE[noise] for lam in LAMBDAS}, seed=seed, hidden=hidden,
    )  # fmt: skip
    best = [fam[best_fit_id(s.true_needs, catalog, set(s.anchors), truth, hidden, s.hidden_needs)]
            for s in pop]  # fmt: skip
    for lam in LAMBDAS:
        per[lam]["best_family_hit"] = [
            float(b in {fam[i] for i in ids}) for b, ids in zip(best, lists[lam], strict=True)
        ]
        per[lam]["n_families"] = [float(len({fam[i] for i in ids})) for ids in lists[lam]]
    trans = [s.transition for s in pop]
    out = {"level": {lam: {m: float(np.nanmean(v)) for m, v in per[lam].items()}
                     for lam in LAMBDAS}, "diff": {}}  # fmt: skip
    for lam in CANDS:
        out["diff"][lam] = {m: paired_stratified(per[lam][m], per[0.0][m], trans)
                            for m in per[lam]}  # fmt: skip
    return out


def passes(diff) -> dict[str, bool]:
    b = diff["best_family_hit"][ALL]
    g = all(st.mean > -MARGIN for s, st in diff["novel_relevant_rate"].items()
            if s == ALL or st.supported)  # fmt: skip
    d12 = not any(st.significantly_below() for s, st in diff["best_family_hit"].items()
                  if s != ALL)  # fmt: skip
    return {"B": b.mean > 2 * b.se, "G": g, "D12": d12}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-diversity.md")
    args = ap.parse_args()
    res = {}
    for run_name, gen, truth, noise, size in product(SEEDS, GENERATORS, TRUTHS, FAM_NOISE, SIZES):
        res[(run_name, gen, truth, noise, size)] = run_condition(
            gen, truth, noise, size, SEEDS[run_name], args.students
        )
        print(run_name, gen, truth, noise, size, flush=True)

    def conds(run_name, gen):
        return [c for c in res if c[0] == run_name and c[1] == gen]

    def all_pass(lam, cs):
        return all(all(passes(res[c]["diff"][lam]).values()) for c in cs)

    sel_conds = conds("main", "archetype")
    ok = [lam for lam in CANDS if all_pass(lam, sel_conds)]
    gain = {lam: float(np.mean([res[c]["diff"][lam]["best_family_hit"][ALL].mean
                                for c in sel_conds])) for lam in CANDS}  # fmt: skip
    selected = max(ok, key=lambda lam: (round(gain[lam], 9), -lam)) if ok else None
    valid = selected is not None and all_pass(selected, conds("main", "continuous"))
    confirm = valid and all(all_pass(selected, conds("fresh", g)) for g in GENERATORS)
    adopted = selected if confirm else 0.0

    L = [
        "# L3：短名單與「可能沒想過」區的族群多樣化（B-C2-9）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_diversity.py`",
        f"- 全領域世界（隱藏領域、轉換類型）；每條件 {args.students} 位學生；"
        f"學程數每族群 {SIZES}；λ 候選 {CANDS}",
        "- 預先登記見腳本檔頭；差 = 候選 − λ 0（同一批學生與作答種子）",
        "",
        "## 判定",
        "",
        f"- 選擇（archetype）：通過全部 8 個條件的候選 {list(ok) or '無'}；"
        f"選出 λ = {selected if selected is not None else '—'}",
        f"- 驗證（continuous）：{'✅' if valid else '❌'}；新種子確認（兩種生成方式）："
        f"{'✅' if confirm else '❌'} → **預設 λ = {adopted}**",
        "",
        "## 各條件的判定與差（平均 ± 標準誤）",
        "",
        "B = 最佳族群出現率顯著提高；G = 新且相關比例下降不超過 0.02（全體與各轉換層）；"
        "D12 = 沒有轉換層的最佳族群出現率顯著下降。",
        "",
        "| 執行 | 生成 | 真相 | 熟悉度 | 每族群 | λ | 最佳族群出現 差 | 新且相關 差 "
        "| 相關 差 | 最佳學程保留 差 | 隱藏不符 差 | 族群數（λ / 0） | B | G | D12 |",
        "|---" * 15 + "|",
    ]

    def f(st):
        return f"{st.mean:+.3f}±{st.se:.3f}"

    for c, r in res.items():
        for lam in CANDS:
            d = r["diff"][lam]
            p = passes(d)
            L.append(
                f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {lam} | "
                f"{f(d['best_family_hit'][ALL])} | {f(d['novel_relevant_rate'][ALL])} | "
                f"{f(d['relevant_rate'][ALL])} | {f(d['best_fit_retained'][ALL])} | "
                f"{f(d['hidden_mismatch_rate'][ALL])} | "
                f"{r['level'][lam]['n_families']:.2f} / {r['level'][0.0]['n_families']:.2f} | "
                + " | ".join("✅" if p[k] else "❌" for k in ("B", "G", "D12"))
                + " |"
            )
    L += ["", "## 水準（λ = 0，只報告）", "",
          "| 執行 | 生成 | 真相 | 熟悉度 | 每族群 | 最佳族群出現 | 新且相關 | 相關 | 族群數 |",
          "|---" * 9 + "|"]  # fmt: skip
    for c, r in res.items():
        lv = r["level"][0.0]
        L.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {lv['best_family_hit']:.3f} | "
                 f"{lv['novel_relevant_rate']:.3f} | {lv['relevant_rate']:.3f} | "
                 f"{lv['n_families']:.2f} |")  # fmt: skip
    L += ["", "## 最差轉換層（選出的 λ；最佳族群出現差、新且相關差）", ""]
    if selected is not None:
        L += ["| 執行 | 生成 | 真相 | 熟悉度 | 每族群 | 層 | n | 最佳族群出現 差 | 新且相關 差 |",
              "|---" * 9 + "|"]  # fmt: skip
        for c, r in res.items():
            d = r["diff"][selected]
            for s, st in d["best_family_hit"].items():
                if s == ALL or not st.supported:
                    continue
                L.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {s} | {st.n} | "
                         f"{f(st)} | {f(d['novel_relevant_rate'][s])} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"ok={ok} selected={selected} valid={valid} confirm={confirm} adopted={adopted}")


if __name__ == "__main__":
    main()
