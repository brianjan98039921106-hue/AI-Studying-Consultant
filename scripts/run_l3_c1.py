"""L3 for C1.2 anchor decomposition v1 (W-23) under partial elicitation.

Writes results/synthetic-l3-c1-v1.md. Deterministic (fixed seeds).

PRE-REGISTERED SELECTION RULE (written before the first run):
  1. Tune on the ARCHETYPE generator only. Among v1 variants, keep those with
     no condition significantly worse than the neutral baseline (paired τ
     difference < -2 SE). Pick the one with the largest mean paired τ
     difference; ties -> the simpler variant (earlier in VARIANTS).
  2. Validate the chosen variant on the CONTINUOUS generator (held out).
     Acceptance (ROADMAP W-23): no continuous condition significantly worse
     than neutral. τ is primary; RMSE is reported only (F9).
  Conditions: 2 truths x 3 prestige weights x 2 stated-noise levels x
  3 coverage levels (share of dimensions answered) = 36 per generator.

CONFIRMATION RUN (pre-registered 2026-10-25, after the first run failed held-out
validation): candidate CONFIRM_CANDIDATE below, chosen post hoc because it was
never significantly worse under either generator. It is accepted only if, on
fresh independent seeds (--catalog-seed 100 --pop-seed 101 --mask-seed 102),
it is significantly worse in 0/36 conditions under BOTH generators.

    python scripts/run_l3_c1.py [--students 250]
    python scripts/run_l3_c1.py --catalog-seed 100 --pop-seed 101 --mask-seed 102 \
        --out results/synthetic-l3-c1-v1-confirm.md
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.profiling import blend_profile, blend_profile_v1, complete_neutral
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import evaluate_profile_methods
from aisc.simulation.population import SimConfig, generate_population

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
PRESTIGE_WEIGHTS = (0.0, 0.5, 0.9)
NOISES = (0.15, 0.30)
COVERAGES = (1.0, 0.5, 0.25)
INF = float("inf")

# (sigma0, prestige_beta, answered_scale)
VARIANTS = [
    (0.6, 0.0, INF),
    (0.6, 3.0, INF),
    (0.3, 0.0, INF),
    (0.3, 3.0, INF),
    (0.6, 0.0, 3.0),
    (0.6, 3.0, 3.0),
    (0.3, 0.0, 3.0),
    (0.3, 3.0, 3.0),
]


CONFIRM_CANDIDATE = (0.6, 3.0, INF)


def vname(v) -> str:
    s0, beta, scale = v
    return f"v1[σ0={s0}, β={beta:g}, 已答×{'∞' if math.isinf(scale) else f'{scale:g}'}]"


def build_methods():
    methods = {
        "neutral": lambda p, c: complete_neutral(p),
        "v0（中性補全後合併）": lambda p, c: blend_profile(complete_neutral(p), c),
    }
    for v in VARIANTS:
        s0, beta, scale = v
        methods[vname(v)] = lambda p, c, s0=s0, beta=beta, scale=scale: blend_profile_v1(
            p, c, sigma0=s0, prestige_beta=beta, answered_scale=scale
        )
    return methods


def run_generator(gen: str, catalog, n: int, pop_seed: int = 1, mask_seed: int = 11):
    """Return {condition: results} for one generator."""
    out = {}
    methods = build_methods()
    for truth, pw, noise in product(TRUTHS, PRESTIGE_WEIGHTS, NOISES):
        cfg = SimConfig(
            n_students=n, prestige_weight=pw, truth=truth, noise_sd=noise, needs_model=gen
        )
        pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed))
        for cov in COVERAGES:
            out[(truth, pw, noise, cov)] = evaluate_profile_methods(
                pop,
                catalog,
                methods,
                baseline="neutral",
                truth=truth,
                coverage=cov,
                seed=mask_seed,
            )
        print(f"  {gen} truth={truth} π={pw} noise={noise}")
    return out


def summarise(results):
    """Per method: mean τ diff, min τ diff, #worse, #better, mean τ, mean RMSE."""
    names = list(next(iter(results.values())))
    rows = {}
    for m in names:
        diffs = [r[m]["tau_diff_vs_baseline"] for r in results.values()]
        worse = sum(d < -2 * se for d, se in diffs if not math.isnan(se))
        better = sum(d > 2 * se for d, se in diffs if not math.isnan(se))
        rows[m] = {
            "mean_diff": float(np.mean([d for d, _ in diffs])),
            "min_diff": float(np.min([d for d, _ in diffs])),
            "worse": worse,
            "better": better,
            "tau": float(np.mean([r[m]["kendall_tau"][0] for r in results.values()])),
            "rmse": float(np.mean([r[m]["rmse"][0] for r in results.values()])),
        }
    return rows


def select(summary) -> str:
    candidates = [vname(v) for v in VARIANTS if summary[vname(v)]["worse"] == 0]
    if not candidates:
        return ""
    return max(candidates, key=lambda m: (summary[m]["mean_diff"], -candidates.index(m)))


def by_coverage(results, method):
    rows = {}
    for cov in COVERAGES:
        sub = [r[method] for (t, p, n, c), r in results.items() if c == cov]
        rows[cov] = (
            float(np.mean([s["tau_diff_vs_baseline"][0] for s in sub])),
            float(np.mean([s["kendall_tau"][0] for s in sub])),
            float(np.mean([s["rmse"][0] for s in sub])),
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=250)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-c1-v1.md")
    parser.add_argument("--catalog-seed", type=int, default=0)
    parser.add_argument("--pop-seed", type=int, default=1)
    parser.add_argument("--mask-seed", type=int, default=11)
    args = parser.parse_args()

    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)
    results = {}
    for gen in GENERATORS:
        print(f"generator: {gen}")
        results[gen] = run_generator(gen, catalog, args.students, args.pop_seed, args.mask_seed)
    summaries = {gen: summarise(results[gen]) for gen in GENERATORS}
    chosen = select(summaries["archetype"])

    lines = [
        "# L3：C1.2 錨點拆解 v1（W-23），部分作答情境",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_c1.py "
        f"--catalog-seed {args.catalog_seed} --pop-seed {args.pop_seed} "
        f"--mask-seed {args.mask_seed}`",
        f"- 每個條件 {args.students} 位學生；條件 = 2 真相 × 3 名氣權重 × 2 陳述雜訊 × "
        "3 作答比例（1.0／0.5／0.25）= 36（每個生成模型）",
        "- 基準：`neutral`（未作答維度補 0.5）。差異為**配對**τ 差（同一學生、同一份遮蔽後的作答）",
        "- 顯著更差／更好：配對差 < −2 SE ／ > +2 SE",
        "- **預先登記的選擇規則**：見 `scripts/run_l3_c1.py` 檔頭；"
        "在原型模型上挑選，在連續模型上驗證",
        "",
    ]
    for gen in GENERATORS:
        role = "調參（tuning）" if gen == "archetype" else "保留驗證（held-out）"
        lines += [f"## 生成模型：{gen}（{role}）", ""]
        lines += [
            "| 方法 | 平均配對 τ 差 | 最差條件 τ 差 | 顯著更差 | 顯著更好 | 平均 τ | 平均 RMSE |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, r in summaries[gen].items():
            mark = " ⭐" if m == chosen else ""
            lines.append(
                f"| {m}{mark} | {r['mean_diff']:+.4f} | {r['min_diff']:+.4f} | "
                f"{r['worse']}/36 | {r['better']}/36 | {r['tau']:.3f} | {r['rmse']:.3f} |"
            )
        lines.append("")

    lines += ["## 選擇與保留驗證", ""]
    if not chosen:
        lines += ["**沒有任何 v1 變體在原型模型上滿足「無顯著更差」條件。**", ""]
    else:
        held = summaries["continuous"][chosen]
        verdict = "✅ 通過" if held["worse"] == 0 else "❌ 未通過"
        lines += [
            f"- 原型模型選出：**{chosen}**",
            f"- 連續模型（保留驗證）：顯著更差 {held['worse']}/36，顯著更好 {held['better']}/36，"
            f"平均配對 τ 差 {held['mean_diff']:+.4f} → {verdict}",
            "",
        ]
        lines += ["### 依作答比例（平均配對 τ 差 / 平均 τ / 平均 RMSE）", ""]
        lines += ["| 生成模型 | 方法 | 作答 100% | 作答 50% | 作答 25% |", "|---|---|---|---|---|"]
        for gen in GENERATORS:
            for m in ("neutral", "v0（中性補全後合併）", chosen):
                cov = by_coverage(results[gen], m)
                cells = " | ".join(f"{d:+.3f} / {t:.3f} / {r:.3f}" for d, t, r in cov.values())
                lines.append(f"| {gen} | {m} | {cells} |")
        lines.append("")

    cand = vname(CONFIRM_CANDIDATE)
    lines += ["## 確認候選（預先登記）", ""]
    lines += ["| 生成模型 | 顯著更差 | 顯著更好 | 平均配對 τ 差 | 判定 |", "|---|---|---|---|---|"]
    for gen in GENERATORS:
        r = summaries[gen][cand]
        ok = "✅" if r["worse"] == 0 else "❌"
        lines.append(
            f"| {gen} | {r['worse']}/36 | {r['better']}/36 | {r['mean_diff']:+.4f} | {ok} |"
        )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
