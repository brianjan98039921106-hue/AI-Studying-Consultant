"""L3 for the C1.3 adaptive questionnaire (W-25).

PRE-REGISTERED RULES (written before the first run, 2026-10-25):
  Metric: Kendall τ between C2's ranking (cosine relevance of the posterior
  mean) and the true-utility ranking; paired per student against "random".
  Conditions per generator: 2 truths x 2 true item-noise levels (0.25 = the
  estimator's assumption, 0.40 = misspecified) = 4. Response biases
  (acquiescence, social desirability) are always on and never modelled.

  1. SELECT on the ARCHETYPE generator between "discriminative" and "voi":
     drop any strategy significantly worse than random (paired diff < -2 SE)
     at any checkpoint in any condition; pick the higher mean τ over
     checkpoints 1..15 averaged over conditions.
  2. VALIDATE on the CONTINUOUS generator (held out). Pass if
     (a) never significantly worse than random at any checkpoint, and
     (b) V-C1-3: questions needed to reach random's τ at 15 questions,
         saving = 1 - Q/15 (linear interpolation), is > 0 in every condition
         and >= 0.30 on average.
  3. CONFIRM on fresh seeds (--catalog-seed 200 --pop-seed 201 --resp-seed 202)
     with the same criteria on both generators.
  "variance" (plain uncertainty sampling) is reported as a secondary baseline.

REPLICATION (pre-registered 2026-10-25, after run 1 failed step 2b): rerun on
fresh seeds (--catalog-seed 200 --pop-seed 201 --resp-seed 202). Hypotheses:
  H1: on the continuous generator, no strategy reaches >= 30% average saving;
  H2: on the continuous generator, "variance" has mean τ (Q=1..15) >= "voi".
  Both must hold for the run-1 conclusions to be reported as replicated.

    python scripts/run_l3_adaptive.py [--students 200]
    python scripts/run_l3_adaptive.py --catalog-seed 200 --pop-seed 201 --resp-seed 202 \\
        --out results/synthetic-l3-adaptive-confirm.md
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from aisc.contracts import StudentProfile
from aisc.discovery import relevance_matrix
from aisc.pipeline import prepare_profile
from aisc.profiling import STRATEGIES, AdaptiveSession, default_item_bank
from aisc.simulation import metrics as M
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.respondent import make_item_responder
from aisc.simulation.truth import utilities

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ("archetype", "continuous")
TRUTHS = ("distance", "cosine")
TRUE_NOISES = (0.25, 0.40)
ASSUMED_NOISE = 0.25
CHECKPOINTS = (0, 1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 45)
SELECT_FROM = ("discriminative", "voi")


def run_condition(catalog, gen, truth, true_noise, n, pop_seed, resp_seed):
    cfg = SimConfig(n_students=n, prestige_weight=0.5, truth=truth, needs_model=gen)
    pop = generate_population(catalog, cfg, np.random.default_rng(pop_seed))
    X = np.array([p.vector for p in catalog])
    bank = default_item_bank()
    taus = {s: {c: [] for c in CHECKPOINTS} for s in STRATEGIES}
    rmses = {s: {c: [] for c in CHECKPOINTS} for s in STRATEGIES}
    for idx, st in enumerate(pop):
        respond = make_item_responder(
            st,
            np.random.default_rng([resp_seed, idx]),
            noise_sd=true_noise,
            acquiescence=cfg.acquiescence,
            desirability=cfg.desirability,
        )
        answers = {item.item_id: respond(item) for item in bank}  # common random numbers
        prior = prepare_profile(
            StudentProfile(st.student_id, {}, anchors=list(st.anchors)), catalog
        )
        true_u = utilities(st.true_needs, catalog, truth)
        for strategy in STRATEGIES:
            session = AdaptiveSession(
                prior,
                catalog,
                strategy=strategy,
                item_noise_sd=ASSUMED_NOISE,
                score_fn=relevance_matrix,
                rng=np.random.default_rng([resp_seed, idx, 7]),
            )
            for q in range(max(CHECKPOINTS) + 1):
                if q in taus[strategy]:
                    taus[strategy][q].append(
                        M.kendall_tau_b(relevance_matrix(session.mean, X), true_u)
                    )
                    rmses[strategy][q].append(M.rmse(session.mean, st.true_needs))
                item = session.next_item()
                if item is None:
                    break
                session.record(item, answers[item.item_id])
    out = {}
    for s in STRATEGIES:
        out[s] = {}
        for c in CHECKPOINTS:
            diff = np.array(taus[s][c]) - np.array(taus["random"][c])
            out[s][c] = {
                "tau": M.mean_se(taus[s][c]),
                "rmse": M.mean_se(rmses[s][c]),
                "diff": M.mean_se(list(diff)),
            }
    return out


def questions_to_reach(curve: dict[int, float], target: float) -> float:
    """Linear interpolation of the budget at which the mean τ curve reaches target."""
    pts = sorted(curve.items())
    for (q0, t0), (q1, t1) in zip(pts, pts[1:], strict=False):
        if t0 >= target:
            return float(q0)
        if t1 >= target:
            return q0 + (target - t0) * (q1 - q0) / (t1 - t0)
    return math.inf


def worse_anywhere(res, strategy) -> int:
    return sum(
        1
        for cond in res.values()
        for c in CHECKPOINTS
        if c > 0
        and not math.isnan(cond[strategy][c]["diff"][1])
        and cond[strategy][c]["diff"][0] < -2 * cond[strategy][c]["diff"][1]
    )


def mean_auc(res, strategy) -> float:
    return float(
        np.mean([cond[strategy][c]["tau"][0] for cond in res.values() for c in CHECKPOINTS
                 if 1 <= c <= 15])
    )  # fmt: skip


def savings(res, strategy) -> list[float]:
    out = []
    for cond in res.values():
        target = cond["random"][15]["tau"][0]
        q = questions_to_reach({c: cond[strategy][c]["tau"][0] for c in CHECKPOINTS}, target)
        out.append(1.0 - q / 15.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=200)
    parser.add_argument("--catalog-seed", type=int, default=0)
    parser.add_argument("--pop-seed", type=int, default=1)
    parser.add_argument("--resp-seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-adaptive.md")
    args = parser.parse_args()

    catalog = synthetic_catalog(np.random.default_rng(args.catalog_seed), n_per_family=10)
    results = {}
    for gen in GENERATORS:
        results[gen] = {}
        for truth, noise in product(TRUTHS, TRUE_NOISES):
            results[gen][(truth, noise)] = run_condition(
                catalog, gen, truth, noise, args.students, args.pop_seed, args.resp_seed
            )
            print(f"  {gen} truth={truth} true_noise={noise}")

    arche, cont = results["archetype"], results["continuous"]
    eligible = [s for s in SELECT_FROM if worse_anywhere(arche, s) == 0]
    chosen = max(eligible, key=lambda s: mean_auc(arche, s)) if eligible else None

    lines = [
        "# L3：C1.3 自適應問卷 v1（W-25）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_adaptive.py "
        f"--catalog-seed {args.catalog_seed} --pop-seed {args.pop_seed} "
        f"--resp-seed {args.resp_seed}`",
        f"- 每個條件 {args.students} 位學生；題庫 45 題（每維度 3 題）；估計器假設題目雜訊 "
        f"{ASSUMED_NOISE}；作答偏誤（默許、社會期許）始終存在且未被建模",
        "- 指標：C2 cosine 排序（後驗平均）與真實效用排序的 Kendall τ；配對比較對象為 random；"
        "每位學生每題的回答只抽一次（共同隨機數）",
        "- 預先登記的規則：見 `scripts/run_l3_adaptive.py` 檔頭",
        "",
    ]
    for gen in GENERATORS:
        role = "挑選" if gen == "archetype" else "保留驗證"
        lines += [f"## 生成模型：{gen}（{role}）", ""]
        for (truth, noise), cond in results[gen].items():
            tag = "與假設一致" if noise == ASSUMED_NOISE else "誤設"
            lines += [f"### 真相 {truth}，真實題目雜訊 {noise}（{tag}）", ""]
            header = "| 策略 | " + " | ".join(f"Q={c}" for c in CHECKPOINTS) + " |"
            lines += [header, "|---" * (len(CHECKPOINTS) + 1) + "|"]
            for s in STRATEGIES:
                cells = []
                for c in CHECKPOINTS:
                    t = cond[s][c]["tau"][0]
                    d, se = cond[s][c]["diff"]
                    sig = (
                        ""
                        if s == "random" or c == 0
                        else ("▲" if d > 2 * se else "▼" if d < -2 * se else "")
                    )
                    cells.append(f"{t:.3f}{sig}")
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += [
            "| 策略 | 顯著劣於 random 的檢查點數 | 平均 τ（Q=1–15） | 各條件節省題數 | 平均節省 |",
            "|---|---|---|---|---|",
        ]
        for s in STRATEGIES[1:]:
            sv = savings(results[gen], s)
            lines.append(
                f"| {s} | {worse_anywhere(results[gen], s)} | {mean_auc(results[gen], s):.3f} | "
                + ", ".join(f"{x:+.0%}" for x in sv)
                + f" | {np.mean(sv):+.0%} |"
            )
        lines.append("")

    lines += ["## RMSE（V-C1-4；條件平均）", ""]
    lines += ["| 生成模型 | 真實題目雜訊 | 策略 | Q=15 | Q=45 |", "|---|---|---|---|---|"]
    for gen in GENERATORS:
        for noise in TRUE_NOISES:
            conds = [c for (t, n), c in results[gen].items() if n == noise]
            for s in STRATEGIES:
                r15 = np.mean([c[s][15]["rmse"][0] for c in conds])
                r45 = np.mean([c[s][45]["rmse"][0] for c in conds])
                lines.append(f"| {gen} | {noise} | {s} | {r15:.3f} | {r45:.3f} |")
    lines.append("")

    sv_cont = {s: np.mean(savings(cont, s)) for s in STRATEGIES[1:]}
    h1 = all(v < 0.30 for v in sv_cont.values())
    h2 = mean_auc(cont, "variance") >= mean_auc(cont, "voi")
    lines += [
        "## 重現假設（預先登記）",
        "",
        f"- H1 連續模型沒有策略平均節省 ≥ 30%：{'✅ 成立' if h1 else '❌ 不成立'}"
        f"（{', '.join(f'{k} {v:+.0%}' for k, v in sv_cont.items())}）",
        f"- H2 連續模型 variance 平均 τ ≥ voi：{'✅ 成立' if h2 else '❌ 不成立'}"
        f"（{mean_auc(cont, 'variance'):.3f} vs {mean_auc(cont, 'voi'):.3f}）",
        "",
    ]

    lines += ["## 挑選與驗證（預先登記）", ""]
    if chosen is None:
        lines += ["**原型模型上沒有策略符合「不顯著劣於 random」條件。**", ""]
    else:
        sv = savings(cont, chosen)
        ok_a = worse_anywhere(cont, chosen) == 0
        ok_b = all(x > 0 for x in sv) and np.mean(sv) >= 0.30
        lines += [
            f"- 原型模型選出：**{chosen}**",
            f"- 連續模型 (a) 不劣於 random：{'✅' if ok_a else '❌'}"
            f"（顯著更差檢查點 {worse_anywhere(cont, chosen)}）",
            f"- 連續模型 (b) V-C1-3 節省題數：{'✅' if ok_b else '❌'}"
            f"（各條件 {', '.join(f'{x:+.0%}' for x in sv)}；平均 {np.mean(sv):+.0%}）",
            f"- 判定：{'✅ 通過' if ok_a and ok_b else '❌ 未通過'}",
            "",
        ]
        if chosen:
            sv_a = savings(arche, chosen)
            lines += [
                f"- 參考：同一策略在原型模型的節省題數 {', '.join(f'{x:+.0%}' for x in sv_a)}",
                "",
            ]
    lines += ["▲／▼：相對 random 的配對差顯著（> 2 SE）較好／較差。", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
