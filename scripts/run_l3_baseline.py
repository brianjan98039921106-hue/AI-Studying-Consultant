"""First L3 synthetic validation of C1.2 v0 and C2 v0 against baselines.

Writes results/synthetic-l3-v0.md. Deterministic (fixed seeds).

    python scripts/run_l3_baseline.py [--students 400]
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np

from aisc.contracts import Program, StudentProfile
from aisc.discovery import passes_constraints, relevance
from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import (
    c2_pipeline,
    evaluate_lists,
    evaluate_profiles,
    popular_baseline,
    similar_baseline,
)
from aisc.simulation.population import SimConfig, generate_population

ROOT = Path(__file__).resolve().parents[1]
TRUTHS = ("distance", "cosine")
PRESTIGE_WEIGHTS = (0.0, 0.5, 0.9)
K = 10


def fit_exclude_known(profile: StudentProfile, catalog: list[Program], k: int) -> list[str]:
    """EXPLORATORY candidate for W-28 (not part of C2): rank by fit only and drop
    programs the student already knows, instead of adding a novelty score."""
    known = set(profile.anchors) | set(profile.familiar)
    target = profile.vector
    pool = [
        p
        for p in catalog
        if p.program_id not in known and passes_constraints(p, profile.hard_constraints)
    ]
    pool.sort(key=lambda p: (-relevance(target, p.vector), p.program_id))
    return [p.program_id for p in pool[:k]]


RECOMMENDERS = {
    "(a) popular": popular_baseline,
    "(b) similar-to-anchors": similar_baseline,
    "C2 fit-only, anchor blend": c2_pipeline(0.0, 0.0, blend=True),
    "C2 fit-only, stated only": c2_pipeline(0.0, 0.0, blend=False),
    "C2 v0 (α=0.3, λ=0.3), anchor blend": c2_pipeline(0.3, 0.3),
    "C2 v0 + familiarity (oracle C1.4)": c2_pipeline(0.3, 0.3),
    "探索：fit-only + 排除已知 (oracle C1.4)": fit_exclude_known,
}
FAMILIAR_FOR = {
    "C2 v0 + familiarity (oracle C1.4)": True,
    "探索：fit-only + 排除已知 (oracle C1.4)": True,
}
NOISE_LEVELS = (0.15, 0.30)  # sensitivity of the C1 result (B-C5-2)
LIST_COLS = [
    ("relevant_rate", "相關率"),
    ("novel_relevant_rate", "相關且不知道"),
    ("unfamiliar_rate", "不熟悉率"),
    ("best_fit_retained", "最佳契合保留"),
    ("coverage", "覆蓋率"),
    ("popularity_kl", "熱門度 KL"),
]


def fmt(v: tuple[float, float]) -> str:
    mean, se = v
    if math.isnan(mean):
        return "—"
    return f"{mean:.3f}" if math.isnan(se) else f"{mean:.3f} ± {se:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=int, default=400)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic-l3-v0.md")
    args = parser.parse_args()

    catalog = synthetic_catalog(np.random.default_rng(0), n_per_family=10)
    lines = [
        "# L3 合成驗證 v0：C1.2 v0 與 C2 v0",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/run_l3_baseline.py`",
        f"- 目錄：{len(catalog)} 個合成學程（12 個原型 × 10，族內 sd=0.12）",
        f"- 學生：每組設定 {args.students} 位 · 推薦長度 k={K} · 「相關」= 真實效用前 10%",
        "- 數值為平均 ± 標準誤。**這是合成資料，參數未校準（B-C5-2），只用來淘汰錯誤設計。**",
        "",
    ]

    for truth in TRUTHS:
        lines += [f"## 真相函數：{truth}", ""]
        for pw in PRESTIGE_WEIGHTS:
            cfg = SimConfig(n_students=args.students, prestige_weight=pw, truth=truth)
            pop = generate_population(catalog, cfg, np.random.default_rng(1))

            lines += [f"### 名氣權重 π = {pw}", "", "**C1 側寫還原（V-C1-1、V-C1-2）**", ""]
            lines += ["| 側寫 | RMSE ↓ | Kendall τ ↑ |", "|---|---|---|"]
            for name, res in evaluate_profiles(pop, catalog, truth).items():
                lines.append(f"| {name} | {fmt(res['rmse'])} | {fmt(res['kendall_tau'])} |")

            lines += ["", "**C2 推薦清單（V-C2-2、V-C2-3、V-C2-4）**", ""]
            header = "| 推薦器 | " + " | ".join(label for _, label in LIST_COLS) + " |"
            lines += [header, "|---" * (len(LIST_COLS) + 1) + "|"]
            results = evaluate_lists(
                pop, catalog, RECOMMENDERS, k=K, truth=truth, include_familiar=FAMILIAR_FOR
            )
            for name, res in results.items():
                lines.append(f"| {name} | " + " | ".join(fmt(res[c]) for c, _ in LIST_COLS) + " |")
            lines.append("")
            print(f"done truth={truth} π={pw}")

    lines += [
        "## 敏感度：陳述需求的雜訊（C1，B-C5-2）",
        "",
        "陳述需求宣稱的不確定性固定為 0.3；改變實際雜訊，"
        "檢查「錨點合併是否有害」是否只是參數造成。",
        "",
        "| 真相 | π | 實際雜訊 | stated_only τ | anchor_blend τ "
        "| stated_only RMSE | anchor_blend RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for truth in TRUTHS:
        for pw in (0.0, 0.5):
            for noise in NOISE_LEVELS:
                cfg = SimConfig(
                    n_students=args.students, prestige_weight=pw, truth=truth, noise_sd=noise
                )
                pop = generate_population(catalog, cfg, np.random.default_rng(1))
                r = evaluate_profiles(pop, catalog, truth)
                lines.append(
                    f"| {truth} | {pw} | {noise} | {fmt(r['stated_only']['kendall_tau'])} | "
                    f"{fmt(r['anchor_blend']['kendall_tau'])} | {fmt(r['stated_only']['rmse'])} | "
                    f"{fmt(r['anchor_blend']['rmse'])} |"
                )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
