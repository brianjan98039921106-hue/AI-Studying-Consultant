"""Real-data check: replace answered interest areas by their posterior mean (F116;
THEORY "已答領域去雜訊"; A-C1-25, A-C1-26, A-C1-27).

Same units, folds and model as scripts/check_area_adaptive.py (O*NET 31.0
occupations as complete profiles over the 41 areas, 10-fold CV, seed 2026, Ledoit-Wolf
joint model of 41 areas + 12 group maxima from the training folds). Every respondent
answers the 12 group items and then q areas in the two-stage order. Methods:
    TS-G2   answered areas = raw answers; unanswered = conditional Gaussian mean
    TS-G2D  every area (answered or not) = posterior mean given all answers, with the
            answer noise variance s2 on the observed diagonal; clipped to [0, 1]
The model's s2 equals the true answer noise in the primary settings.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  Settings: true answer noise {0.10, 0.15, 0.20} x q {4, 8, 12} = 9.
  Primary: RMSE over all 41 areas. Guard: top-5 recall (as before).
  TS-G2D QUALIFIES if in all 9 settings its paired RMSE difference vs TS-G2 is below
  -2 SE and its top-5 recall is not lower by more than 2 SE. If it qualifies, it
  becomes an option of the correlated completion (rated areas returned as posterior
  means); the default stays unchanged until W-41 (A-C1-25).
  Report only: misspecified noise (true 0.15, model 0.10 or 0.20); noise 0 (identical
  by construction up to the 1e-6 jitter).
  Disclosure: F116 (the motivation) came from the adaptive check's raw output; no
  denoising was run on these data before this commit. Smoke-tested on fake random
  occupations (exit status only).

    python scripts/check_area_denoise.py --basic-interests specific_interest_areas.csv \\
        --basic-map specific_interest_areas_to_career_interest_types.csv
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from aisc.catalog.occupation import (
    basic_interest_key,
    load_basic_interest_map,
    load_onet_basic_interests,
)
from aisc.dimensions import BASIC_INTEREST_KEYS
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS, areas_in_group

ROOT = Path(__file__).resolve().parents[1]
QS = (4, 8, 12)
FOLDS = 10
SEED = 2026
GROUPS = sorted(GROUP_LABELS)
D = len(BASIC_INTEREST_KEYS)
PRIMARY = {g: [BASIC_INTEREST_KEYS.index(k) for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g]
           for g in GROUPS}  # fmt: skip
# (true noise, model noise); the first three are the pre-registered settings
SETTINGS = ((0.10, 0.10), (0.15, 0.15), (0.20, 0.20), (0.15, 0.10), (0.15, 0.20), (0.0, 0.0))
PRIMARY_SETTINGS = SETTINGS[:3]


def group_max(T: np.ndarray) -> np.ndarray:
    return np.column_stack([T[:, PRIMARY[g]].max(axis=1) for g in GROUPS])


def post_mean(mu, S, obs, y, s2):
    Soo = S[np.ix_(obs, obs)] + np.eye(len(obs)) * s2
    return mu + S[:, obs] @ np.linalg.solve(Soo, y - mu[obs])


def top5(v: np.ndarray) -> set[int]:
    return set(np.argsort(-v, kind="stable")[:5].tolist())


def two_stage_order(g_ans: np.ndarray) -> list[int]:
    order: list[int] = []
    for gi in np.argsort(-g_ans, kind="stable"):
        for k in areas_in_group(GROUPS[gi]):
            j = BASIC_INTEREST_KEYS.index(k)
            if j not in order:
                order.append(j)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic-interests", type=Path, required=True)
    ap.add_argument("--basic-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "interest-area-denoise.md")
    args = ap.parse_args()
    bmap = load_basic_interest_map(args.basic_map)
    areas = tuple(bmap)
    if tuple(basic_interest_key(a) for a in areas) != BASIC_INTEREST_KEYS:
        raise SystemExit("area order differs from BASIC_INTEREST_KEYS")
    occ = load_onet_basic_interests(args.basic_interests, areas)
    T = np.array([[(occ[c][a] - 1) / 6 for a in areas] for c in sorted(occ)])
    n = len(T)
    fold = np.random.default_rng(SEED).permutation(np.arange(n) % FOLDS)
    methods = ("TS-G2", "TS-G2D")
    res = {(st, q): {m: {"rmse": [], "rec": []} for m in methods} for st in SETTINGS for q in QS}
    gidx = D + np.arange(len(GROUPS))
    for f in range(FOLDS):
        tr, te = T[fold != f], np.flatnonzero(fold == f)
        Z = np.hstack([tr, group_max(tr)])
        mu, S = Z.mean(0), LedoitWolf().fit(Z).covariance_
        for st in SETTINGS:
            nz, model = st
            s2 = max(model**2, 1e-6)
            nrng = np.random.default_rng([SEED, f, int(nz * 100), int(model * 100)])
            for i in te:
                t = T[i]
                y = np.clip(t + nrng.normal(0, nz, D), 0, 1)
                g = np.clip(group_max(t[None])[0] + nrng.normal(0, nz, len(GROUPS)), 0, 1)
                order = two_stage_order(g)
                for q in QS:
                    asked = order[:q]
                    obs = np.concatenate([asked, gidx]).astype(int)
                    pm = post_mean(mu, S, obs, np.concatenate([y[asked], g]), s2)[:D]
                    raw = pm.copy()
                    raw[asked] = y[asked]
                    for name, full in (("TS-G2", raw), ("TS-G2D", pm)):
                        full = np.clip(full, 0, 1)
                        r = res[(st, q)][name]
                        r["rmse"].append(math.sqrt(float(np.mean((full - t) ** 2))))
                        r["rec"].append(len(top5(full) & top5(t)) / 5)

    def paired(a, b):
        d = np.array(a) - np.array(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    ok = True
    for st in PRIMARY_SETTINGS:
        for q in QS:
            r = res[(st, q)]
            dr, se_r = paired(r["TS-G2D"]["rmse"], r["TS-G2"]["rmse"])
            dc, se_c = paired(r["TS-G2D"]["rec"], r["TS-G2"]["rec"])
            if not (dr < -2 * se_r) or dc < -2 * se_c:
                ok = False

    def ms(v):
        return f"{np.mean(v):.4f}±{np.std(v, ddof=1) / math.sqrt(len(v)):.4f}"

    L = [
        "# 已答領域去雜訊：O\\*NET 職業資料交叉驗證（F116）",
        "",
        f"- 產生日期：{date.today().isoformat()} · O\\*NET 31.0 · {n} 個職業 · "
        f"{FOLDS} 折交叉驗證（種子 {SEED}）；兩階段順序，先答 12 群再答 q 個領域",
        "- 預先登記見 `scripts/check_area_denoise.py` 檔頭",
        "",
        f"## 判定：TS-G2D {'✅ 通過' if ok else '❌ 未通過'}（預先登記的 9 個設定）",
        "",
        "| 真實雜訊 | 模型雜訊 | q | 預先登記 | RMSE TS-G2 | RMSE TS-G2D | 差 "
        "| 前5 TS-G2 | 前5 TS-G2D | 差 |",
        "|---" * 10 + "|",
    ]
    for (st, q), r in res.items():
        dr = paired(r["TS-G2D"]["rmse"], r["TS-G2"]["rmse"])
        dc = paired(r["TS-G2D"]["rec"], r["TS-G2"]["rec"])
        L.append(f"| {st[0]} | {st[1]} | {q} | {'是' if st in PRIMARY_SETTINGS else '只報告'} | "
                 f"{ms(r['TS-G2']['rmse'])} | {ms(r['TS-G2D']['rmse'])} | "
                 f"{dr[0]:+.4f}±{dr[1]:.4f} | {np.mean(r['TS-G2']['rec']):.3f} | "
                 f"{np.mean(r['TS-G2D']['rec']):.3f} | {dc[0]:+.3f}±{dc[1]:.3f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"TS-G2D qualifies={ok} wrote {args.out}")


if __name__ == "__main__":
    main()
