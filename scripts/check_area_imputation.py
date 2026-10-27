"""Real-data check: complete unanswered interest areas from their correlations
(docs/architecture-review.md §2.1; THEORY "領域相關補全"; A-C1-25, A-C1-26).

Units: the O*NET 31.0 occupations with Specific Interest Area ratings (OI scale,
(v - 1) / 6), used as stand-ins for complete need profiles over the 41 areas.
10-fold cross-validation over occupations (seed 2026): means and covariances come
from the training folds only. For each test occupation a two-stage respondent is
simulated (dimensions-v2 §2, A-C5-12): 12 group answers = max of the group's primary
areas, then every area of the top-m groups is answered; answers = clip(true + noise).
Unanswered areas are predicted by
    H   current hierarchy prior: rated group score, else occupation base rate
        (profiling.hierarchy.area_prior; its groups and base rates were derived from
        all occupations, which can only favour H)
    B   training mean (reference)
    G1  conditional Gaussian on the answered areas
    G2  conditional Gaussian on the answered areas and the 12 group answers (joint
        model of 41 areas + 12 group maxima)
    Covariance: Ledoit-Wolf shrinkage; answer noise variance added to the observed
    diagonal (1e-6 when noise is 0).

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  Settings: answer noise {0, 0.15} x m {1, 2, 3} = 6.
  Primary metric: RMSE over the unanswered areas of each test occupation.
  Guard metric: top-5 recall = overlap between the 5 highest areas of the completed
  profile (answered values + predictions) and of the true profile, divided by 5.
  Candidates in order G2, G1. A candidate is ADOPTED if in all 6 settings its paired
  RMSE difference vs H is negative by more than 2 SE, and its top-5 recall is not
  lower than H's by more than 2 SE. The first candidate that qualifies is adopted;
  otherwise H stays. Report only: B, worst primary group (D12), per-area errors.
  Caveat written in advance: occupations are expert-rated profiles, not students;
  adoption is provisional until W-41 student data (A-C1-25).
  Disclosure: before the first run on O*NET data the script was only dry-run on a
  file of 60 fake occupations with uniform random ratings, to check that it runs.

    python scripts/check_area_imputation.py --basic-interests specific_interest_areas.csv \\
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
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS, area_prior, areas_in_group

ROOT = Path(__file__).resolve().parents[1]
NOISES = (0.0, 0.15)
MS = (1, 2, 3)
FOLDS = 10
SEED = 2026
GROUPS = sorted(GROUP_LABELS)
PRIMARY = {g: [BASIC_INTEREST_KEYS.index(k) for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g]
           for g in GROUPS}  # fmt: skip


def group_max(T: np.ndarray) -> np.ndarray:
    return np.column_stack([T[:, PRIMARY[g]].max(axis=1) for g in GROUPS])


def conditional(mu, S, obs_idx, obs_val, noise_var):
    """Mean of the unobserved coordinates given noisy observations of obs_idx."""
    n = len(mu)
    un = np.setdiff1d(np.arange(n), obs_idx)
    Soo = S[np.ix_(obs_idx, obs_idx)] + np.eye(len(obs_idx)) * noise_var
    Suo = S[np.ix_(un, obs_idx)]
    return un, mu[un] + Suo @ np.linalg.solve(Soo, obs_val - mu[obs_idx])


def top5(v: np.ndarray) -> set[int]:
    return set(np.argsort(-v, kind="stable")[:5].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic-interests", type=Path, required=True)
    ap.add_argument("--basic-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "interest-area-imputation.md")
    args = ap.parse_args()
    bmap = load_basic_interest_map(args.basic_map)
    areas = tuple(bmap)
    keys = tuple(basic_interest_key(a) for a in areas)
    if keys != BASIC_INTEREST_KEYS:
        raise SystemExit("area order differs from BASIC_INTEREST_KEYS")
    occ = load_onet_basic_interests(args.basic_interests, areas)
    codes = sorted(occ)
    T = np.array([[(occ[c][a] - 1) / 6 for a in areas] for c in codes])
    n, d = T.shape
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(np.arange(n) % FOLDS)
    methods = ("H", "B", "G1", "G2")
    res = {(nz, m): {k: {"rmse": [], "rec": []} for k in methods} for nz in NOISES for m in MS}
    top_group = []
    for i in range(n):
        g_true = group_max(T[i : i + 1])[0]
        top_group.append(GROUPS[int(np.argmax(g_true))])
    for f in range(FOLDS):
        tr, te = T[fold != f], np.flatnonzero(fold == f)
        mu1 = tr.mean(0)
        S1 = LedoitWolf().fit(tr).covariance_
        Z = np.hstack([tr, group_max(tr)])
        mu2 = Z.mean(0)
        S2 = LedoitWolf().fit(Z).covariance_
        for nz, m in res:
            nrng = np.random.default_rng([SEED, f, int(nz * 100), m])
            for i in te:
                t = T[i]
                y = np.clip(t + nrng.normal(0, nz, d), 0, 1)
                g = np.clip(group_max(t[None])[0] + nrng.normal(0, nz, len(GROUPS)), 0, 1)
                gs = dict(zip(GROUPS, g.tolist(), strict=True))
                top = [GROUPS[j] for j in np.argsort(-g, kind="stable")[:m]]
                asked = sorted({BASIC_INTEREST_KEYS.index(k) for gg in top
                                for k in areas_in_group(gg)})  # fmt: skip
                un = np.setdiff1d(np.arange(d), asked)
                if not len(un):
                    continue
                nv = max(nz**2, 1e-6)
                preds = {
                    "H": np.array([area_prior(BASIC_INTEREST_KEYS[j], gs)[0] for j in un]),
                    "B": mu1[un],
                    "G1": conditional(mu1, S1, np.array(asked), y[asked], nv)[1],
                }
                obs2 = np.concatenate([asked, d + np.arange(len(GROUPS))])
                u2, p2 = conditional(mu2, S2, obs2, np.concatenate([y[asked], g]), nv)
                if not np.array_equal(u2, un):
                    raise RuntimeError("group variables must all be observed")
                preds["G2"] = p2
                for k, p in preds.items():
                    p = np.clip(p, 0, 1)
                    full = y.copy()
                    full[un] = p
                    res[(nz, m)][k]["rmse"].append(math.sqrt(float(np.mean((p - t[un]) ** 2))))
                    res[(nz, m)][k]["rec"].append(len(top5(full) & top5(t)) / 5)
                res[(nz, m)].setdefault("_idx", []).append(i)

    def paired(a, b):
        dlt = np.array(a) - np.array(b)
        return float(dlt.mean()), float(dlt.std(ddof=1) / math.sqrt(len(dlt)))

    verdict = {}
    for cand in ("G2", "G1"):
        ok = True
        for r in res.values():
            dr, se_r = paired(r[cand]["rmse"], r["H"]["rmse"])
            dc, se_c = paired(r[cand]["rec"], r["H"]["rec"])
            if not (dr < -2 * se_r) or dc < -2 * se_c:
                ok = False
        verdict[cand] = ok
    adopted = next((c for c in ("G2", "G1") if verdict[c]), "H")

    L = [
        "# 領域相關補全：O\\*NET 職業資料交叉驗證（architecture-review §2.1）",
        "",
        f"- 產生日期：{date.today().isoformat()} · O\\*NET 31.0 Specific Interest Areas · "
        f"{n} 個職業 × {d} 個領域 · {FOLDS} 折交叉驗證（種子 {SEED}）",
        "- 預先登記見 `scripts/check_area_imputation.py` 檔頭；職業剖面只是學生的替代（A-C1-25）",
        "",
        "## 判定",
        "",
        f"- G2（領域＋群分數）：{'✅' if verdict['G2'] else '❌'}；"
        f"G1（只用領域）：{'✅' if verdict['G1'] else '❌'} → 採用 **{adopted}**",
        "",
        "## 未答領域 RMSE 與前 5 名領域找回率（平均 ± 標準誤；差 = 方法 − H）",
        "",
        "| 雜訊 | m | 未答數 | H | B | G1 | G2 | G1 − H | G2 − H | 前5 H | 前5 G1 | 前5 G2 "
        "| 前5 G2 − H |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (nz, m), r in res.items():

        def ms(v):
            return f"{np.mean(v):.4f}±{np.std(v, ddof=1) / math.sqrt(len(v)):.4f}"

        g1 = paired(r["G1"]["rmse"], r["H"]["rmse"])
        g2 = paired(r["G2"]["rmse"], r["H"]["rmse"])
        c2 = paired(r["G2"]["rec"], r["H"]["rec"])
        L.append(f"| {nz} | {m} | {len(r['H']['rmse'])} | {ms(r['H']['rmse'])} | "
                 f"{ms(r['B']['rmse'])} | {ms(r['G1']['rmse'])} | {ms(r['G2']['rmse'])} | "
                 f"{g1[0]:+.4f}±{g1[1]:.4f} | {g2[0]:+.4f}±{g2[1]:.4f} | "
                 f"{np.mean(r['H']['rec']):.3f} | {np.mean(r['G1']['rec']):.3f} | "
                 f"{np.mean(r['G2']['rec']):.3f} | {c2[0]:+.3f}±{c2[1]:.3f} |")  # fmt: skip
    L += ["", "## 依主要領域群（雜訊 0.15、m = 2；G2 − H 的未答 RMSE 差）", "",
          "| 群 | 名稱 | 職業數 | G2 − H |", "|---|---|---|---|"]  # fmt: skip
    r = res[(0.15, 2)]
    idx = np.array(r["_idx"])
    for g in GROUPS:
        sel = np.array([top_group[i] == g for i in idx])
        if sel.sum() >= 2:
            dlt = np.array(r["G2"]["rmse"])[sel] - np.array(r["H"]["rmse"])[sel]
            se = dlt.std(ddof=1) / math.sqrt(sel.sum())
            L.append(f"| {g} | {GROUP_LABELS[g][0]} | {int(sel.sum())} | "
                     f"{dlt.mean():+.4f}±{se:.4f} |")  # fmt: skip
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"G2={verdict['G2']} G1={verdict['G1']} adopted={adopted} wrote {args.out}")


if __name__ == "__main__":
    main()
