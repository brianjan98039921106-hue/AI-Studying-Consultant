"""Real-data check: adaptive choice of interest-area questions (B-C1-21; THEORY
"領域的自適應提問"; A-C1-25, A-C1-26, A-C1-27).

Units and model as scripts/check_area_imputation.py: the O*NET 31.0 occupations with
Specific Interest Area ratings ((v - 1) / 6) stand in for complete profiles over the
41 areas; 10-fold cross-validation over occupations (seed 2026); mean and Ledoit-Wolf
covariance of the 41 areas + 12 group maxima from the training folds only. Every
simulated respondent first answers the 12 group items (group max + noise), then q
area items (true + noise; answers clipped to [0, 1]). Methods, all with the same q:
    TS-H   two-stage order: areas of the highest-rated group first, then the next
           group (areas_in_group order, no repeats), cut at q; unanswered areas from
           the current hierarchy prior (area_prior)
    TS-G2  same questions; unanswered areas from the conditional Gaussian (G2)
    AD-A   sequential: next area = largest drop in the summed posterior variance of
           the 41 areas, sum_k C[k, j]^2 / (C[j, j] + s2); completion by G2
    AD-U   sequential: next area = largest posterior mean + posterior sd; G2
    s2 = answer noise variance (1e-6 when noise is 0), as in the imputation check.
Completed profile = answers for asked areas, predictions (clipped) for the rest.

PRE-REGISTERED (written and committed before the first run, 2026-10-28):
  Settings: answer noise {0, 0.15} x q {4, 8, 12} = 6.
  Primary: RMSE over all 41 areas of the completed profile.
  Guard: top-5 recall (overlap of the 5 highest areas of completed and true profile / 5).
  Comparator: TS-G2 (best existing option, F105). A candidate (AD-A, AD-U) QUALIFIES
  if in all 6 settings its paired RMSE difference vs TS-G2 is below -2 SE and its
  top-5 recall is not lower than TS-G2's by more than 2 SE. If both qualify, the one
  with the larger mean RMSE reduction over the 6 settings is chosen; it becomes an
  opt-in question-order function for v2 (the default v2 flow stays two-stage until
  W-41 student data, A-C1-25). None qualifies -> negative result, nothing added.
  Report only: all methods vs TS-H (the default), and whether a candidate at q = 4 or
  8 reaches TS-H's q = 12 RMSE (possible item savings, B-C1-21).
  Caveat written in advance: occupations are expert-rated profiles, not students.
  Disclosure: the script reuses the imputation check's data and folds, whose results
  (F105-F108) are known; no adaptive strategy was run on these data before this commit.
  The finished script was smoke-tested on 80 fake occupations with uniform random
  ratings (exit status only).

    python scripts/check_area_adaptive.py --basic-interests specific_interest_areas.csv \\
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
QS = (4, 8, 12)
FOLDS = 10
SEED = 2026
GROUPS = sorted(GROUP_LABELS)
D = len(BASIC_INTEREST_KEYS)
PRIMARY = {g: [BASIC_INTEREST_KEYS.index(k) for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == g]
           for g in GROUPS}  # fmt: skip
METHODS = ("TS-H", "TS-G2", "AD-A", "AD-U")


def group_max(T: np.ndarray) -> np.ndarray:
    return np.column_stack([T[:, PRIMARY[g]].max(axis=1) for g in GROUPS])


def posterior(mu, S, obs, y, s2):
    """Posterior mean and covariance of all variables given noisy observations."""
    if not len(obs):
        return mu.copy(), S.copy()
    Soo = S[np.ix_(obs, obs)] + np.eye(len(obs)) * s2
    K = np.linalg.solve(Soo, S[obs, :]).T  # (n, |obs|)
    return mu + K @ (y - mu[obs]), S - K @ S[obs, :]


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
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "interest-area-adaptive.md")
    args = ap.parse_args()
    bmap = load_basic_interest_map(args.basic_map)
    areas = tuple(bmap)
    if tuple(basic_interest_key(a) for a in areas) != BASIC_INTEREST_KEYS:
        raise SystemExit("area order differs from BASIC_INTEREST_KEYS")
    occ = load_onet_basic_interests(args.basic_interests, areas)
    codes = sorted(occ)
    T = np.array([[(occ[c][a] - 1) / 6 for a in areas] for c in codes])
    n = len(T)
    fold = np.random.default_rng(SEED).permutation(np.arange(n) % FOLDS)
    res = {(nz, q): {m: {"rmse": [], "rec": []} for m in METHODS} for nz in NOISES for q in QS}
    gidx = D + np.arange(len(GROUPS))
    for f in range(FOLDS):
        tr, te = T[fold != f], np.flatnonzero(fold == f)
        Z = np.hstack([tr, group_max(tr)])
        mu, S = Z.mean(0), LedoitWolf().fit(Z).covariance_
        for nz in NOISES:
            s2 = max(nz**2, 1e-6)
            nrng = np.random.default_rng([SEED, f, int(nz * 100)])
            for i in te:
                t = T[i]
                y = np.clip(t + nrng.normal(0, nz, D), 0, 1)
                g = np.clip(group_max(t[None])[0] + nrng.normal(0, nz, len(GROUPS)), 0, 1)
                gs = dict(zip(GROUPS, g.tolist(), strict=True))
                ts = two_stage_order(g)
                seq = {}
                for name in ("AD-A", "AD-U"):
                    asked: list[int] = []
                    for _ in range(max(QS)):
                        obs = np.concatenate([asked, gidx]).astype(int)
                        m_post, C = posterior(mu, S, obs, np.concatenate([y[asked], g]), s2)
                        cand = [j for j in range(D) if j not in asked]
                        if name == "AD-A":
                            crit = [float(np.sum(C[:D, j] ** 2) / (C[j, j] + s2)) for j in cand]
                        else:
                            crit = [m_post[j] + math.sqrt(max(C[j, j], 0)) for j in cand]
                        asked.append(cand[int(np.argmax(crit))])
                    seq[name] = asked
                for q in QS:
                    plans = {"TS-H": ts[:q], "TS-G2": ts[:q], "AD-A": seq["AD-A"][:q],
                             "AD-U": seq["AD-U"][:q]}  # fmt: skip
                    for name, asked in plans.items():
                        full = np.empty(D)
                        un = [j for j in range(D) if j not in asked]
                        full[asked] = y[asked]
                        if name == "TS-H":
                            full[un] = [area_prior(BASIC_INTEREST_KEYS[j], gs)[0] for j in un]
                        else:
                            obs = np.concatenate([asked, gidx]).astype(int)
                            m_post, _ = posterior(mu, S, obs, np.concatenate([y[asked], g]), s2)
                            full[un] = m_post[un]
                        full = np.clip(full, 0, 1)
                        r = res[(nz, q)][name]
                        r["rmse"].append(math.sqrt(float(np.mean((full - t) ** 2))))
                        r["rec"].append(len(top5(full) & top5(t)) / 5)

    def paired(a, b):
        d = np.array(a) - np.array(b)
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))

    verdict, gain = {}, {}
    for cand in ("AD-A", "AD-U"):
        ok, gains = True, []
        for r in res.values():
            dr, se_r = paired(r[cand]["rmse"], r["TS-G2"]["rmse"])
            dc, se_c = paired(r[cand]["rec"], r["TS-G2"]["rec"])
            gains.append(-dr)
            if not (dr < -2 * se_r) or dc < -2 * se_c:
                ok = False
        verdict[cand], gain[cand] = ok, float(np.mean(gains))
    qual = [c for c in ("AD-A", "AD-U") if verdict[c]]
    chosen = max(qual, key=lambda c: gain[c]) if qual else None

    def ms(v):
        return f"{np.mean(v):.4f}±{np.std(v, ddof=1) / math.sqrt(len(v)):.4f}"

    L = [
        "# 領域的自適應提問：O\\*NET 職業資料交叉驗證（B-C1-21）",
        "",
        f"- 產生日期：{date.today().isoformat()} · O\\*NET 31.0 Specific Interest Areas · "
        f"{n} 個職業 · {FOLDS} 折交叉驗證（種子 {SEED}）",
        "- 每位作答者先答 12 個群，再答 q 個領域；預先登記見 `scripts/check_area_adaptive.py` 檔頭",
        "",
        "## 判定",
        "",
        f"- AD-A（A 最適）：{'✅' if verdict['AD-A'] else '❌'}；AD-U（上信賴界）："
        f"{'✅' if verdict['AD-U'] else '❌'} → "
        f"{'採用 **' + chosen + '**（可選用）' if chosen else '都不採用'}",
        "",
        "## 41 領域 RMSE 與前 5 名找回率（平均 ± 標準誤）",
        "",
        "| 雜訊 | q | TS-H | TS-G2 | AD-A | AD-U | AD-A − TS-G2 | AD-U − TS-G2 "
        "| 前5 TS-H | 前5 TS-G2 | 前5 AD-A | 前5 AD-U | 前5 AD-A − TS-G2 | 前5 AD-U − TS-G2 |",
        "|---" * 14 + "|",
    ]
    for (nz, q), r in res.items():
        da, du = (
            paired(r["AD-A"]["rmse"], r["TS-G2"]["rmse"]),
            paired(r["AD-U"]["rmse"], r["TS-G2"]["rmse"]),
        )
        ca, cu = (
            paired(r["AD-A"]["rec"], r["TS-G2"]["rec"]),
            paired(r["AD-U"]["rec"], r["TS-G2"]["rec"]),
        )
        L.append(
            f"| {nz} | {q} | "
            + " | ".join(ms(r[m]["rmse"]) for m in METHODS)
            + f" | {da[0]:+.4f}±{da[1]:.4f} | {du[0]:+.4f}±{du[1]:.4f} | "
            + " | ".join(f"{np.mean(r[m]['rec']):.3f}" for m in METHODS)
            + f" | {ca[0]:+.3f}±{ca[1]:.3f} | {cu[0]:+.3f}±{cu[1]:.3f} |"
        )
    L += ["", "## 可能省題（只報告）：達到 TS-H 在 q = 12 的 RMSE 所需的最小 q", "",
          "| 雜訊 | TS-H q=12 | TS-G2 | AD-A | AD-U |", "|---|---|---|---|---|"]  # fmt: skip
    for nz in NOISES:
        ref = np.mean(res[(nz, 12)]["TS-H"]["rmse"])
        cells = []
        for m in ("TS-G2", "AD-A", "AD-U"):
            hit = [q for q in QS if np.mean(res[(nz, q)][m]["rmse"]) <= ref]
            cells.append(str(min(hit)) if hit else "> 12")
        L.append(f"| {nz} | {ref:.4f} | " + " | ".join(cells) + " |")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"AD-A={verdict['AD-A']} AD-U={verdict['AD-U']} chosen={chosen} wrote {args.out}")


if __name__ == "__main__":
    main()
