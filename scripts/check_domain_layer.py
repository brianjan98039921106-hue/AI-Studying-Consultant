"""Real-data check for W-64: does the occupation bridge carry field information
with RIASEC alone, and does adding O*NET's 41 specific interest areas help?

Units: every 6-digit CIP 2020 code with at least one first-major award at IPEDS
award level 7 (master's), 17 (research doctorate) or 18 (professional practice
doctorate) in the completions file, and with an occupation profile in both
spaces (all fields, not only the MVP scope).
Spaces (all values on [0, 1] via (v - 1) / 6, equal weight over SOC codes):
  S1  RIASEC (6)
  S2  41 specific interest areas
  S3  S1 and S2 concatenated, each block divided by sqrt(its size) (equal block weight)
External label: the 2-digit CIP family (NCES), independent of O*NET ratings.

PRE-REGISTERED (written before the first run, 2026-10-27):
  M1 field agreement: share of a CIP's 10 nearest other CIPs (Euclidean) in the
     same 2-digit family; averaged over CIPs.
  H-D1: M1(S2) > M1(S1), paired over CIPs, difference > 2 SE.
  H-D2: M1(S3) > M1(S1), same test.
  Report only: silhouette of the family labels in each space; M1 by family;
  nearest neighbours of a few named CIPs in S1 and S3.
  Caveat written in advance: CIPs of one family often share SOC codes (e.g.
  subject-specific postsecondary teachers), which raises M1 in every space
  equally; only the differences between spaces are interpreted.

    python scripts/check_domain_layer.py --crosswalk ... --interests ... \\
        --basic-interests ... --basic-map ... --completions ... --cip ...
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np

from aisc.catalog.occupation import (
    RIASEC_KEYS,
    basic_interest_key,
    load_basic_interest_map,
    load_cip_soc_crosswalk,
    load_onet_basic_interests,
    load_onet_interests,
    occupation_profile,
    soc_profiles,
)
from aisc.catalog.reference import load_cip2020, normalize_cip, read_table

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {7, 17, 18}
K = 10
EXAMPLES = ("27.0501", "26.0101", "51.0401", "30.7102", "50.0401", "22.0101")


def offered_cips(path: Path) -> set[str]:
    rows = read_table(path, required=["CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"])
    out = set()
    for r in rows:
        if int(r["AWLEVEL"]) in LEVELS and r["MAJORNUM"] == "1" and float(r["CTOTALT"] or 0) > 0:
            try:
                code = normalize_cip(r["CIPCODE"])
            except ValueError:
                continue
            if len(code) == 7:
                out.add(code)
    return out


def knn_agreement(X: np.ndarray, fam: np.ndarray, k: int = K) -> np.ndarray:
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]
    return (fam[nn] == fam[:, None]).mean(axis=1)


def silhouette(X: np.ndarray, fam: np.ndarray) -> float:
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = (fam == fam[i]) & (np.arange(len(X)) != i)
        if not same.any():
            continue
        a = d[i, same].mean()
        b = min(d[i, fam == f].mean() for f in set(fam) if f != fam[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def mean_se(v: np.ndarray) -> tuple[float, float]:
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("cip", "crosswalk", "interests", "basic-interests", "basic-map", "completions"):
        ap.add_argument(f"--{a}", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "domain-layer-check.md")
    args = ap.parse_args()

    titles = {c: e.title for c, e in load_cip2020(args.cip).items()}
    xw = load_cip_soc_crosswalk(args.crosswalk)
    r_socs = soc_profiles(load_onet_interests(args.interests))
    bmap = load_basic_interest_map(args.basic_map)
    areas = tuple(bmap)
    bkeys = tuple(basic_interest_key(a) for a in areas)
    b_socs = soc_profiles(load_onet_basic_interests(args.basic_interests, areas), areas)

    cips, R, B = [], [], []
    for c in sorted(offered_cips(args.completions)):
        pr = occupation_profile(c, xw, r_socs)
        pb = occupation_profile(c, xw, b_socs, bkeys, areas)
        if pr.vector and pb.vector:
            cips.append(c)
            R.append([pr.vector[k] for k in RIASEC_KEYS])
            B.append([pb.vector[k] for k in bkeys])
    R, B = np.array(R), np.array(B)
    fam = np.array([c[:2] for c in cips])
    spaces = {
        "S1 RIASEC": R,
        "S2 41 areas": B,
        "S3 both": np.hstack([R / math.sqrt(R.shape[1]), B / math.sqrt(B.shape[1])]),
    }
    agree = {name: knn_agreement(X, fam) for name, X in spaces.items()}
    sil = {name: silhouette(X, fam) for name, X in spaces.items()}
    d21 = agree["S2 41 areas"] - agree["S1 RIASEC"]
    d31 = agree["S3 both"] - agree["S1 RIASEC"]
    h1 = mean_se(d21)
    h2 = mean_se(d31)
    ok1, ok2 = h1[0] > 2 * h1[1], h2[0] > 2 * h2[1]

    L = [
        "# W-64 真實資料檢查：職業橋接能否區分領域",
        "",
        f"- 產生日期：{date.today().isoformat()}",
        "- 資料：IPEDS 2025（award level 7／17／18，第一主修）、CIP 2020–SOC 2018 對照表、"
        "O\\*NET 31.0（Career Interest Types、Specific Interest Areas）",
        f"- 單位：{len(cips)} 個 6 碼 CIP（全領域，不限 MVP），{len(set(fam))} 個 2 碼族群",
        "- 預先登記：見 `scripts/check_domain_layer.py` 檔頭",
        "",
        f"## 判定：H-D1 {'✅' if ok1 else '❌'} · H-D2 {'✅' if ok2 else '❌'}",
        "",
        "| 空間 | 最近 10 個 CIP 中同族群的比例（M1） | 族群輪廓係數（silhouette） |",
        "|---|---|---|",
    ]
    for name in spaces:
        m, se = mean_se(agree[name])
        L.append(f"| {name} | {m:.3f} ± {se:.3f} | {sil[name]:.3f} |")
    L += [
        "",
        f"- 配對差（跨 CIP）：S2 − S1 {h1[0]:+.3f} ± {h1[1]:.3f}；"
        f"S3 − S1 {h2[0]:+.3f} ± {h2[1]:.3f}",
        "",
        "## 依 2 碼族群（M1，只報告；CIP 數 ≥ 5 的族群）",
        "",
        "| 族群 | CIP 數 | S1 | S2 | S3 |",
        "|---|---|---|---|---|",
    ]
    for f in sorted(set(fam)):
        sel = fam == f
        if sel.sum() >= 5:
            L.append(f"| {f} | {int(sel.sum())} | " + " | ".join(
                f"{agree[n][sel].mean():.2f}" for n in spaces) + " |")  # fmt: skip
    L += ["", "## 例子：最近的 5 個 CIP（只報告）", ""]
    for c in EXAMPLES:
        if c not in cips:
            continue
        i = cips.index(c)
        L.append(f"### {c} {titles.get(c, '')}")
        L.append("")
        for name in ("S1 RIASEC", "S3 both"):
            X = spaces[name]
            d = np.sqrt(((X - X[i]) ** 2).sum(1))
            d[i] = np.inf
            nn = np.argsort(d, kind="stable")[:5]
            L.append(f"- {name}：" + "；".join(f"{cips[j]} {titles.get(cips[j], '')}" for j in nn))
        L.append("")
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(cips)} CIPs; H-D1 {ok1}; H-D2 {ok2} -> {args.out}")


if __name__ == "__main__":
    main()
