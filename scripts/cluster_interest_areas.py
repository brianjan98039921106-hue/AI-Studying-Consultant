"""Group O*NET's 41 specific interest areas into broad clusters for the first
stage of the two-stage questionnaire (W-64 option B, W-66).

Data: O*NET 31.0 specific_interest_areas.csv, OI scale (1-7), all 891 occupations,
each occupation weighted equally.

PRE-REGISTERED (written and committed before the first run, 2026-10-27):
  Similarity between two areas = Pearson correlation of their OI ratings across
  occupations; distance = 1 - r.
  Method: agglomerative clustering, average linkage, on that distance.
  Number of clusters k: the k in 6..14 with the highest mean silhouette
  (computed on the same distance); ties -> the smaller k.
  Stability: 200 bootstrap resamples of occupations (seed 2026); recluster at the
  chosen k; adjusted Rand index (ARI) against the full-data clustering.
  Acceptance: mean ARI >= 0.60. If not met, the clustering is reported as
  unstable and the fallback grouping is used instead: each area under the first
  RIASEC type it links to in specific_interest_areas_to_career_interest_types.csv
  (6 groups).
  Report only: members of each cluster with their RIASEC links; silhouette for
  every k; per-area silhouette; the fallback grouping's silhouette.
  Cluster labels are written afterwards for readability and do not affect the rule.

    python scripts/cluster_interest_areas.py --areas specific_interest_areas.csv \\
        --map specific_interest_areas_to_career_interest_types.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from aisc.catalog.occupation import (
    basic_interest_key,
    load_basic_interest_map,
    load_onet_basic_interests,
)

ROOT = Path(__file__).resolve().parents[1]
K_RANGE = range(6, 15)
N_BOOT = 200
SEED = 2026
ARI_MIN = 0.60


def distance_matrix(X: np.ndarray) -> np.ndarray:
    """1 - Pearson correlation between columns (areas)."""
    r = np.corrcoef(X, rowvar=False)
    d = 1.0 - r
    np.fill_diagonal(d, 0.0)
    return np.clip((d + d.T) / 2, 0.0, 2.0)


def cluster(d: np.ndarray, k: int) -> np.ndarray:
    z = linkage(squareform(d, checks=False), method="average")
    return fcluster(z, t=k, criterion="maxclust")


def silhouette_samples(d: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels))
    for i in range(len(labels)):
        same = (labels == labels[i]) & (np.arange(len(labels)) != i)
        if not same.any():
            continue  # singleton: silhouette 0 by convention
        a = d[i, same].mean()
        b = min(d[i, labels == c].mean() for c in set(labels) if c != labels[i])
        out[i] = (b - a) / max(a, b)
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    from math import comb

    ca, cb = (
        {x: i for i, x in enumerate(sorted(set(a)))},
        {x: i for i, x in enumerate(sorted(set(b)))},
    )
    table = np.zeros((len(ca), len(cb)), int)
    for x, y in zip(a, b, strict=True):
        table[ca[x], cb[y]] += 1
    s_ij = sum(comb(int(v), 2) for v in table.ravel())
    s_a = sum(comb(int(v), 2) for v in table.sum(1))
    s_b = sum(comb(int(v), 2) for v in table.sum(0))
    n = comb(len(a), 2)
    expected = s_a * s_b / n
    max_index = (s_a + s_b) / 2
    return 1.0 if max_index == expected else (s_ij - expected) / (max_index - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument(
        "--out-json", type=Path, default=ROOT / "data" / "reference" / "interest_area_clusters.json"
    )
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "interest-area-clusters.md")
    args = ap.parse_args()  # fmt: skip

    bmap = load_basic_interest_map(args.map)
    areas = tuple(bmap)
    ratings = load_onet_basic_interests(args.areas, areas)
    occs = sorted(ratings)
    X = np.array([[ratings[o][a] for a in areas] for o in occs])
    d = distance_matrix(X)

    sil = {k: float(silhouette_samples(d, cluster(d, k)).mean()) for k in K_RANGE}
    k_best = max(K_RANGE, key=lambda k: (round(sil[k], 12), -k))
    labels = cluster(d, k_best)

    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(occs), len(occs))
        aris.append(ari(labels, cluster(distance_matrix(X[idx]), k_best)))
    aris = np.array(aris)
    stable = aris.mean() >= ARI_MIN

    fallback = np.array([["Realistic", "Investigative", "Artistic", "Social", "Enterprising",
                          "Conventional"].index(bmap[a][0]) + 1 for a in areas])  # fmt: skip
    sil_fb = float(silhouette_samples(d, fallback).mean())
    final = labels if stable else fallback
    per_area = silhouette_samples(d, final)

    groups: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(final):
        groups[int(g)].append(i)
    # order clusters by their most common RIASEC link, then by size, for readability
    order = sorted(groups, key=lambda g: (-len(groups[g]), areas[groups[g][0]]))
    spec = {
        "version": "clusters-v1",
        "built": date.today().isoformat(),
        "source": "O*NET 31.0 specific_interest_areas.csv (OI), 891 occupations",
        "rule": "scripts/cluster_interest_areas.py (pre-registered)",
        "method": "average-linkage clustering on 1 - Pearson r" if stable else
                  "fallback: first linked RIASEC type (clustering unstable)",
        "k": int(len(groups)),
        "mean_silhouette": float(per_area.mean()),
        "bootstrap_mean_ari": float(aris.mean()),
        "clusters": [
            {"id": n + 1,
             "areas": [{"key": basic_interest_key(areas[i]), "name": areas[i],
                        "riasec": list(bmap[areas[i]])} for i in groups[g]]}
            for n, g in enumerate(order)
        ],
    }  # fmt: skip
    args.out_json.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", "utf-8")

    L = [
        "# 41 個細分興趣領域的分群（W-64 方案 B）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 資料：O\\*NET 31.0 Specific Interest Areas"
        f"（{len(occs)} 個職業，等權）",
        "- 預先登記：見 `scripts/cluster_interest_areas.py` 檔頭",
        "",
        "## 判定",
        "",
        f"- 依輪廓係數選出 k = **{k_best}**（6–14 中最高）",
        f"- 自助法穩定度：ARI 平均 {aris.mean():.3f}（第 5–95 百分位 {np.percentile(aris, 5):.3f}–"
        f"{np.percentile(aris, 95):.3f}），門檻 {ARI_MIN} → "
        + (
            "✅ 穩定，採用分群結果"
            if stable
            else "❌ 不穩定，改用備案（依第一個 RIASEC 類型分 6 群）"
        ),
        f"- 備案分組（RIASEC）的輪廓係數：{sil_fb:.3f}",
        "",
        "## 各 k 的平均輪廓係數",
        "",
        "| k | " + " | ".join(str(k) for k in K_RANGE) + " |",
        "|---" * (len(K_RANGE) + 1) + "|",
        "| 輪廓係數 | " + " | ".join(f"{sil[k]:.3f}" for k in K_RANGE) + " |",
        "",
        "## 採用的分組",
        "",
        "| 群 | 領域（RIASEC 連結；領域輪廓係數） |",
        "|---|---|",
    ]
    for n, g in enumerate(order):
        members = "；".join(f"{areas[i]}（{'/'.join(t[0] for t in bmap[areas[i]])}；"
                           f"{per_area[i]:.2f}）" for i in groups[g])  # fmt: skip
        L.append(f"| {n + 1} | {members} |")
    L.append("")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"k={k_best} stable={stable} mean ARI={aris.mean():.3f} -> {args.out_json}")


if __name__ == "__main__":
    main()
