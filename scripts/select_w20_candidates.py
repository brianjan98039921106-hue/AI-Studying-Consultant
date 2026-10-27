"""Select candidate programmes for the W-20 gold standard and W-76 stimuli.

W-20 needs ~30 real programmes rated by two raters; W-76 needs cross-field
programme descriptions whose layer differences vary enough (B-C2-18, F99-F101).
One set serves both. Rules (fixed before looking at the selection):
  universe   6-digit CIPs from the all-fields reference (rollout v3) with a
             non-academic 41-area profile (not teacher-only, A-C0-10) and at least
             500 first-major master's awards or 200 professional-doctorate awards
  coverage   for each of the 12 area groups, the eligible CIP with the most awards
             among those whose highest-scoring area belongs to that group (primary
             group, F67); groups with no eligible CIP are skipped
  spread     fill to N = 30 by farthest-point selection (maximise the minimum
             Euclidean distance to the selected set) on the 41-area vectors; ties by
             awards
  schools    for each CIP, the 3 schools with the most awards at the chosen level,
             as rater options (W-20 picks one programme page per CIP)
Outputs data/reference/w20_candidates.csv and results/w20-candidates.md.

    python scripts/select_w20_candidates.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from aisc.dimensions import BASIC_INTEREST_KEYS
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
N = 30
MIN_MASTERS, MIN_PROF_DOC = 500, 200
LABELS_ZH = {}


def _read(name):
    with open(REF / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()
    cips = {r["cip"]: r for r in _read("cip2020_all.csv")}
    prof = {r["cip"]: r for r in _read("occupation_basic_interests_all_nonacademic.csv")}
    elig = []
    for c, r in cips.items():
        p = prof[c]
        if p[BASIC_INTEREST_KEYS[0]] == "" or p["fallback_teacher_only"] == "True":
            continue
        m, d = int(r["awards_level_7"]), int(r["awards_level_18"])
        if m >= MIN_MASTERS or d >= MIN_PROF_DOC:
            elig.append(c)
    X = {c: np.array([float(prof[c][k]) for k in BASIC_INTEREST_KEYS]) for c in elig}
    awards = {c: int(cips[c]["awards_level_7"]) + int(cips[c]["awards_level_18"]) for c in elig}
    level = {c: "18" if int(cips[c]["awards_level_18"]) > int(cips[c]["awards_level_7"])
             else "7" for c in elig}  # fmt: skip
    group = {c: AREA_GROUPS[BASIC_INTEREST_KEYS[int(np.argmax(X[c]))]][0] for c in elig}
    chosen: list[str] = []
    for g in sorted(GROUP_LABELS):
        cands = [c for c in elig if group[c] == g]
        if cands:
            chosen.append(max(cands, key=lambda c: (awards[c], c)))
    while len(chosen) < min(args.n, len(elig)):
        rest = [c for c in elig if c not in chosen]
        dmin = {c: min(np.linalg.norm(X[c] - X[s]) for s in chosen) for c in rest}
        chosen.append(max(rest, key=lambda c: (round(dmin[c], 6), awards[c], c)))

    schools = defaultdict(list)
    for r in _read("ipeds_graduate_all.csv"):
        if r["cip"] in chosen and r["award_level"] == level[r["cip"]]:
            schools[r["cip"]].append((int(r["awards"]), r["institution"], r["state"]))
    rows = []
    for c in chosen:
        top = sorted(schools[c], reverse=True)[:3]
        areas = sorted(BASIC_INTEREST_KEYS, key=lambda k: -float(prof[c][k]))[:3]
        rows.append({
            "cip": c, "title": cips[c]["title"], "family": cips[c]["family"],
            "primary_group": group[c], "award_level": level[c], "awards": awards[c],
            "top_areas": " ".join(areas),
            "school_options": " | ".join(f"{n} ({s}, {a})" for a, n, s in top),
        })  # fmt: skip
    with open(REF / "w20_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    D = np.array([[np.linalg.norm(X[a] - X[b]) for b in chosen] for a in chosen])
    iu = np.triu_indices(len(chosen), 1)
    rand = np.random.default_rng(0).choice(elig, len(chosen), replace=False)
    Dr = np.array([[np.linalg.norm(X[a] - X[b]) for b in rand] for a in rand])
    L = [
        "# W-20／W-76 候選學程（全領域、跨領域分散）",
        "",
        f"- 產生日期：{date.today().isoformat()} · 指令：`python scripts/select_w20_candidates.py`",
        f"- 合格 CIP：{len(elig)}（非學術剖面、碩士 ≥ {MIN_MASTERS} 或專業博士 ≥ {MIN_PROF_DOC}）；"
        f"選出 {len(chosen)} 個；涵蓋 {len({group[c] for c in chosen})} / 12 個領域群",
        f"- 41 領域向量的兩兩距離：中位數 {np.median(D[iu]):.3f}、最小 {D[iu].min():.3f}"
        f"（同數量隨機抽樣：中位數 {np.median(Dr[iu]):.3f}、最小 {Dr[iu].min():.3f}）",
        "- 每個 CIP 列出學位數最多的 3 所學校，供評分者挑一個學程頁面；規則見腳本檔頭",
        "",
        "| CIP | 名稱 | 群 | 學位層級 | 學位數 | 前 3 領域 | 學校選項 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['cip']} | {r['title']} | {r['primary_group']} "
                 f"{GROUP_LABELS[r['primary_group']][0]} | {r['award_level']} | {r['awards']:,} | "
                 f"{r['top_areas'].replace('bi_', '')} | {r['school_options']} |")  # fmt: skip
    (ROOT / "results" / "w20-candidates.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(elig)} eligible, {len(chosen)} chosen")


if __name__ == "__main__":
    main()
