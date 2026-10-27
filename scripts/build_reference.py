"""Build the C0 reference tables from the official files (W-14, W-15).

Inputs are the files exactly as downloaded (zip/xlsx/csv/txt). Outputs go to
data/reference/ with a SOURCES.json provenance record (URL, retrieval date,
SHA-256 per input) and a summary report in results/.

    python scripts/build_reference.py \\
        --cip CIPCode2020.xlsx --crosswalk CIP2020_SOC2018_Crosswalk.xlsx \\
        --interests Interests.csv --completions C2024_A.zip --directory HD2024.zip \\
        --retrieved-at 2026-10-26 --onet-release 31.0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from aisc.catalog.occupation import (
    RIASEC,
    RIASEC_KEYS,
    basic_interest_key,
    load_basic_interest_map,
    load_cip_soc_crosswalk,
    load_onet_basic_interests,
    load_onet_interests,
    occupation_profile,
    soc_profiles,
)
from aisc.catalog.reference import (
    file_provenance,
    load_cip2020,
    load_ipeds_completions,
    load_ipeds_directory,
    load_scope,
    school_cip_rows,
    select_cips,
)

ROOT = Path(__file__).resolve().parents[1]
CIP_URL = "https://nces.ed.gov/ipeds/cipcode/resources.aspx?y=56"
XWALK_URL = "https://nces.ed.gov/IPEDS/CIPCODE/Files/CIP2020_SOC2018_Crosswalk.xlsx"
ONET_URL = "https://www.onetcenter.org/database.html"
IPEDS_URL = "https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _year(name: str) -> str:
    m = re.search(r"(?:C|HD)(\d{4})", name.upper())
    if not m:
        raise ValueError(f"cannot read the IPEDS year from {name!r}")
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("cip", "crosswalk", "interests", "completions", "directory"):
        ap.add_argument(f"--{a}", type=Path, required=True)
    ap.add_argument("--retrieved-at", required=True, help="ISO date the files were downloaded")
    ap.add_argument("--onet-release", required=True, help="O*NET database release, e.g. 31.0")
    ap.add_argument(
        "--basic-interests",
        type=Path,
        default=None,
        help="O*NET specific_interest_areas.csv (W-64)",
    )
    ap.add_argument(
        "--basic-map",
        type=Path,
        default=None,
        help="O*NET specific_interest_areas_to_career_interest_types.csv",
    )
    ap.add_argument(
        "--basic-retrieved-at",
        default=None,
        help="download date of the two basic-interest files (default --retrieved-at)",
    )
    ap.add_argument("--scope", type=Path, default=ROOT / "data" / "reference" / "mvp_scope.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "reference")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "reference-w14-w15.md")
    args = ap.parse_args()

    year = _year(args.completions.name)
    if _year(args.directory.name) != year:
        raise SystemExit("completions and directory files must be from the same IPEDS year")

    scope = load_scope(args.scope)
    cip = load_cip2020(args.cip)
    fam = select_cips(cip, scope)
    comps = load_ipeds_completions(args.completions, fam, scope.award_level, scope.majornum)
    directory = load_ipeds_directory(args.directory)
    school_rows = school_cip_rows(comps, directory, cip, fam, year)

    xw = load_cip_soc_crosswalk(args.crosswalk)
    socs = soc_profiles(load_onet_interests(args.interests))
    offered = sorted({c.cip for c in comps})
    profiles = {c: occupation_profile(c, xw, socs) for c in sorted(fam)}

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "cip2020_mvp.csv", [
        {"cip": c, "title": cip[c].title, "family": f, "offered_masters": c in offered,
         "definition": cip[c].definition}
        for c, f in fam.items()
    ])  # fmt: skip
    _write_csv(args.out / "ipeds_masters_mvp.csv", school_rows)
    _write_csv(args.out / "cip_soc_mvp.csv", [
        {"cip": c, "soc": s, "soc_has_onet": s in socs}
        for c in sorted(fam) for s in sorted(xw.get(c, set()))
    ])  # fmt: skip
    used = sorted({s for c in fam for s in xw.get(c, set()) if s in socs})
    _write_csv(args.out / "soc_profiles_mvp.csv", [
        {"soc": s, **{n.lower(): f"{socs[s][n]:.4f}" for n in RIASEC}} for s in used
    ])  # fmt: skip
    _write_csv(args.out / "occupation_profiles_mvp.csv", [
        {"cip": c, **{k: f"{p.vector[k]:.4f}" if p.vector else "" for k in RIASEC_KEYS},
         "n_soc": len(p.socs), "n_soc_with_onet": len(p.socs_with_onet),
         "missing_soc": " ".join(p.missing)}
        for c, p in profiles.items()
    ])  # fmt: skip
    extra_inputs = []
    if (args.basic_interests is None) != (args.basic_map is None):
        raise SystemExit("pass both --basic-interests and --basic-map, or neither")
    if args.basic_interests is not None:
        bmap = load_basic_interest_map(args.basic_map)
        areas = tuple(bmap)
        bkeys = tuple(basic_interest_key(a) for a in areas)
        b_occ = load_onet_basic_interests(args.basic_interests, areas)
        bsocs = soc_profiles(b_occ, areas)
        # base rate per area: mean over all O*NET occupations, on [0, 1] (prior for areas
        # a student did not answer, dimensions-v2 §3)
        base = {a: (sum(o[a] for o in b_occ.values()) / len(b_occ) - 1) / 6 for a in areas}
        _write_csv(args.out / "basic_interest_map.csv", [
            {"key": k, "area": a, "riasec": " ".join(bmap[a]), "occupation_mean": f"{base[a]:.4f}"}
            for k, a in zip(bkeys, areas, strict=True)
        ])  # fmt: skip
        bused = sorted({s for c in fam for s in xw.get(c, set()) if s in bsocs})
        _write_csv(args.out / "soc_basic_interests_mvp.csv", [
            {"soc": s, **{k: f"{bsocs[s][a]:.4f}" for k, a in zip(bkeys, areas, strict=True)}}
            for s in bused
        ])  # fmt: skip
        bprof = {c: occupation_profile(c, xw, bsocs, bkeys, areas) for c in sorted(fam)}
        _write_csv(args.out / "occupation_basic_interests_mvp.csv", [
            {"cip": c, **{k: f"{p.vector[k]:.4f}" if p.vector else "" for k in bkeys},
             "n_soc": len(p.socs), "n_soc_with_onet": len(p.socs_with_onet)}
            for c, p in bprof.items()
        ])  # fmt: skip
        b_date = args.basic_retrieved_at or args.retrieved_at
        extra_inputs = [
            ("onet_specific_interest_areas", args.basic_interests, ONET_URL, b_date),
            ("onet_specific_interest_areas_map", args.basic_map, ONET_URL, b_date),
        ]
    sources = {
        "built": date.today().isoformat(),
        "scope_version": scope.version,
        "ipeds_year": year,
        "onet_release": args.onet_release,
        "inputs": {
            key: file_provenance(path, url, args.retrieved_at).to_dict()
            for key, path, url in [
                ("cip2020", args.cip, CIP_URL),
                ("cip_soc_crosswalk", args.crosswalk, XWALK_URL),
                ("onet_interests", args.interests, ONET_URL),
                ("ipeds_completions", args.completions, IPEDS_URL),
                ("ipeds_directory", args.directory, IPEDS_URL),
            ]
        },
    }
    for key, path, url, when in extra_inputs:
        sources["inputs"][key] = file_provenance(path, url, when).to_dict()
    (args.out / "SOURCES.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")

    # --- report ---------------------------------------------------------------------
    by_fam_cips, by_fam_schools, by_fam_awards = Counter(), defaultdict(set), Counter()
    for f in fam.values():
        by_fam_cips[f] += 1
    for r in school_rows:
        by_fam_schools[r["family"]].add(r["unitid"])
        by_fam_awards[r["family"]] += r["masters_awards"]
    n_schools = len({r["unitid"] for r in school_rows})
    with_vec = [p for c, p in profiles.items() if p.vector and c in offered]
    no_vec = [c for c in offered if not profiles[c].vector]
    L = [
        "# 參考資料建置報告（W-14、W-15）",
        "",
        f"- 產生日期：{date.today().isoformat()} · IPEDS {year} · O*NET {args.onet_release} · "
        f"範圍 {scope.version}",
        f"- 來源與檔案雜湊：`data/reference/SOURCES.json`（下載日期 {args.retrieved_at}）",
        "",
        "## W-14：範圍內 CIP 與學校",
        "",
        f"- 範圍內 6 碼 CIP：{len(fam)}；其中有學校頒發碩士學位的：{len(offered)}",
        f"- 「學校 × CIP」列數：{len(school_rows)}；不同學校：{n_schools}（B-C0-7：還不是學程）",
        f"- 碩士學位總數（第一主修）：{sum(r['masters_awards'] for r in school_rows):,}",
        "",
        "| 族群 | CIP 數 | 有碩士的學校數 | 碩士學位數 |",
        "|---|---|---|---|",
    ]
    for f in scope.families:
        L.append(f"| {f} | {by_fam_cips[f]} | {len(by_fam_schools[f])} | {by_fam_awards[f]:,} |")
    L += [
        "",
        "## W-15：職業橋接",
        "",
        f"- 有碩士學程的 CIP 中，取得職業剖面的：{len(with_vec)} / {len(offered)}",
        f"- 沒有任何 SOC 有 O*NET 資料（或對照表為 NO MATCH）的 CIP：{', '.join(no_vec) or '無'}",
        f"- SOC 覆蓋率（有 O*NET 資料的 SOC ÷ 對照的 SOC）中位數："
        f"{statistics.median([p.coverage for p in with_vec]) if with_vec else float('nan'):.2f}",
        "",
        "### 區辨度檢查（B-C0-2：多對多對照會把剖面稀釋成平均值）",
        "",
        "| 維度 | 平均 | 標準差（跨 CIP） | 最小 | 最大 |",
        "|---|---|---|---|---|",
    ]
    for k in RIASEC_KEYS:
        v = [p.vector[k] for p in with_vec]
        if len(v) > 1:
            L.append(f"| {k} | {statistics.mean(v):.3f} | {statistics.stdev(v):.3f} | "
                     f"{min(v):.3f} | {max(v):.3f} |")  # fmt: skip
    L.append("")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(fam)} CIPs in scope, {len(offered)} offered, {len(school_rows)} school x CIP rows")


if __name__ == "__main__":
    main()
