"""Write series-mean fallback profiles for CIPs without an occupation match (B-C0-9).

Adopted by the pre-registered leave-one-out check (scripts/check_series_fallback.py,
results/series-fallback-findings.md). Input: the committed all-fields non-academic
reference tables; output: data/reference/occupation_fallback_all_nonacademic.csv with
one row per CIP of cip2020_all.csv that has no profile but has siblings with one.
Every row is a LOW-CONFIDENCE fallback (level "group" = same 4-digit group,
"series" = same 2-digit series), never a measured profile.

    python scripts/build_series_fallback.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from aisc.catalog.occupation import series_fallback
from aisc.dimensions import BASIC_INTEREST_KEYS

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
RIASEC = ["riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c"]


def _read(name):
    with open(REF / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ria = {r["cip"]: r for r in _read("occupation_profiles_all_nonacademic.csv")}
    are = {r["cip"]: r for r in _read("occupation_basic_interests_all_nonacademic.csv")}
    keys = RIASEC + list(BASIC_INTEREST_KEYS)
    ok = [c for c in ria if ria[c][RIASEC[0]] != "" and are.get(c, {}).get(keys[6], "") != ""]
    have = {c: {**{k: float(ria[c][k]) for k in RIASEC},
                **{k: float(are[c][k]) for k in BASIC_INTEREST_KEYS}} for c in ok}  # fmt: skip
    all_cips = [r["cip"] for r in _read("cip2020_all.csv")]
    missing = [c for c in all_cips if c not in have]
    fb = series_fallback(have, missing)
    rows = []
    for c in missing:
        if c in fb:
            prof, level = fb[c]
            n = sum(
                1
                for h in have
                if h[: 5 if level == "group" else 2] == c[: 5 if level == "group" else 2]
            )
            rows.append({"cip": c, "level": level, "n_siblings": n,
                         **{k: f"{prof[k]:.4f}" for k in keys}})  # fmt: skip
    out = REF / "occupation_fallback_all_nonacademic.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cip", "level", "n_siblings", *keys])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(missing)} CIPs without a profile; {len(rows)} fallbacks written "
          f"({sum(r['level'] == 'group' for r in rows)} group, "
          f"{sum(r['level'] == 'series' for r in rows)} series) -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
