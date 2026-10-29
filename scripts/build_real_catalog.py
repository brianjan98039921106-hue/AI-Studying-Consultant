"""Build the real programme catalogue v0 (W-77) from the committed reference tables.

    python scripts/build_real_catalog.py                    # master's, >= 10 awards
    python scripts/build_real_catalog.py --levels 7 18 --min-awards 1

Writes data/catalog/real_v0.jsonl (gitignored: ~17k programmes, rebuilt in seconds)
and prints coverage. Load it with aisc.catalog.load_programs.
"""

import argparse
from collections import Counter
from pathlib import Path

from aisc.catalog.real import load_real_catalog_v0, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="+", default=["7"], help="IPEDS award levels")
    ap.add_argument("--min-awards", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "catalog" / "real_v0.jsonl")
    args = ap.parse_args()
    cat = load_real_catalog_v0(ROOT / "data" / "reference", args.levels, args.min_awards)
    write_jsonl(cat.programs, args.out)
    src = Counter(p.attribute_source for p in cat.programs)
    progs = cat.programs
    n_cip, n_school = len({p.cip for p in progs}), len({p.school for p in progs})
    print(f"programmes: {len(progs)} (rows at levels {args.levels}: {cat.n_rows_considered})")
    print(f"below min_awards={args.min_awards}: {cat.n_below_min_awards}")
    print(f"CIPs: {n_cip}; schools: {n_school}")
    print(f"profile source: {dict(src)}; CIPs skipped (no profile): {list(cat.skipped_no_profile)}")
    print(f"unmeasured (placeholder 0.5): {list(progs[0].unmeasured) if progs else []}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
