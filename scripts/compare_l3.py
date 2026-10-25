"""Side-by-side comparison of L3 result files produced by run_l3_baseline.py.

    python scripts/compare_l3.py results/synthetic-l3-v1.md results/synthetic-l3-v1-continuous.md

Prints, for each truth function and prestige weight, the first two metric
columns (C1: RMSE, τ; C2: relevant rate, novel-relevant rate) of each row
that appears in every file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

Key = tuple[str, str, str]


def parse(path: Path) -> dict[Key, list[str]]:
    rows: dict[Key, list[str]] = {}
    truth = pw = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"## 真相函數：(\w+)", line):
            truth = m.group(1)
        elif m := re.match(r"### 名氣權重 π = ([\d.]+)", line):
            pw = m.group(1)
        elif line.startswith("## 敏感度"):
            break
        elif line.startswith("| ") and not line.startswith(("| 推薦器", "| 側寫", "|---")):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[(truth, pw, cells[0])] = cells[1:]
    return rows


def main(paths: list[str]) -> None:
    files = [Path(p) for p in paths]
    parsed = [parse(f) for f in files]
    keys = [k for k in parsed[0] if all(k in p for p in parsed[1:])]
    current = None
    for truth, pw, row in keys:
        if (truth, pw) != current:
            current = (truth, pw)
            print(f"\n--- truth={truth} π={pw}  |  " + "  |  ".join(f.stem for f in files))
        cells = "  |  ".join(" / ".join(p[(truth, pw, row)][:2]) for p in parsed)
        print(f"{row[:42]:<42} {cells}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
