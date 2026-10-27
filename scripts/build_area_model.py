"""Build the correlated interest-area model (architecture-review §2.1, THEORY
"領域相關補全"; opt-in `area_model="correlated"` in profiling).

Mean and Ledoit-Wolf covariance of the 41 O*NET specific interest areas and the 12
area-group maxima (dimensions-v2 §2) over all O*NET 31.0 occupations, on [0, 1].
Writes src/aisc/profiling/interest_area_model.json (package data) with provenance.

    python scripts/build_area_model.py --basic-interests specific_interest_areas.csv \\
        --basic-map specific_interest_areas_to_career_interest_types.csv --retrieved-at 2026-10-27
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from aisc.catalog.occupation import (
    basic_interest_key,
    load_basic_interest_map,
    load_onet_basic_interests,
)
from aisc.catalog.reference import file_provenance
from aisc.dimensions import BASIC_INTEREST_KEYS
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
ONET_URL = "https://www.onetcenter.org/database.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basic-interests", type=Path, required=True)
    ap.add_argument("--basic-map", type=Path, required=True)
    ap.add_argument("--retrieved-at", required=True)
    default_out = ROOT / "src" / "aisc" / "profiling" / "interest_area_model.json"
    ap.add_argument("--out", type=Path, default=default_out)
    args = ap.parse_args()
    bmap = load_basic_interest_map(args.basic_map)
    areas = tuple(bmap)
    if tuple(basic_interest_key(a) for a in areas) != BASIC_INTEREST_KEYS:
        raise SystemExit("area order differs from BASIC_INTEREST_KEYS")
    occ = load_onet_basic_interests(args.basic_interests, areas)
    T = np.array([[(occ[c][a] - 1) / 6 for a in areas] for c in sorted(occ)])
    groups = sorted(GROUP_LABELS)
    G = np.column_stack([
        T[:, [i for i, k in enumerate(BASIC_INTEREST_KEYS) if AREA_GROUPS[k][0] == g]].max(1)
        for g in groups
    ])  # fmt: skip
    Z = np.hstack([T, G])
    lw = LedoitWolf().fit(Z)
    model = {
        "description": "Mean and Ledoit-Wolf covariance of 41 interest areas + 12 group "
        "maxima over O*NET occupations, (OI - 1) / 6 scale",
        "variables": list(BASIC_INTEREST_KEYS) + [f"group_{g}" for g in groups],
        "n_occupations": int(len(T)),
        "shrinkage": round(float(lw.shrinkage_), 6),
        "mean": [round(float(v), 6) for v in Z.mean(0)],
        "covariance": [[round(float(v), 8) for v in row] for row in lw.covariance_],
        "inputs": {
            "onet_specific_interest_areas": file_provenance(
                args.basic_interests, ONET_URL, args.retrieved_at).to_dict(),
            "onet_specific_interest_areas_map": file_provenance(
                args.basic_map, ONET_URL, args.retrieved_at).to_dict(),
        },
        "evidence": "results/interest-area-imputation-findings.md",
    }  # fmt: skip
    args.out.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"{len(T)} occupations, shrinkage {lw.shrinkage_:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
