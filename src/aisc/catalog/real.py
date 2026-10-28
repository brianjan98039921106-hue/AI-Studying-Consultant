"""Real programme catalogue v0 (W-77): IPEDS school x CIP rows joined with the C0.2
occupation-bridge profiles.

What is real: every programme exists in IPEDS 2025 Completions (school, CIP, award
level, number of awards); interest (RIASEC) and the 41 interest areas come from
O*NET 31.0 through the CIP -> SOC crosswalk (C0.2, data/reference/SOURCES_all.json).

What is not measured yet: the intensity and orientation layers need curriculum
evidence (C0.3, W-21). They get one placeholder value shared by the whole catalogue
and are listed in `Program.unmeasured` (A-C0-15). Under the layer-balanced v2 scorer a
constant layer adds the same amount to every programme, so it cannot change the
ranking (tested); it does inflate the absolute relevance and must not be shown as a
reason.

Known limits: programmes of one CIP share the same vector (B-C0-1), and a "school x
CIP" row may hide several real programmes or none with that exact name (B-C0-7).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from aisc.contracts import Program
from aisc.dimensions import dims_keys

DIMS = "v2"
AWARD_LEVEL_NAMES = {
    "6": "Post-baccalaureate certificate",
    "7": "Master's",
    "8": "Post-master's certificate",
    "17": "Doctor's (research/scholarship)",
    "18": "Doctor's (professional practice)",
    "19": "Doctor's (other)",
}
RESEARCH_DOCTORATE = "17"  # keeps academic SOCs in its profile (A-C0-10)


def unmeasured_keys() -> tuple[str, ...]:
    """v2 dimensions that the occupation bridge does not measure (intensity, orientation)."""
    from aisc.dimensions import DIMENSION_SETS

    return tuple(d.key for d in DIMENSION_SETS[DIMS] if d.layer in ("intensity", "orientation"))


@dataclass(frozen=True)
class RealCatalog:
    programs: list[Program]
    n_rows_considered: int  # rows at the requested award levels
    n_below_min_awards: int
    skipped_no_profile: tuple[str, ...]  # CIPs without an occupation profile
    n_fallback: int = 0  # programmes whose profile is the low-confidence series fallback


def build_real_catalog_v0(
    completions: Iterable[Mapping[str, str]],
    profiles: Mapping[str, Mapping[str, Mapping[str, float]]],
    titles: Mapping[str, str],
    source_url: str,
    retrieved_at: str,
    award_levels: Iterable[str] = ("7",),
    min_awards: int = 10,
    placeholder: float = 0.5,
) -> RealCatalog:
    """Join IPEDS rows with occupation profiles.

    `completions`: rows with unitid, institution, cip, award_level, awards.
    `profiles`: {"nonacademic" | "academic": {cip: {v2 interest/area key: value}}};
    research doctorates use "academic", every other level "nonacademic" (A-C0-10).
    An optional "fallback" table (the CIP's 4-digit group or 2-digit series mean,
    F142-F143) fills CIPs with no SOC link, for non-research levels only; those
    programmes get attribute_source="occupation_fallback" (low confidence).
    `min_awards`: drop school x CIP rows with fewer awards in the IPEDS year (A-C0-16).
    """
    if not 0 <= placeholder <= 1:
        raise ValueError("placeholder must lie in [0, 1]")
    if min_awards < 1:
        raise ValueError("min_awards must be >= 1")
    levels = {str(x) for x in award_levels}
    unknown = levels - set(AWARD_LEVEL_NAMES)
    if unknown:
        raise ValueError(f"unknown IPEDS award levels: {sorted(unknown)}")
    keys = dims_keys(DIMS)
    missing_layer = set(unmeasured_keys())
    measured = [k for k in keys if k not in missing_layer]
    programs: list[Program] = []
    n_rows = n_small = n_fb = 0
    skipped: set[str] = set()
    seen: set[str] = set()
    for row in completions:
        level = str(row["award_level"]).strip()
        if level not in levels:
            continue
        n_rows += 1
        if int(row["awards"]) < min_awards:
            n_small += 1
            continue
        cip = row["cip"].strip()
        table = profiles["academic" if level == RESEARCH_DOCTORATE else "nonacademic"]
        prof = table.get(cip)
        source = "occupation"
        if prof is None and level != RESEARCH_DOCTORATE and cip in profiles.get("fallback", {}):
            prof, source = profiles["fallback"][cip], "occupation_fallback"
            n_fb += 1
        if prof is None:
            skipped.add(cip)
            continue
        absent = [k for k in measured if k not in prof]
        if absent:
            raise ValueError(f"profile for {cip} lacks {absent[:3]}…")
        pid = f"ipeds-{row['unitid'].strip()}-{cip}-{level}"
        if pid in seen:
            raise ValueError(f"duplicate IPEDS row for {pid}")
        seen.add(pid)
        attrs = {k: float(prof[k]) for k in measured}
        attrs.update({k: placeholder for k in missing_layer})
        programs.append(
            Program(
                program_id=pid,
                school=row["institution"].strip(),
                degree_name=f"{titles.get(cip, cip)} ({AWARD_LEVEL_NAMES[level]})",
                cip=cip,
                attributes=attrs,
                synthetic=False,
                attribute_source=source,
                source_url=source_url,
                retrieved_at=retrieved_at,
                dims=DIMS,
                unmeasured=tuple(k for k in keys if k in missing_layer),
            )
        )
    return RealCatalog(programs, n_rows, n_small, tuple(sorted(skipped)), n_fb)


def _read_profiles(path: Path) -> dict[str, dict[str, float]]:
    """{cip: values}; rows with blank values (CIP without a SOC link) are left out."""
    keys = set(dims_keys(DIMS))
    out: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            vals = {k: v for k, v in r.items() if k in keys}
            if all(str(v).strip() for v in vals.values()):
                out[r["cip"]] = {k: float(v) for k, v in vals.items()}
    return out


def _merge(*tables: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    common = set.intersection(*(set(t) for t in tables))  # a CIP needs every table
    return {cip: {k: v for t in tables for k, v in t[cip].items()} for cip in common}


def load_real_catalog_v0(
    reference_dir: str | Path = "data/reference",
    award_levels: Iterable[str] = ("7",),
    min_awards: int = 10,
    placeholder: float = 0.5,
) -> RealCatalog:
    """Build the catalogue from the committed reference tables (no network access)."""
    ref = Path(reference_dir)
    sources = json.loads((ref / "SOURCES_all.json").read_text(encoding="utf-8"))["inputs"]
    ipeds = sources["ipeds_completions"]
    profiles = {
        "nonacademic": _merge(
            _read_profiles(ref / "occupation_profiles_all_nonacademic.csv"),
            _read_profiles(ref / "occupation_basic_interests_all_nonacademic.csv"),
        ),
        "academic": _merge(
            _read_profiles(ref / "occupation_profiles_all.csv"),
            _read_profiles(ref / "occupation_basic_interests_all.csv"),
        ),
        "fallback": _read_profiles(ref / "occupation_fallback_all_nonacademic.csv"),
    }
    with open(ref / "cip2020_all.csv", encoding="utf-8", newline="") as fh:
        titles = {r["cip"]: r["title"] for r in csv.DictReader(fh)}
    with open(ref / "ipeds_graduate_all.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return build_real_catalog_v0(
        rows, profiles, titles, ipeds["source_url"], ipeds["retrieved_at"],
        award_levels, min_awards, placeholder,
    )  # fmt: skip


def write_jsonl(programs: Iterable[Program], path: str | Path) -> None:
    """Write programmes in the format `catalog.load_programs` reads."""
    from dataclasses import asdict

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in programs:
            rec = asdict(p)
            rec["unmeasured"] = list(p.unmeasured)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = [
    "AWARD_LEVEL_NAMES",
    "RealCatalog",
    "build_real_catalog_v0",
    "load_real_catalog_v0",
    "unmeasured_keys",
    "write_jsonl",
]
