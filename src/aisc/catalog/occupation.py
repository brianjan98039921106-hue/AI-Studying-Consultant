"""C0.2 occupation bridge v1: CIP 2020 -> SOC 2018 -> O*NET RIASEC (W-15).

x_occ covers only the six RIASEC dimensions. For one CIP code:
  1. its SOC 2018 codes from the NCES/BLS crosswalk ("NO MATCH" rows dropped);
  2. for each SOC code, the O*NET-SOC occupations under it (15-2051 ->
     15-2051.00, 15-2051.01, ...), whose OI interest ratings (1-7) are averaged;
  3. equal-weight mean over the SOC codes that have O*NET data (v1; v2 weights
     by employment), rescaled to [0, 1] as (v - 1) / 6 (A-C0-6).
Coverage is reported with every vector so that thin bridges are visible (B-C0-2).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from aisc.catalog.reference import normalize_cip, read_table

RIASEC = ("Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional")
RIASEC_KEYS = ("riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c")
_SOC = re.compile(r"^\d{2}-\d{4}$")
_ONET = re.compile(r"^(\d{2}-\d{4})\.\d{2}$")
NO_MATCH = "99-9999"


def load_cip_soc_crosswalk(path: str | Path) -> dict[str, set[str]]:
    """CIP 2020 code -> set of SOC 2018 detailed codes (NO MATCH dropped)."""
    rows = read_table(path, required=["CIP2020Code", "SOC2018Code"])
    out: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        soc = r["SOC2018Code"].strip()
        try:
            cip = normalize_cip(r["CIP2020Code"])
        except ValueError:
            continue
        if soc == NO_MATCH or "no match" in r.get("SOC2018Title", "").lower():
            out.setdefault(cip, set())
            continue
        if not _SOC.match(soc):
            raise ValueError(f"{path}: malformed SOC code {soc!r} for CIP {cip}")
        out[cip].add(soc)
    return dict(out)


def load_onet_elements(
    path: str | Path, elements: tuple[str, ...], what: str = "RIASEC"
) -> dict[str, dict[str, float]]:
    """O*NET-SOC code -> {element name: OI value (1-7)} for the given elements.
    Other scales (IH high points, DS display rank) and other elements are ignored."""
    rows = read_table(path, required=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"])
    wanted = set(elements)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["Scale ID"] != "OI" or r["Element Name"] not in wanted:
            continue
        v = float(r["Data Value"])
        if not 1.0 <= v <= 7.0:
            raise ValueError(f"{path}: OI value {v} outside 1-7 for {r['O*NET-SOC Code']}")
        out[r["O*NET-SOC Code"]][r["Element Name"]] = v
    incomplete = [k for k, d in out.items() if set(d) != wanted]
    if incomplete:
        raise ValueError(f"{path}: occupations missing some {what} values: {incomplete[:5]}")
    if not out:
        raise ValueError(f"{path}: no OI rows found")
    return dict(out)


def load_onet_interests(path: str | Path) -> dict[str, dict[str, float]]:
    """O*NET-SOC code -> {RIASEC name: OI value (1-7)} (Career Interest Types)."""
    return load_onet_elements(path, RIASEC, "RIASEC")


def basic_interest_key(name: str) -> str:
    """'Applied Arts and Design' -> 'bi_applied_arts_and_design'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"bi_{slug}"


def load_basic_interest_map(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Specific interest area name -> RIASEC type names it links to (O*NET 31.0
    specific_interest_areas_to_career_interest_types.csv; one or two each)."""
    rows = read_table(path, required=["Specific Interest Areas Element Name",
                                      "Career Interest Types Element Name"])  # fmt: skip
    out: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        t = r["Career Interest Types Element Name"]
        if t not in RIASEC:
            raise ValueError(f"{path}: unknown RIASEC type {t!r}")
        out[r["Specific Interest Areas Element Name"]].append(t)
    if not out:
        raise ValueError(f"{path}: empty mapping")
    return {k: tuple(v) for k, v in out.items()}


def load_onet_basic_interests(
    path: str | Path, areas: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """O*NET-SOC code -> {area name: OI value (1-7)} (Specific Interest Areas)."""
    return load_onet_elements(path, areas, "specific interest area")


def soc_profiles(
    interests: dict[str, dict[str, float]], elements: tuple[str, ...] = RIASEC
) -> dict[str, dict[str, float]]:
    """SOC detailed code -> mean OI over its O*NET-SOC occupations."""
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for code, prof in interests.items():
        m = _ONET.match(code)
        if not m:
            raise ValueError(f"malformed O*NET-SOC code {code!r}")
        groups[m.group(1)].append(prof)
    return {
        soc: {k: sum(p[k] for p in profs) / len(profs) for k in elements}
        for soc, profs in groups.items()
    }


@dataclass(frozen=True)
class OccupationProfile:
    cip: str
    vector: dict[str, float]  # riasec_* in [0, 1]; empty when no SOC has O*NET data
    socs: tuple[str, ...]
    socs_with_onet: tuple[str, ...]
    missing: tuple[str, ...] = field(default=())
    excluded: tuple[str, ...] = field(default=())  # SOCs left out by `exclude_prefixes`
    fallback: bool = False  # every SOC was excluded, so all were kept (A-C0-10)

    @property
    def coverage(self) -> float:
        return len(self.socs_with_onet) / len(self.socs) if self.socs else 0.0


def occupation_profile(
    cip: str,
    crosswalk: dict[str, set[str]],
    socs: dict[str, dict[str, float]],
    keys: tuple[str, ...] = RIASEC_KEYS,
    names: tuple[str, ...] = RIASEC,
    exclude_prefixes: tuple[str, ...] = (),
) -> OccupationProfile:
    """Equal-weight mean over the CIP's SOC codes with O*NET data, rescaled
    (v - 1) / 6. `keys`/`names` select the element set (RIASEC by default; the
    41 specific interest areas for W-64). `exclude_prefixes` drops matching SOCs
    (e.g. postsecondary teachers "25-1", A-C0-10) unless that would leave none."""
    if len(keys) != len(names):
        raise ValueError("keys and names must have the same length")
    mapped = tuple(sorted(crosswalk.get(cip, set())))
    have = tuple(s for s in mapped if s in socs)
    missing = tuple(s for s in mapped if s not in socs)
    excluded: tuple[str, ...] = ()
    fallback = False
    if exclude_prefixes and have:
        kept = tuple(s for s in have if not s.startswith(exclude_prefixes))
        if kept:
            excluded = tuple(s for s in have if s not in kept)
            have = kept
        else:
            fallback = True  # only excluded SOCs: keep them rather than lose the profile
    if not have:
        return OccupationProfile(cip, {}, mapped, have, missing)
    vec = {
        key: (sum(socs[s][name] for s in have) / len(have) - 1.0) / 6.0
        for key, name in zip(keys, names, strict=True)
    }
    return OccupationProfile(cip, vec, mapped, have, missing, excluded, fallback)
