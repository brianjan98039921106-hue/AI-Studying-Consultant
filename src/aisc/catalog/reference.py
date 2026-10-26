"""C0.1 reference tables: CIP 2020, IPEDS Completions and Directory (W-14).

Reads the official files as distributed (CSV, tab-delimited text, Excel, or a
zip holding one of these), normalises codes, and filters to the MVP scope in
`data/reference/mvp_scope.json`. Nothing here fills values from memory: every
title and count comes from the source file, and every output carries the
source's provenance (URL, retrieval date, SHA-256 of the file).

Excel files need `openpyxl` (dev extra); CSV/text need only the standard library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

_CIP6 = re.compile(r"^\d{2}\.\d{4}$")
_PREFIX = re.compile(r"^\d{2}(\.\d{2}(\d{2})?)?$")


# --- provenance -----------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    file_name: str
    source_url: str
    retrieved_at: str  # ISO date the file was downloaded
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def file_provenance(path: str | Path, source_url: str, retrieved_at: str) -> Provenance:
    if not source_url.startswith("https://"):
        raise ValueError("source_url must be an https URL of the official source")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", retrieved_at):
        raise ValueError("retrieved_at must be an ISO date (YYYY-MM-DD)")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Provenance(Path(path).name, source_url, retrieved_at, digest)


# --- generic table reading ----------------------------------------------------------


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _rows_from_text(text: str, delimiter: str | None = None) -> list[dict[str, str]]:
    if delimiter is None:
        first = text.split("\n", 1)[0]
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def _rows_from_xlsx(data: bytes, required: Iterable[str]) -> list[dict[str, str]]:
    from openpyxl import load_workbook  # optional dependency

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    need = {r.lower() for r in required}
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        for header in rows:  # header may not be the first row (title rows)
            names = [str(c).strip() if c is not None else "" for c in header]
            if need <= {n.lower() for n in names}:
                out = []
                for r in rows:
                    if r is None or all(c is None for c in r):
                        continue
                    out.append({n: _cell(c) for n, c in zip(names, r, strict=False) if n})
                return out
    raise ValueError(f"no worksheet has the columns {sorted(required)}")


def _cell(c) -> str:
    if c is None:
        return ""
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c).strip()


def read_table(
    path: str | Path, required: Iterable[str] = (), prefer: str = ""
) -> list[dict[str, str]]:
    """Rows of a CSV / tab text / xlsx file, or of the table inside a zip.

    Inside a zip, a member whose name contains `prefer` wins (IPEDS ships a
    revised `*_rv.csv` next to the original; pass prefer="_rv").
    """
    path = Path(path)
    required = list(required)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            tables = [n for n in members if n.lower().endswith((".csv", ".txt", ".xlsx"))]
            if not tables:
                raise ValueError(f"{path}: no table inside the zip")
            chosen = sorted(tables, key=lambda n: (prefer not in n.lower() if prefer else 0, n))[0]
            data, name = zf.read(chosen), chosen
    else:
        data, name = path.read_bytes(), path.name
    if name.lower().endswith(".xlsx"):
        rows = _rows_from_xlsx(data, required)
    else:
        rows = _rows_from_text(_decode(data), "\t" if name.lower().endswith(".txt") else None)
    if rows and required:
        missing = [r for r in required if r not in rows[0]]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; found {list(rows[0])}")
    return rows


# --- CIP 2020 -----------------------------------------------------------------------


def normalize_cip(raw: str) -> str:
    """'="52.0201"' / '52.0201' / '5.0201' -> '52.0201'; series '52.02'; family '52'."""
    s = str(raw).strip().strip('="').strip()
    if not s:
        raise ValueError("empty CIP code")
    if "." in s:
        head, tail = s.split(".", 1)
        if not (head.isdigit() and tail.isdigit() and len(tail) in (2, 4)):
            raise ValueError(f"malformed CIP code {raw!r}")
        return f"{int(head):02d}.{tail}"
    if s.isdigit() and len(s) <= 2:
        return f"{int(s):02d}"
    raise ValueError(f"malformed CIP code {raw!r}")


@dataclass(frozen=True)
class CipEntry:
    code: str
    title: str
    definition: str


def load_cip2020(path: str | Path) -> dict[str, CipEntry]:
    """All codes valid in CIP 2020 (rows whose Action is 'Moved to' or 'Deleted'
    describe codes that no longer exist and are dropped)."""
    rows = read_table(path, required=["CIPCode", "CIPTitle"])
    out: dict[str, CipEntry] = {}
    for r in rows:
        action = r.get("Action", "").lower()
        if action.startswith(("moved to", "deleted")):
            continue
        code = normalize_cip(r["CIPCode"])
        out[code] = CipEntry(code, r["CIPTitle"].strip().rstrip("."), r.get("CIPDefinition", ""))
    if not any(_CIP6.match(c) for c in out):
        raise ValueError(f"{path}: no 6-digit CIP codes found")
    return out


@dataclass(frozen=True)
class Scope:
    version: str
    award_level: int
    majornum: int
    families: dict[str, list[str]]  # family -> prefixes

    def family_of(self, cip6: str) -> str | None:
        best, best_len = None, -1
        for fam, prefixes in self.families.items():
            for p in prefixes:
                if cip6.startswith(p) and len(p) > best_len:
                    best, best_len = fam, len(p)
        return best


def load_scope(path: str | Path) -> Scope:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    fams: dict[str, list[str]] = {}
    for f in spec["families"]:
        for p in f["prefixes"]:
            if not _PREFIX.match(p):
                raise ValueError(f"bad CIP prefix {p!r} in family {f['name']}")
        fams[f["name"]] = list(f["prefixes"])
    return Scope(spec["version"], int(spec["award_level"]), int(spec["majornum"]), fams)


def select_cips(cip: dict[str, CipEntry], scope: Scope) -> dict[str, str]:
    """6-digit codes in scope -> family. Every prefix must exist in the official
    file (as a code itself or as the start of a 6-digit code)."""
    six = [c for c in cip if _CIP6.match(c)]
    unknown = [
        p for ps in scope.families.values() for p in ps
        if p not in cip and not any(c.startswith(p) for c in six)
    ]  # fmt: skip
    if unknown:
        raise ValueError(f"scope prefixes not in CIP 2020: {unknown}")
    out = {}
    for c in sorted(six):
        fam = scope.family_of(c)
        if fam is not None:
            out[c] = fam
    return out


# --- IPEDS ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Completion:
    unitid: str
    cip: str
    awards: int


def load_ipeds_completions(
    path: str | Path, codes: Iterable[str], award_level: int = 7, majornum: int = 1
) -> list[Completion]:
    """Rows of an IPEDS C<year>_A file for the given 6-digit codes, award level and
    first/second major, with at least one award (CTOTALT > 0)."""
    wanted = set(codes)
    rows = read_table(path, required=["UNITID", "CIPCODE", "AWLEVEL", "CTOTALT"], prefer="_rv")
    out = []
    for r in rows:
        if int(r["AWLEVEL"]) != award_level:
            continue
        if "MAJORNUM" in r and r["MAJORNUM"] and int(r["MAJORNUM"]) != majornum:
            continue
        try:
            code = normalize_cip(r["CIPCODE"])
        except ValueError:
            continue  # e.g. "99" grand-total rows
        if code not in wanted:
            continue
        n = int(float(r["CTOTALT"] or 0))
        if n > 0:
            out.append(Completion(r["UNITID"], code, n))
    keys = [(c.unitid, c.cip) for c in out]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate (UNITID, CIPCODE) rows after filtering")
    return out


@dataclass(frozen=True)
class Institution:
    unitid: str
    name: str
    city: str
    state: str
    web: str


def load_ipeds_directory(path: str | Path) -> dict[str, Institution]:
    rows = read_table(path, required=["UNITID", "INSTNM", "STABBR"], prefer="_rv")
    return {
        r["UNITID"]: Institution(
            r["UNITID"], r["INSTNM"], r.get("CITY", ""), r["STABBR"], r.get("WEBADDR", "")
        )
        for r in rows
    }


def school_cip_rows(
    completions: list[Completion],
    directory: dict[str, Institution],
    cip: dict[str, CipEntry],
    families: dict[str, str],
    year: str,
) -> list[dict[str, str | int]]:
    """The C0.1 'school x CIP' list (B-C0-7: not yet programs)."""
    missing = sorted({c.unitid for c in completions} - set(directory))
    if missing:
        raise ValueError(f"UNITIDs missing from the directory file: {missing[:5]}...")
    out = []
    for c in sorted(completions, key=lambda c: (c.cip, directory[c.unitid].name)):
        inst = directory[c.unitid]
        out.append({
            "unitid": c.unitid, "institution": inst.name, "city": inst.city,
            "state": inst.state, "web": inst.web, "cip": c.cip, "cip_title": cip[c.cip].title,
            "family": families[c.cip], "masters_awards": c.awards, "completions_year": year,
        })  # fmt: skip
    return out
