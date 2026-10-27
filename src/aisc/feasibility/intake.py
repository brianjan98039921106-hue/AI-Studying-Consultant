"""C3.1 data intake: de-identified application outcomes and official admission rates
from CSV files (docs/data-intake.md; THEORY "申請結果資料載入"; A-C3-5, A-C3-9).

Records file (one row per application; template data/templates/admission_records.csv)
    record_id      opaque id, e.g. R0001 (never a name or student number)
    year           entering year, e.g. 2025 for Fall 2025 entry (all rows or none);
                   the same meaning as in the rates file
    program_id     the consultant's id for the programme
    cip            6-digit CIP 2020 code ("52.1301"; "9.0101" is padded)
    program_type   optional: standard / interview / portfolio / professional / ...
                   (all rows or none)
    source         complete (consultant's own case, every outcome recorded) or
                   self_report (e.g. shared online results)
    outcome        admit / reject (also 1 / 0, 錄取 / 未錄取); waitlist must be resolved
    group          optional applicant group, e.g. taiwan / international (for V-C3-4)
    x_*            numeric applicant features (e.g. x_gpa_pct, x_toefl, x_work_years);
                   blanks allowed
Rates file (template data/templates/official_rates.csv)
    program_id, year, admit_rate (0-1) or applicants + admitted, source_url, retrieved_at

Files must be UTF-8 (Excel: "CSV UTF-8"). Only the template columns and x_* features are
accepted; any other column stops the load, and so do cells that look like an e-mail, a
Taiwan ID / resident certificate number or a mobile number (B-C3-13). Nothing here
guesses: rows that cannot be read are rejected with their line number (B-C3-15).
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aisc.feasibility.hierarchical import AdmissionModel, AdmissionRecord, fit_admission_model

OUTCOMES = {
    "admit": 1, "admitted": 1, "accept": 1, "1": 1, "錄取": 1,
    "reject": 0, "rejected": 0, "deny": 0, "0": 0, "未錄取": 0, "拒絕": 0,
}  # fmt: skip
RECORD_COLUMNS = {"record_id", "year", "program_id", "cip", "program_type", "source", "outcome",
                  "group"}  # fmt: skip
RATE_COLUMNS = {"program_id", "year", "admit_rate", "applicants", "admitted", "source_url",
                "retrieved_at"}  # fmt: skip
# identifier words, matched after removing spaces, underscores and hyphens (B-C3-13)
PII_WORDS = ("name", "email", "mail", "phone", "mobile", "cell", "dob", "birth", "passport",
             "address", "studentid", "studentno", "idnumber", "idno", "lineid", "wechat",
             "姓名", "名字", "電話", "手機", "學號", "身分證", "身份證", "護照", "地址", "住址",
             "生日", "居留證")  # fmt: skip
PII_VALUES = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # e-mail
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z][1289]\d{8}(?!\d)"),  # Taiwan ID / resident cert.
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{8}(?!\d)"),  # old resident certificate
    re.compile(r"(?<!\d)(\+?886|0)9\d{8}(?!\d)"),  # Taiwan mobile (spaces/hyphens removed)
)
DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


@dataclass(frozen=True)
class IntakeReport:
    n_rows: int
    features: tuple[str, ...]  # model feature columns in order (incl. missing indicators)
    means: dict[str, float]
    sds: dict[str, float]
    standardized_on: str  # "complete records", "all records" or "given statistics"
    missing: dict[str, int]
    by_source: dict[str, int]
    groups: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntakeData:
    records: list[AdmissionRecord]
    families: dict[str, str]
    types: dict[str, str] | None
    groups: list[str]  # per record, "" when not given
    report: IntakeReport
    raw_features: list[dict[str, float | None]] = field(default_factory=list)


def _norm_cip(cip: str, where: str) -> str:
    head, dot, tail = cip.strip().partition(".")
    if not dot or not head.isdigit() or not tail.isdigit() or len(tail) != 4:
        raise ValueError(f"{where}: CIP must look like 52.1301, got {cip!r}")
    return f"{head.zfill(2)}.{tail}"


def _check_columns(header: Sequence[str], allowed: set[str], path: Path) -> None:
    bad = [h for h in header if any(w in re.sub(r"[\s_\-]", "", h) for w in PII_WORDS)]
    if bad:
        raise ValueError(f"{path}: columns look like personal identifiers {bad}; "
                         "remove them before loading (docs/data-intake.md)")  # fmt: skip
    other = [h for h in header if h not in allowed and not h.startswith("x_")]
    if other:
        raise ValueError(f"{path}: columns {other} are not in the template; remove them or "
                         "name numeric features x_... (docs/data-intake.md)")  # fmt: skip


def _check_cells(rows: list[tuple[int, dict[str, str]]], path: Path) -> None:
    for n, r in rows:
        for k, v in r.items():
            compact = re.sub(r"[\s\-]", "", v)
            if v and any(p.search(v) or p.search(compact) for p in PII_VALUES):
                raise ValueError(f"{path}:{n}: column {k!r} looks like an e-mail, ID number "
                                 "or phone number; de-identify the file first")  # fmt: skip


def _read(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Header (stripped, lower case) and non-blank rows with their line numbers."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not UTF-8; in Excel use Save As > CSV UTF-8 "
                         "(docs/data-intake.md)") from None  # fmt: skip
    reader = csv.reader(io.StringIO(text, newline=""))
    raw = next(reader, None)
    if raw is None:
        raise ValueError(f"{path}: empty file")
    header = [h.strip().lower() for h in raw]
    if len(set(header)) != len(header) or "" in header:
        raise ValueError(f"{path}: column names must be non-empty and unique")
    rows = []
    for cells in reader:
        n = reader.line_num
        if not any(c.strip() for c in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(f"{path}:{n}: {len(cells)} fields but the header has {len(header)}")
        rows.append((n, dict(zip(header, cells, strict=True))))
    return header, rows


def load_admission_records(
    path: str | Path,
    years: set[int] | None = None,
    standardize_with: IntakeReport | None = None,
) -> IntakeData:
    """Read, validate and standardise an outcomes file (module docstring).

    `years` keeps only those years (after validating the whole file);
    `standardize_with` reuses another load's means, SDs and missing indicators, e.g. the
    training years' statistics for a held-out year (B-C3-14)."""
    path = Path(path)
    header, rows = _read(path)
    _check_columns(header, RECORD_COLUMNS, path)
    _check_cells(rows, path)
    need = {"record_id", "program_id", "cip", "source", "outcome"}
    missing_cols = need - set(header)
    if missing_cols:
        raise ValueError(f"{path}: missing columns {sorted(missing_cols)}")
    if not rows:
        raise ValueError(f"{path}: no rows")
    xcols = [h for h in header if h.startswith("x_")]
    ids, parsed = set(), []
    fam_of: dict[str, str] = {}
    for n, r in rows:
        where = f"{path}:{n}"
        rid = r["record_id"].strip()
        if not rid or rid in ids:
            raise ValueError(f"{where}: record_id must be present and unique")
        ids.add(rid)
        pid = r["program_id"].strip()
        if not pid:
            raise ValueError(f"{where}: program_id is empty")
        fam = _norm_cip(r["cip"], where)[:5]
        if fam_of.setdefault(pid, fam) != fam:
            raise ValueError(f"{where}: programme {pid} has two different CIP families")
        out = OUTCOMES.get(r["outcome"].strip().lower())
        if out is None:
            raise ValueError(f"{where}: outcome {r['outcome']!r} is not admit/reject "
                             "(resolve waitlists before loading)")  # fmt: skip
        src = r["source"].strip().lower()
        if src not in ("complete", "self_report"):
            raise ValueError(f"{where}: source must be complete or self_report")
        year = (r.get("year") or "").strip()
        if year and not (year.isdigit() and 1990 <= int(year) <= 2100):
            raise ValueError(f"{where}: year {year!r} is not a year")
        x: dict[str, float | None] = {}
        for c in xcols:
            v = r[c].strip()
            if v == "":
                x[c] = None
                continue
            try:
                x[c] = float(v)
            except ValueError as exc:
                raise ValueError(f"{where}: {c} = {v!r} is not a number") from exc
            if not math.isfinite(x[c]):
                raise ValueError(f"{where}: {c} must be finite")
        parsed.append((pid, fam, out, src, int(year) if year else None,
                       (r.get("program_type") or "").strip().lower(),
                       (r.get("group") or "").strip(), x))  # fmt: skip
    all_years = {p[4] for p in parsed}
    if None in all_years and len(all_years) > 1:
        raise ValueError(f"{path}: give a year for every row or for none")
    ptypes = {p[5] for p in parsed}
    if "" in ptypes and len(ptypes) > 1:
        raise ValueError(f"{path}: give program_type for every row or for none")
    type_of: dict[str, str] = {}
    if "" not in ptypes:
        for p in parsed:
            if type_of.setdefault(p[0], p[5]) != p[5]:
                raise ValueError(f"{path}: programme {p[0]} has two program_type values")
    if years is not None:
        if None in all_years:
            raise ValueError(f"{path}: years requested but the file has no year column")
        parsed = [p for p in parsed if p[4] in years]
        if not parsed:
            raise ValueError(f"{path}: no rows in years {sorted(years)}")
    progs = {p[0] for p in parsed}
    families = {k: v for k, v in fam_of.items() if k in progs}
    types = {k: v for k, v in type_of.items() if k in progs} if type_of else None

    means, sds, miss, warns = {}, {}, {}, []
    for c in xcols:
        miss[c] = sum(p[7][c] is None for p in parsed)
    if standardize_with is not None:
        given = [f for f in standardize_with.features if not f.endswith("_missing")]
        if set(given) != set(xcols):
            raise ValueError(f"{path}: features {xcols} differ from the given statistics {given}")
        means, sds, on = (
            dict(standardize_with.means),
            dict(standardize_with.sds),
            "given statistics",
        )
        indicators = [c for c in xcols if f"{c}_missing" in standardize_with.features]
    else:
        # A-C3-9: complete records when there are enough of them, per feature
        complete = [p for p in parsed if p[3] == "complete"]
        base = complete if len(complete) >= 30 else parsed
        on = "complete records" if base is complete else "all records"
        for c in xcols:
            vals = [p[7][c] for p in base if p[7][c] is not None]
            if len(vals) < 2 and base is complete:
                vals = [p[7][c] for p in parsed if p[7][c] is not None]
                warns.append(f"feature {c} has fewer than 2 values in complete records; "
                             "standardised on all records (A-C3-9)")  # fmt: skip
            if len(vals) < 2:
                raise ValueError(f"{path}: feature {c} has fewer than 2 values")
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            if sd == 0:
                raise ValueError(f"{path}: feature {c} is constant")
            means[c], sds[c] = m, sd
        if on == "all records" and any(p[3] == "self_report" for p in parsed):
            warns.append("features standardised on all records, including self-reports "
                         "that over-represent admits (A-C3-9); add complete records")  # fmt: skip
        indicators = [c for c in xcols if miss[c] > 0]
    feats = tuple(xcols + [f"{c}_missing" for c in indicators])
    records = []
    for pid, fam, out, src, year, _t, _g, x in parsed:
        z = [0.0 if x[c] is None else (x[c] - means[c]) / sds[c] for c in xcols]
        z += [1.0 if x[c] is None else 0.0 for c in indicators]
        records.append(AdmissionRecord(pid, fam, tuple(z), out, src, year))
    groups = [p[6] for p in parsed]
    report = IntakeReport(
        n_rows=len(parsed), features=feats, means=means, sds=sds, standardized_on=on,
        missing=miss, by_source={s: sum(p[3] == s for p in parsed) for s in ("complete",
                                 "self_report")},
        groups=tuple(sorted({g for g in groups if g})), warnings=tuple(warns),
    )  # fmt: skip
    return IntakeData(records, families, types, groups, report, [p[7] for p in parsed])


def load_official_rates(path: str | Path, year: int | None = None) -> dict[str, float]:
    """{program_id: admission rate} for `year` (default: each programme's latest year).
    Every row needs source_url and retrieved_at (YYYY-MM-DD; real programme facts,
    CLAUDE.md); a programme may have one rate per year (B-C3-15)."""
    path = Path(path)
    header, rows = _read(path)
    _check_columns(header, RATE_COLUMNS, path)
    _check_cells(rows, path)
    need = {"program_id", "year", "source_url", "retrieved_at"}
    if need - set(header):
        raise ValueError(f"{path}: missing columns {sorted(need - set(header))}")
    seen: dict[tuple[str, int], float] = {}
    for n, r in rows:
        where = f"{path}:{n}"
        pid = r["program_id"].strip()
        if not pid:
            raise ValueError(f"{where}: program_id is empty")
        if not r["source_url"].strip() or not r["retrieved_at"].strip():
            raise ValueError(f"{where}: source_url and retrieved_at are required")
        if not DATE.match(r["retrieved_at"].strip()):
            raise ValueError(f"{where}: retrieved_at must be a date like 2026-10-28")
        y = r["year"].strip()
        if not (y.isdigit() and 1990 <= int(y) <= 2100):
            raise ValueError(f"{where}: year {y!r} is not a year")
        rate_s = (r.get("admit_rate") or "").strip()
        try:
            if rate_s:
                rate = float(rate_s)
            else:
                rate = float(r.get("admitted") or "") / float(r.get("applicants") or "")
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{where}: give admit_rate as a fraction (0.25, not 25%) or "
                             "admitted and applicants") from None  # fmt: skip
        if not 0 < rate < 1:
            raise ValueError(f"{where}: admission rate must lie strictly between 0 and 1")
        key = (pid, int(y))
        if key in seen and abs(seen[key] - rate) > 1e-9:
            raise ValueError(f"{where}: programme {pid} has two different rates for {y}")
        seen[key] = rate
    best: dict[str, tuple[int, float]] = {}
    for (pid, yi), rate in seen.items():
        if year is not None and yi != year:
            continue
        if pid not in best or yi > best[pid][0]:
            best[pid] = (yi, rate)
    return {p: v for p, (_, v) in best.items()}


def fit_from_files(
    records_path: str | Path, rates_path: str | Path | None = None
) -> tuple[AdmissionModel, IntakeData]:
    """Load both files and fit C3.2 with the current defaults. Official rates are taken
    for the latest year in the records (or each programme's latest year)."""
    data = load_admission_records(records_path)
    years = {r.year for r in data.records}
    latest = max(years) if None not in years else None
    rates = load_official_rates(rates_path, latest) if rates_path else None
    model = fit_admission_model(data.records, families=data.families,
                                official_rates=rates, types=data.types)  # fmt: skip
    return model, data
