"""C3.1 data intake: de-identified application outcomes and official admission rates
from CSV files (docs/data-intake.md; THEORY "申請結果資料載入"; A-C3-5, A-C3-9).

Records file (one row per application; template data/templates/admission_records.csv)
    record_id      opaque id, e.g. R0001 (never a name or student number)
    year           application year (all rows or none)
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

Nothing here guesses: rows that cannot be read are rejected with their line number;
columns or cells that look like personal identifiers stop the load.
"""

from __future__ import annotations

import csv
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
PII_COLUMNS = re.compile(
    r"(^|_)(name|姓名|名字|email|e-mail|mail|phone|tel|電話|手機|student_?id|學號|id_?number|"
    r"身分證|passport|護照|address|地址|birth|生日|line_?id)($|_)",
    re.IGNORECASE,
)
PII_VALUES = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # e-mail
    re.compile(r"^[A-Z][12]\d{8}$"),  # Taiwan national ID
    re.compile(r"^(\+?886-?|0)9\d{2}-?\d{3}-?\d{3}$"),  # Taiwan mobile number
)


@dataclass(frozen=True)
class IntakeReport:
    n_rows: int
    features: tuple[str, ...]  # model feature columns in order (incl. missing indicators)
    means: dict[str, float]
    sds: dict[str, float]
    standardized_on: str  # "complete records" or "all records"
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


def _check_pii(header: Sequence[str], rows: list[dict[str, str]], path: Path) -> None:
    bad = [h for h in header if PII_COLUMNS.search(h.strip())]
    if bad:
        raise ValueError(f"{path}: columns look like personal identifiers {bad}; "
                         "remove them before loading (docs/data-intake.md)")  # fmt: skip
    for n, r in enumerate(rows, start=2):
        for k, v in r.items():
            if v and any(p.search(v.strip()) for p in PII_VALUES):
                raise ValueError(f"{path}:{n}: column {k!r} looks like an e-mail, ID number "
                                 "or phone number; de-identify the file first")  # fmt: skip


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
        return list(reader.fieldnames or []), rows


def load_admission_records(path: str | Path) -> IntakeData:
    """Read, validate and standardise an outcomes file (module docstring)."""
    path = Path(path)
    header, rows = _read(path)
    _check_pii(header, rows, path)
    need = {"record_id", "program_id", "cip", "source", "outcome"}
    missing_cols = need - set(header)
    if missing_cols:
        raise ValueError(f"{path}: missing columns {sorted(missing_cols)}")
    if not rows:
        raise ValueError(f"{path}: no rows")
    xcols = [h for h in header if h.startswith("x_")]
    ids, parsed = set(), []
    fam_of: dict[str, str] = {}
    for n, r in enumerate(rows, start=2):
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
            v = (r.get(c) or "").strip()
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
                       (r.get("program_type") or "").strip(), (r.get("group") or "").strip(),
                       x))  # fmt: skip
    years = {p[4] for p in parsed}
    if None in years and len(years) > 1:
        raise ValueError(f"{path}: give a year for every row or for none")
    ptypes = {p[5] for p in parsed}
    if "" in ptypes and len(ptypes) > 1:
        raise ValueError(f"{path}: give program_type for every row or for none")
    types: dict[str, str] | None = None
    if "" not in ptypes:
        types = {}
        for pid, *_rest in parsed:
            t = _rest[4]
            if types.setdefault(pid, t) != t:
                raise ValueError(f"{path}: programme {pid} has two program_type values")

    # standardisation (A-C3-9): complete records when there are enough of them
    complete = [p for p in parsed if p[3] == "complete"]
    base = complete if len(complete) >= 30 else parsed
    on = "complete records" if base is complete else "all records"
    means, sds, miss, warns = {}, {}, {}, []
    for c in xcols:
        vals = [p[7][c] for p in base if p[7][c] is not None]
        miss[c] = sum(p[7][c] is None for p in parsed)
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
    return IntakeData(records, fam_of, types, groups, report, [p[7] for p in parsed])


def load_official_rates(path: str | Path, year: int | None = None) -> dict[str, float]:
    """{program_id: admission rate} for `year` (default: each programme's latest year).
    Every row needs source_url and retrieved_at (real programme facts, CLAUDE.md)."""
    path = Path(path)
    header, rows = _read(path)
    _check_pii(header, rows, path)
    need = {"program_id", "year", "source_url", "retrieved_at"}
    if need - set(header):
        raise ValueError(f"{path}: missing columns {sorted(need - set(header))}")
    best: dict[str, tuple[int, float]] = {}
    for n, r in enumerate(rows, start=2):
        where = f"{path}:{n}"
        if not r["source_url"].strip() or not r["retrieved_at"].strip():
            raise ValueError(f"{where}: source_url and retrieved_at are required")
        y = r["year"].strip()
        if not y.isdigit():
            raise ValueError(f"{where}: year {y!r} is not a year")
        rate_s = (r.get("admit_rate") or "").strip()
        if rate_s:
            rate = float(rate_s)
        else:
            try:
                rate = float(r["admitted"]) / float(r["applicants"])
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"{where}: give admit_rate or admitted and applicants") from exc
        if not 0 < rate < 1:
            raise ValueError(f"{where}: admission rate must lie strictly between 0 and 1")
        pid, yi = r["program_id"].strip(), int(y)
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
