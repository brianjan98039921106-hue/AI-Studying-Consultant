"""C1 / C6.4 (W-61): student-record intake -> contracts.StudentRecord.

What the student has DONE (degrees, courses, exams, experience). The result is a
StudentRecord for C6 (and, when resumed, C3); C2 never reads it (D9).

Input format (primary): one JSON object per student, schema
"student_record_intake/1" (docs/USAGE.md §4.6). JSON rather than CSV because a
record nests several lists (degrees, courses, exams, experience) of different
shapes; CSV would need one file per list. A Python dict with the same shape is
accepted directly.

Course names are mapped to requirement-taxonomy categories by an explicit,
versioned table (data/reference/course_name_map_tw.json): exact match after a
documented normalisation, no fuzzy matching, no LLM. Names the table does not
know, or deliberately leaves to a human (`review`), are listed in
StudentRecord.unmapped with the reason and are never counted as met (A-C6-10,
B-C6-2, B-C6-10).
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aisc.contracts import Degree, StudentRecord

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COURSE_MAP = ROOT / "data" / "reference" / "course_name_map_tw.json"
DEFAULT_TAXONOMY = ROOT / "data" / "reference" / "requirement_taxonomy.json"
SCHEMA = "student_record_intake/1"

DEGREE_LEVELS = ("bachelor", "master", "doctorate", "professional", "other")
IMPLIES_BACHELOR = ("bachelor", "master", "doctorate")  # A-C6-12
DEGREE_ZH = {"bachelor": "學士", "master": "碩士", "doctorate": "博士",
             "professional": "專業學位", "other": "其他學歷"}  # fmt: skip
STATUSES = ("completed", "planned")
PASS_MARK = 60  # A-C6-11: 0-100 scale, the usual undergraduate pass mark in Taiwan
_PASS = {"p", "pass", "passed", "通過", "抵免", "免修", "及格"}
_FAIL = {"f", "e", "x", "fail", "failed", "不及格", "不通過", "w", "withdrawn", "停修", "棄修",
         "i", "incomplete", "未完成"}  # fmt: skip
_LETTER = re.compile(r"^[a-c][+-]?$")  # D is a pass in some systems, a fail in others

_ORD = {"一": 1, "二": 2, "三": 3, "四": 4, "上": 1, "下": 2,
        "1": 1, "2": 2, "3": 3, "4": 4, "i": 1, "ii": 2, "iii": 3, "iv": 4}  # fmt: skip
_ORD_ALT = "iv|iii|ii|i|[1-4一二三四上下]"
_PAREN_ORD = re.compile(rf"^(?P<base>.+?)\s*\((?P<ord>{_ORD_ALT})\)$")
_SPACE_ORD = re.compile(r"^(?P<base>.+?)\s+(?P<ord>iv|iii|ii|i|[1-4])$")
_CJK_ORD = re.compile(rf"^(?P<base>.*[一-鿿])(?P<ord>{_ORD_ALT})$")
_LANG_NOTE = re.compile(r"\s*\([^()]*(?:授課|emi)[^()]*\)\s*$")


class RecordIntakeError(ValueError):
    """All validation problems of one intake, each prefixed with its JSON path."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("student record intake failed:\n  - " + "\n  - ".join(self.errors))


def normalize_name(name: str, ordinals: bool = True) -> tuple[str, int | None]:
    """(base, ordinal). NFKC, lower case, teaching-language note dropped, ordinal
    split off (（一）/(I)/一/上/ I / 1 ...), then all whitespace removed."""
    s = unicodedata.normalize("NFKC", name).strip().lower()
    s = _LANG_NOTE.sub("", s)
    ordinal = None
    if ordinals:
        for pat in (_PAREN_ORD, _SPACE_ORD, _CJK_ORD):
            m = pat.match(s)
            if m:
                s, ordinal = m.group("base"), _ORD[m.group("ord")]
                break
    return re.sub(r"\s+", "", s), ordinal


# --- mapping table -------------------------------------------------------------


@dataclass(frozen=True)
class CourseMap:
    """Loaded, validated course-name / exam-name mapping table."""

    version: str
    status: str
    taxonomy: Mapping[str, frozenset[str]]  # kind -> category keys
    courses: Mapping[str, tuple[str, tuple[str, ...], Mapping[str, tuple[str, ...]], str]]
    review: Mapping[str, str]  # normalised course name -> reason
    exams: Mapping[str, str]  # normalised exam name -> category
    exam_review: Mapping[str, str]

    @property
    def categories(self) -> frozenset[str]:
        return frozenset().union(*self.taxonomy.values())

    def kind_of(self, category: str) -> str | None:
        for kind, cats in self.taxonomy.items():
            if category in cats:
                return kind
        return None

    def map_course(self, name: str) -> tuple[str, tuple[str, ...], str]:
        """(source, categories, note); source is 'table:<id>', 'review' or 'unmatched'."""
        base, ordinal = normalize_name(name)
        if base in self.review:
            return "review", (), self.review[base]
        if base not in self.courses:
            return "unmatched", (), "課名不在對應表中，請手動對應"
        entry_id, cats, by_ord, note = self.courses[base]
        key = "none" if ordinal is None else str(ordinal)
        cats = by_ord.get(key, cats)
        if not cats:
            return "review", (), f"對應表沒有「{name}」這個序號的規則，請手動對應"
        return f"table:{entry_id}", cats, note

    def map_exam(self, name: str) -> tuple[str, tuple[str, ...], str]:
        base, _ = normalize_name(name, ordinals=False)
        if base in self.exam_review:
            return "review", (), self.exam_review[base]
        if base in self.exams:
            return "table:exam", (self.exams[base],), ""
        return "unmatched", (), "考試名稱不在對應表中，請手動對應"


def _load_taxonomy(path: str | Path) -> tuple[str, dict[str, frozenset[str]]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec.get("version", ""), {k: frozenset(v) for k, v in spec["categories"].items()}


def load_course_map(
    path: str | Path = DEFAULT_COURSE_MAP, taxonomy: str | Path = DEFAULT_TAXONOMY
) -> CourseMap:
    """Load the mapping table and check it against the requirement taxonomy:
    versions agree, every category exists and has the right kind, no alias is
    claimed twice, and no alias carries an ordinal (those go in `by_ordinal`)."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    tax_version, tax = _load_taxonomy(taxonomy)
    if spec["taxonomy_version"] != tax_version:
        raise ValueError(f"course map is for taxonomy {spec['taxonomy_version']!r}, "
                         f"taxonomy file is {tax_version!r}")  # fmt: skip
    seen: dict[str, str] = {}

    def claim(alias: str, owner: str, ordinals: bool = True) -> str:
        base, ordinal = normalize_name(alias, ordinals)
        if ordinal is not None:
            raise ValueError(f"alias {alias!r} ({owner}) has an ordinal; use by_ordinal")
        if base in seen:
            raise ValueError(f"alias {alias!r} claimed by {seen[base]!r} and {owner!r}")
        seen[base] = owner
        return base

    def check(cats, kinds, owner):
        for c in cats:
            if not any(c in tax.get(k, ()) for k in kinds):
                raise ValueError(f"{owner}: {c!r} is not a {'/'.join(kinds)} category")
        return tuple(cats)

    courses, review, exams, exam_review = {}, {}, {}, {}
    for e in spec["courses"]:
        cats = check(e["categories"], ("course",), e["id"])
        by_ord = {}
        for k, v in e.get("by_ordinal", {}).items():
            if k not in ("none", "1", "2", "3", "4"):
                raise ValueError(f"{e['id']}: by_ordinal key {k!r} must be none or 1-4")
            by_ord[k] = check(v, ("course",), e["id"])
        for alias in e["names"]:
            courses[claim(alias, e["id"])] = (e["id"], cats, by_ord, e.get("note", ""))
    for e in spec["review"]:
        for alias in e["names"]:
            review[claim(alias, "review")] = e["reason"]
    seen.clear()  # exam names are a separate namespace
    for e in spec["exams"]:
        (cat,) = check([e["category"]], ("test", "language"), e["category"])
        for alias in e["names"]:
            exams[claim(alias, cat, ordinals=False)] = cat
    for e in spec["exam_review"]:
        for alias in e["names"]:
            exam_review[claim(alias, "exam_review", ordinals=False)] = e["reason"]
    return CourseMap(spec["version"], spec["status"], tax, courses, review, exams, exam_review)


# --- intake --------------------------------------------------------------------


@dataclass(frozen=True)
class EntryMapping:
    """How one course or exam entry was handled. `source`: 'table:<id>', 'manual',
    'review', 'unmatched', 'not_passed' or 'none' (student confirmed no category)."""

    section: str  # "courses" | "exams"
    name: str
    source: str
    categories: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class IntakeResult:
    record: StudentRecord
    entries: tuple[EntryMapping, ...]
    map_version: str
    map_status: str
    synthetic: bool

    @property
    def unmatched(self) -> list[str]:
        return list(self.record.unmapped)


_TOP = {"schema", "student_id", "synthetic", "target_apply_by", "degrees", "courses", "exams",
        "experience", "other"}  # fmt: skip
_FIELDS = {
    "degrees": {"level", "field", "cip", "year"},
    "courses": {"name", "credits", "grade", "online", "status", "expected", "categories"},
    "exams": {"name", "score", "date", "status", "expected", "category"},
    "experience": {"category", "description", "months", "status", "expected"},
    "other": {"category", "evidence", "status", "expected"},
}
_REQUIRED = {
    "degrees": ("level", "field"),
    "courses": ("name",),
    "exams": ("name",),
    "experience": ("category", "description"),
    "other": ("category", "evidence"),
}
_OTHER_KINDS = ("portfolio", "credential", "other", "language")


def _passed(grade, where: str, errors: list[str]) -> bool | None:
    """True / False, or None after recording an error. A-C6-11."""
    if grade is None or (isinstance(grade, str) and not grade.strip()):
        return True
    if isinstance(grade, bool):
        errors.append(f"{where}: grade must be a score or letter, not {grade!r}")
        return None
    if isinstance(grade, int | float):
        num = float(grade)
    else:
        g = unicodedata.normalize("NFKC", str(grade)).strip().lower()
        if g in _PASS or _LETTER.match(g):
            return True
        if g in _FAIL:
            return False
        if re.fullmatch(r"d[+-]?", g):
            errors.append(f"{where}: letter grade {grade!r} is a pass at some schools and a "
                          "fail at others; enter the 0-100 score instead")  # fmt: skip
            return None
        try:
            num = float(g)
        except ValueError:
            errors.append(f"{where}: unrecognised grade {grade!r}; use 0-100, a letter "
                          "A+..F, P/通過/抵免, or F/不及格/W/停修")  # fmt: skip
            return None
    if not 0 <= num <= 100:
        errors.append(f"{where}: grade {grade!r} is outside 0-100")
        return None
    if num <= 4.3 and num != int(num):
        errors.append(f"{where}: grade {grade!r} looks like a grade point; enter the letter "
                      "grade or the 0-100 score")  # fmt: skip
        return None
    return num >= PASS_MARK


def _category(cat, kinds, where, cmap: CourseMap, errors) -> str | None:
    if not isinstance(cat, str) or cmap.kind_of(cat) not in kinds:
        valid = sorted(c for k in kinds for c in cmap.taxonomy.get(k, ()))
        hint = difflib.get_close_matches(str(cat), valid, n=3)
        errors.append(f"{where}: {cat!r} is not a {'/'.join(kinds)} category"
                      + (f" (did you mean {', '.join(hint)}?)" if hint else
                         f"; valid: {', '.join(valid)}"))  # fmt: skip
        return None
    return cat


def _status(e, where, errors) -> str | None:
    st = e.get("status", "completed")
    if st not in STATUSES:
        errors.append(f"{where}.status: {st!r} must be one of {STATUSES}")
        return None
    if st == "planned" and not str(e.get("expected") or "").strip():
        errors.append(f"{where}.expected: planned items need an expected completion "
                      "(ISO date or free text, e.g. '2027 春季')")  # fmt: skip
        return None
    return st


def intake_student_record(data: Mapping, course_map: CourseMap | None = None) -> IntakeResult:
    """Validate one intake object and build the StudentRecord.

    Every problem is collected and raised together as RecordIntakeError. Courses
    are mapped by the table; a course with a failing grade (or W) counts as
    nothing; unmatched and review-listed names go to `record.unmapped`; an entry
    with explicit `categories` (course) / `category` (exam) is mapped manually.
    """
    cmap = course_map or load_course_map()
    errors: list[str] = []
    if not isinstance(data, Mapping):
        raise RecordIntakeError([f"top level must be a JSON object, got {type(data).__name__}"])
    for k in sorted(set(data) - _TOP):
        errors.append(f"{k}: unknown field; allowed: {', '.join(sorted(_TOP))}")
    if data.get("schema") != SCHEMA:
        errors.append(f"schema: must be {SCHEMA!r}, got {data.get('schema')!r}")
    sid = data.get("student_id")
    if not isinstance(sid, str) or not sid.strip():
        errors.append("student_id: required non-empty string (use a pseudonym, not a name)")
    synthetic = data.get("synthetic", False)
    if not isinstance(synthetic, bool):
        errors.append("synthetic: must be true or false")
    target = data.get("target_apply_by")
    if target is not None:
        try:
            date.fromisoformat(str(target))
        except ValueError:
            errors.append(f"target_apply_by: {target!r} is not an ISO date (YYYY-MM-DD); "
                          "leave it null while exploring")  # fmt: skip

    sections: dict[str, list] = {}
    for sec in _FIELDS:
        rows = data.get(sec, [])
        if not isinstance(rows, list):
            errors.append(f"{sec}: must be a list")
            rows = []
        good = []
        for i, e in enumerate(rows):
            where = f"{sec}[{i}]"
            if not isinstance(e, Mapping):
                errors.append(f"{where}: must be an object")
                continue
            bad = sorted(set(e) - _FIELDS[sec])
            if bad:
                errors.append(f"{where}: unknown field(s) {bad}; allowed: {sorted(_FIELDS[sec])}")
            miss = [f for f in _REQUIRED[sec] if not str(e.get(f) or "").strip()]
            if miss:
                errors.append(f"{where}: missing {miss}")
                continue
            good.append((where, e))
        sections[sec] = good

    completed: dict[str, list[str]] = {}
    planned: dict[str, list[str]] = {}
    online_flags: dict[str, list[bool]] = {}
    unmapped: dict[str, str] = {}
    entries: list[EntryMapping] = []

    def add(cat: str, status: str, evidence: str, expected=None, online: bool = False):
        if status == "completed":
            completed.setdefault(cat, []).append(evidence)
            online_flags.setdefault(cat, []).append(online)
        else:
            planned.setdefault(cat, []).append(f"{evidence}：預計 {expected}")

    degrees = []
    for where, e in sections["degrees"]:
        level, cip, year = e["level"], e.get("cip"), e.get("year")
        ok = True
        if level not in DEGREE_LEVELS:
            errors.append(f"{where}.level: {level!r} must be one of {DEGREE_LEVELS} "
                          "(專科／副學士 -> 'other')")  # fmt: skip
            ok = False
        if cip is not None and not re.fullmatch(r"\d{2}\.\d{4}", str(cip)):
            errors.append(f"{where}.cip: {cip!r} must look like '52.0201' (6-digit CIP 2020)")
            ok = False
        if year is not None and (not isinstance(year, int) or not 1950 <= year <= 2100):
            errors.append(f"{where}.year: {year!r} must be a year such as 2025")
            ok = False
        if ok:
            degrees.append(Degree(level, str(e["field"]), cip, year))
            if level in IMPLIES_BACHELOR:
                yr = f"，{year}" if year else ""
                add("bachelor_degree", "completed", f"學位：{DEGREE_ZH[level]} {e['field']}{yr}")

    for where, e in sections["courses"]:
        name = str(e["name"]).strip()
        st = _status(e, where, errors)
        credits, online = e.get("credits"), e.get("online", False)
        if credits is not None and (
            isinstance(credits, bool) or not isinstance(credits, int | float)
            or not 0 < credits <= 20
        ):  # fmt: skip
            errors.append(f"{where}.credits: {credits!r} must be a number in (0, 20]")
            continue
        if not isinstance(online, bool):
            errors.append(f"{where}.online: must be true or false")
            continue
        if st is None:
            continue
        if st == "planned" and e.get("grade") not in (None, ""):
            errors.append(f"{where}.grade: a planned course has no grade yet")
            continue
        passed = _passed(e.get("grade"), f"{where}.grade", errors) if st == "completed" else True
        if passed is None:
            continue
        bits = [f"{credits:g} 學分" if credits is not None else "",
                str(e.get("grade") or "").strip(), "線上" if online else ""]  # fmt: skip
        evidence = name + ("（" + "，".join(b for b in bits if b) + "）" if any(bits) else "")
        if not passed:
            entries.append(EntryMapping("courses", name, "not_passed", (),
                                        f"成績 {e.get('grade')} 未通過，不列為已完成"))  # fmt: skip
            continue
        if "categories" in e:
            cats = e["categories"]
            if not isinstance(cats, list):
                errors.append(f"{where}.categories: must be a list ([] = no category)")
                continue
            checked = [_category(c, ("course",), f"{where}.categories", cmap, errors) for c in cats]
            if None in checked:
                continue
            source, note = ("manual", "手動對應") if cats else ("none", "學生確認不對應任何類別")
            cats = tuple(checked)
        else:
            source, cats, note = cmap.map_course(name)
        entries.append(EntryMapping("courses", name, source, cats, note))
        if not cats:
            if source in ("review", "unmatched"):
                unmapped[name] = note
            continue
        for c in cats:
            add(c, st, evidence, e.get("expected"), online)

    for where, e in sections["exams"]:
        name = str(e["name"]).strip()
        st = _status(e, where, errors)
        if st is None:
            continue
        if "category" in e:
            cat = _category(e["category"], ("test", "language"), f"{where}.category", cmap, errors)
            if cat is None:
                continue
            source, cats, note = "manual", (cat,), "手動對應"
        else:
            source, cats, note = cmap.map_exam(name)
        entries.append(EntryMapping("exams", name, source, cats, note))
        if not cats:
            unmapped[name] = note
            continue
        score = str(e.get("score") or "").strip()
        when = str(e.get("date") or "").strip()
        detail = "，".join(b for b in (score, when) if b)
        add(cats[0], st, name + (f"（{detail}）" if detail else ""), e.get("expected"))

    for where, e in sections["experience"]:
        st = _status(e, where, errors)
        cat = _category(e["category"], ("experience",), f"{where}.category", cmap, errors)
        months = e.get("months")
        if months is not None and (isinstance(months, bool) or not isinstance(months, int | float)
                                   or months < 0):  # fmt: skip
            errors.append(f"{where}.months: {months!r} must be a non-negative number")
            continue
        if st is None or cat is None:
            continue
        ev = str(e["description"]).strip() + (f"（{months:g} 個月）" if months is not None else "")
        add(cat, st, ev, e.get("expected"))

    for where, e in sections["other"]:
        st = _status(e, where, errors)
        cat = _category(e["category"], _OTHER_KINDS, f"{where}.category", cmap, errors)
        if st is None or cat is None:
            continue
        add(cat, st, str(e["evidence"]).strip(), e.get("expected"))

    if errors:
        raise RecordIntakeError(errors)
    record = StudentRecord(
        student_id=sid,
        degrees=degrees,
        completed={c: "；".join(v) for c, v in completed.items()},
        planned={c: "；".join(v) for c, v in planned.items() if c not in completed},
        target_apply_by=target,
        online_only={c for c, flags in online_flags.items() if all(flags)},
        unmapped=unmapped,
    )
    return IntakeResult(record, tuple(entries), cmap.version, cmap.status, synthetic)


def load_student_record(path: str | Path, course_map: CourseMap | None = None) -> IntakeResult:
    """Read one intake JSON file (UTF-8). Real student files belong in data/private/."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordIntakeError([f"{path}: not valid JSON ({exc})"]) from exc
    return intake_student_record(data, course_map)
