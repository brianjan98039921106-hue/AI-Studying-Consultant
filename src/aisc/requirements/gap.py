"""C6 v0: deterministic comparison of published requirements with a student record.

Rules (docs/components/C6-requirements.md):
- every program passed in gets a RequirementGap: C6 never drops a program;
- an item is 'met' only if its category is in StudentRecord.completed, 'planned'
  if in StudentRecord.planned, 'unknown' if it has no category, else 'missing';
- completed evidence that is online only (StudentRecord.online_only) does not
  meet an item whose program states accepts_online=False (B-C6-11);
- 'before_target' is filled only when the student set target_apply_by, and lists
  missing required items that the program does not let admitted students finish
  later; C6 does not estimate how long a gap takes to close (no source for it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from aisc.contracts import RequirementGap, RequirementItem, StudentRecord


def load_taxonomy(path: str | Path) -> dict[str, set[str]]:
    """kind -> set of category keys."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {kind: set(cats) for kind, cats in spec["categories"].items()}
    seen: dict[str, str] = {}
    for kind, cats in out.items():
        for c in cats:
            if c in seen:
                raise ValueError(f"category {c!r} appears under {seen[c]!r} and {kind!r}")
            seen[c] = kind
    return out


def _check(items: Iterable[RequirementItem], record: StudentRecord,
           taxonomy: Mapping[str, set[str]] | None) -> None:  # fmt: skip
    if taxonomy is None:
        return
    every = set().union(*taxonomy.values())
    for it in items:
        if it.category is not None and it.category not in taxonomy.get(it.kind, set()):
            raise ValueError(f"{it.program_id}: category {it.category!r} is not a {it.kind!r}")
    unknown = (set(record.completed) | set(record.planned)) - every
    if unknown:
        raise ValueError(f"student record uses unknown categories {sorted(unknown)}")
    stray = set(record.online_only) - set(record.completed)
    if stray:
        raise ValueError(f"online_only categories {sorted(stray)} are not in completed")


def compute_gap(
    program_id: str,
    items: Iterable[RequirementItem],
    record: StudentRecord,
    taxonomy: Mapping[str, set[str]] | None = None,
) -> RequirementGap:
    items = list(items)
    if any(it.program_id != program_id for it in items):
        raise ValueError(f"items for another program passed to {program_id}")
    _check(items, record, taxonomy)
    met, planned, missing, unknown = [], [], [], []
    for it in items:
        if it.category is None:
            unknown.append(it)
        elif it.category in record.completed and not (
            it.accepts_online is False and it.category in record.online_only
        ):
            met.append(it)
        elif it.category in record.planned:
            planned.append(it)
        else:
            missing.append(it)
    before = []
    if record.target_apply_by is not None:
        before = [
            it
            for it in missing
            if it.level == "required" and it.completable_after_admission is not True
        ]
    return RequirementGap(
        program_id=program_id,
        met=tuple(met),
        planned=tuple(planned),
        missing=tuple(missing),
        unknown=tuple(unknown),
        none_published=not items,
        before_target=tuple(before),
    )


def compute_gaps(
    program_ids: Iterable[str],
    items: Iterable[RequirementItem],
    record: StudentRecord,
    taxonomy: Mapping[str, set[str]] | None = None,
) -> dict[str, RequirementGap]:
    """One gap per program id, in input order; programs without published
    requirements get `none_published=True` rather than being skipped."""
    by_prog: dict[str, list[RequirementItem]] = {}
    for it in items:
        by_prog.setdefault(it.program_id, []).append(it)
    ids = list(program_ids)
    stray = set(by_prog) - set(ids)
    if stray:
        raise ValueError(f"requirements for programs not in the list: {sorted(stray)[:5]}")
    return {pid: compute_gap(pid, by_prog.get(pid, []), record, taxonomy) for pid in ids}
