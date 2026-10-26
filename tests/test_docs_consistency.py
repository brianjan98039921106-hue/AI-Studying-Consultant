"""Documentation consistency (runs in CI).

Single sources of truth:
    blind spots / assumptions -> docs/REGISTER.md
    findings (F-numbers)      -> results/FINDINGS.md
    work-item status          -> docs/ROADMAP.md
    gate status               -> docs/VALIDATION.md §5
These tests catch references that point nowhere and statuses that contradict
the probe tests.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "docs" / "REGISTER.md"
FINDINGS = ROOT / "results" / "FINDINGS.md"
ROADMAP = ROOT / "docs" / "ROADMAP.md"
VALIDATION = ROOT / "docs" / "VALIDATION.md"

REG_ID = r"(?:[AB]-C\d-\d+|S-\d+)"
DOC_FILES = sorted(
    [*ROOT.joinpath("docs").rglob("*.md"), *ROOT.joinpath("results").glob("*.md")]
    + [ROOT / name for name in ("README.md", "CLAUDE.md", "CONTRIBUTING.md")]
    + [ROOT / "data" / "README.md"]
)
CODE_FILES = sorted(
    [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").glob("*.py")]
    + [*ROOT.joinpath("scripts").glob("*.py")]
)


def _register_rows() -> dict[str, list[str]]:
    rows = {}
    for line in REGISTER.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"^\| ({REG_ID}) \|", line)
        if m:
            assert m.group(1) not in rows, f"duplicate register id {m.group(1)}"
            rows[m.group(1)] = [c.strip() for c in line.strip().strip("|").split("|")]
    return rows


def test_register_ids_unique_and_present():
    rows = _register_rows()
    assert len(rows) > 50


def test_every_referenced_register_id_is_defined():
    defined = set(_register_rows())
    missing = {}
    for f in DOC_FILES + CODE_FILES:
        if f.name == "test_docs_consistency.py":
            continue
        for ref in re.findall(rf"(?<![\w-])({REG_ID})(?!\d)", f.read_text(encoding="utf-8")):
            if ref not in defined:
                missing.setdefault(ref, set()).add(str(f.relative_to(ROOT)))
    assert not missing, f"referenced but not in REGISTER: {missing}"


def _probe_reasons() -> list[tuple[str, str, str]]:
    """(file, B-id, reason) for every xfail probe marker."""
    out = []
    for f in ROOT.joinpath("tests").glob("test_*.py"):
        text = f.read_text(encoding="utf-8")
        for reason in re.findall(r'reason="([^"]*)"', text):
            m = re.match(r"(B-C\d-\d+)", reason)
            if m:
                out.append((f.name, m.group(1), reason))
    return out


def test_failing_probes_are_not_marked_fixed_in_register():
    """A probe still xfail-ing must not have a 🟢 register status, unless the probe
    only documents a kept reference implementation (reason says 'reference')."""
    rows = _register_rows()
    probes = _probe_reasons()
    assert probes, "expected at least one probe"
    contradictions = []
    for fname, bid, reason in probes:
        assert bid in rows, f"{fname}: probe cites unknown {bid}"
        status = rows[bid][-1]
        if "reference" not in reason and status.startswith("🟢"):
            contradictions.append((fname, bid, status))
    assert not contradictions, f"probe still failing but register says fixed: {contradictions}"


def test_every_probe_marker_requires_assertion_failure():
    """CLAUDE.md rule: probes use raises=AssertionError so code errors are not hidden."""
    offenders = []
    for f in ROOT.joinpath("tests").glob("test_*.py"):
        text = f.read_text(encoding="utf-8")
        for block in re.findall(r"xfail\((.*?)\)", text, re.S):
            if "strict=True" in block and "raises=AssertionError" not in block:
                offenders.append(f.name)
    assert not offenders, offenders


def test_every_referenced_finding_exists():
    defined = set(re.findall(r"^\| (F\d+[a-c]?) \|", FINDINGS.read_text(encoding="utf-8"), re.M))
    assert {"F1", "F29"} <= defined
    missing = {}
    for f in DOC_FILES:
        for ref in set(re.findall(r"(?<![\w-])(F\d{1,2}[a-c]?)(?!\w)", f.read_text("utf-8"))):
            if ref not in defined:
                missing.setdefault(ref, set()).add(str(f.relative_to(ROOT)))
    assert not missing, f"findings cited but not in results/FINDINGS.md: {missing}"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    broken = []
    for link in re.findall(r"\]\(([^)\s]+)\)", doc.read_text(encoding="utf-8")):
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not (doc.parent / link.split("#")[0]).exists():
            broken.append(link)
    assert not broken, broken


def test_roadmap_work_items_unique():
    ids = re.findall(r"^\| (?:\*\*)?(W-\d+[a-z]?)(?:\*\*)?", ROADMAP.read_text("utf-8"), re.M)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert ids and not dupes, dupes


def test_validation_status_has_one_row_per_component():
    section = VALIDATION.read_text(encoding="utf-8").split("## 5. 目前狀態")[1]
    comps = re.findall(r"^\| (C[0-5]) \|", section, re.M)
    assert sorted(comps) == [f"C{i}" for i in range(6)], comps
