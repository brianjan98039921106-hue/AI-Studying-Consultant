"""C0.1 reference tables (W-14) and C0.2 occupation bridge (W-15): verification (L2).

All fixture files are built here in the official layouts with FICTIONAL titles
and numbers; they are format fixtures, not data.
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest

from aisc.catalog.occupation import (
    RIASEC,
    load_cip_soc_crosswalk,
    load_onet_interests,
    occupation_profile,
    soc_profiles,
)
from aisc.catalog.reference import (
    Scope,
    file_provenance,
    load_cip2020,
    load_ipeds_completions,
    load_ipeds_directory,
    load_scope,
    normalize_cip,
    read_table,
    school_cip_rows,
    select_cips,
)

ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path, header, rows, delimiter=",", encoding="utf-8") -> Path:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter)
    w.writerow(header)
    w.writerows(rows)
    path.write_bytes(buf.getvalue().encode(encoding))
    return path


def _xlsx(path: Path, header, rows, title_rows=0, sheet="Sheet1") -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for i in range(title_rows):
        ws.append([f"Fixture title row {i}"])
    ws.append(header)
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


CIP_HEADER = ["CIPFamily", "CIPCode", "Action", "TextChange", "CIPTitle", "CIPDefinition"]
CIP_ROWS = [
    ['="52"', '="52"', "No substantive changes", "No", "Fixture Business.", "def"],
    ['="52"', '="52.02"', "No substantive changes", "No", "Fixture Management.", "def"],
    ['="52"', '="52.0201"', "No substantive changes", "No", "Fixture Admin.", "def a"],
    ['="52"', '="52.0205"', "No substantive changes", "No", "Fixture Ops.", "def b"],
    ['="52"', '="52.0299"', "Moved to", "No", "Fixture Moved.", "gone"],
    ['="30"', '="30.7102"', "New", "No", "Fixture Analytics.", "def c"],
    ['="27"', '="27.0501"', "No substantive changes", "No", "Fixture Stats.", "def d"],
]


@pytest.fixture
def cip_file(tmp_path):
    return _csv(tmp_path / "CIPCode2020.csv", CIP_HEADER, CIP_ROWS)


def _scope(families):
    return Scope("test", 7, 1, families)


# --- normalisation and table reading ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "code"),
    [('="52.0201"', "52.0201"), ("52.0201", "52.0201"), ("5.0201", "05.0201"),
     ("52.02", "52.02"), ("52", "52"), ("1", "01"), (" 30.7102 ", "30.7102")],
)  # fmt: skip
def test_normalize_cip(raw, code):
    assert normalize_cip(raw) == code


@pytest.mark.parametrize("raw", ["", "52.201", "52.02011", "ab.cdef", "520201", "99-9999"])
def test_normalize_cip_rejects_malformed(raw):
    with pytest.raises(ValueError):
        normalize_cip(raw)


def test_read_table_formats(tmp_path):
    rows = [["1", "x"], ["2", "y"]]
    c = _csv(tmp_path / "a.csv", ["ID", "V"], rows)
    t = _csv(tmp_path / "a.txt", ["ID", "V"], rows, delimiter="\t")
    x = _xlsx(tmp_path / "a.xlsx", ["ID", "V"], [[1, "x"], [2, "y"]], title_rows=2)
    for p in (c, t, x):
        assert read_table(p, required=["ID", "V"]) == [{"ID": "1", "V": "x"}, {"ID": "2", "V": "y"}]
    with pytest.raises(ValueError, match="missing columns"):
        read_table(c, required=["ID", "W"])
    with pytest.raises(ValueError, match="no worksheet"):
        read_table(x, required=["NOPE"])


def test_read_table_zip_prefers_revised_member(tmp_path):
    z = tmp_path / "c.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("c2024_a.csv", "ID,V\n1,old\n")
        zf.writestr("c2024_a_rv.csv", "ID,V\n1,revised\n")
    assert read_table(z, prefer="_rv")[0]["V"] == "revised"
    assert read_table(z)[0]["V"] == "old"


def test_read_table_falls_back_to_cp1252(tmp_path):
    p = _csv(tmp_path / "hd.csv", ["UNITID", "INSTNM", "STABBR"],
             [["1", "Université Fixture", "NY"]], encoding="cp1252")  # fmt: skip
    assert read_table(p)[0]["INSTNM"] == "Université Fixture"


def test_provenance(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"abc")
    prov = file_provenance(p, "https://nces.ed.gov/x.csv", "2026-10-26")
    assert prov.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert prov.to_dict()["file_name"] == "f.csv"
    with pytest.raises(ValueError):
        file_provenance(p, "http://nces.ed.gov/x.csv", "2026-10-26")
    with pytest.raises(ValueError):
        file_provenance(p, "https://nces.ed.gov/x.csv", "26/10/2026")


# --- CIP 2020 and scope -----------------------------------------------------------


def test_load_cip2020_drops_moved_codes_and_cleans_titles(cip_file):
    cip = load_cip2020(cip_file)
    assert "52.0299" not in cip
    assert cip["52.0201"].title == "Fixture Admin"
    assert {"52", "52.02", "52.0201", "52.0205", "30.7102", "27.0501"} == set(cip)


def test_load_cip2020_from_xlsx(tmp_path):
    rows = [[c.strip('="') for c in r] for r in CIP_ROWS]
    x = _xlsx(tmp_path / "CIPCode2020.xlsx", CIP_HEADER, rows, title_rows=1)
    assert set(load_cip2020(x)) == {"52", "52.02", "52.0201", "52.0205", "30.7102", "27.0501"}


def test_select_cips_longest_prefix_and_unknown_prefix(cip_file):
    cip = load_cip2020(cip_file)
    sel = select_cips(cip, _scope({"mgmt": ["52.02"], "ops": ["52.0205"], "an": ["30.71"]}))
    assert sel == {"52.0201": "mgmt", "52.0205": "ops", "30.7102": "an"}
    with pytest.raises(ValueError, match="not in CIP 2020"):
        select_cips(cip, _scope({"x": ["14.37"]}))


def test_real_mvp_scope_file_is_well_formed():
    scope = load_scope(ROOT / "data" / "reference" / "mvp_scope.json")
    assert scope.award_level == 7 and scope.majornum == 1
    prefixes = [p for ps in scope.families.values() for p in ps]
    assert len(prefixes) == len(set(prefixes))
    # no family prefix is nested inside another family's prefix (unambiguous families)
    owner = {p: f for f, ps in scope.families.items() for p in ps}
    for a in prefixes:
        for b in prefixes:
            if a != b and b.startswith(a):
                assert owner[a] == owner[b], (a, b)


def test_load_scope_rejects_bad_prefix(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"version": "x", "award_level": 7, "majornum": 1,
                             "families": [{"name": "f", "prefixes": ["52.2"]}]}))  # fmt: skip
    with pytest.raises(ValueError, match="bad CIP prefix"):
        load_scope(p)


# --- IPEDS --------------------------------------------------------------------------

C_HEADER = ["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"]


def test_load_ipeds_completions_filters(tmp_path):
    rows = [
        ["100", "52.0201", "1", "7", "40"],
        ["100", "52.0201", "2", "7", "3"],  # second major
        ["100", "52.0201", "1", "5", "90"],  # bachelor's
        ["100", "99", "1", "7", "500"],  # grand total
        ["200", "30.7102", "1", "7", "0"],  # no awards
        ["200", "27.0501", "1", "7", "12"],
        ["300", "14.3701", "1", "7", "8"],  # out of scope
    ]
    p = _csv(tmp_path / "c.csv", C_HEADER, rows)
    got = load_ipeds_completions(p, ["52.0201", "30.7102", "27.0501"])
    assert [(c.unitid, c.cip, c.awards) for c in got] == [
        ("100", "52.0201", 40),
        ("200", "27.0501", 12),
    ]
    dup = _csv(tmp_path / "d.csv", C_HEADER, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_ipeds_completions(dup, ["52.0201"])


def test_school_cip_rows_join(tmp_path, cip_file):
    cip = load_cip2020(cip_file)
    c = load_ipeds_completions(
        _csv(tmp_path / "c.csv", C_HEADER, [["100", "52.0201", "1", "7", "40"]]), ["52.0201"]
    )
    hd = _csv(tmp_path / "hd.csv", ["UNITID", "INSTNM", "CITY", "STABBR", "WEBADDR"],
              [["100", "Fixture University", "Town", "NY", "www.fixture.example"]])  # fmt: skip
    directory = load_ipeds_directory(hd)
    rows = school_cip_rows(c, directory, cip, {"52.0201": "mgmt"}, "2024")
    assert rows == [{
        "unitid": "100", "institution": "Fixture University", "city": "Town", "state": "NY",
        "web": "www.fixture.example", "cip": "52.0201", "cip_title": "Fixture Admin",
        "family": "mgmt", "masters_awards": 40, "completions_year": "2024",
    }]  # fmt: skip
    with pytest.raises(ValueError, match="missing from the directory"):
        school_cip_rows(c, {}, cip, {"52.0201": "mgmt"}, "2024")


# --- occupation bridge (W-15) ------------------------------------------------------

XW_HEADER = ["CIP2020Code", "CIP2020Title", "SOC2018Code", "SOC2018Title"]
ONET_HEADER = ["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "Date",
               "Domain Source"]  # fmt: skip


def _interest_rows(code, values, extra=True):
    rows = [[code, f"1.B.1.{i}", name, "OI", f"{v:.2f}", "11/2023", "Analyst"]
            for i, (name, v) in enumerate(zip(RIASEC, values, strict=True))]  # fmt: skip
    if extra:
        rows.append([code, "1.B.3.a", "First Interest High-Point", "IH", "2.00", "11/2023", "x"])
        rows.append([code, "1.B.2.a", "Fixture Basic Interest", "OI", "6.00", "11/2023", "x"])
    return rows


def test_crosswalk_drops_no_match_and_validates(tmp_path):
    rows = [
        ["52.0201", "Fixture Admin", "11-1021", "Fixture GM"],
        ["52.0201", "Fixture Admin", "13-1111", "Fixture Analyst"],
        ["30.7102", "Fixture Analytics", "99-9999", "NO MATCH"],
    ]
    xw = load_cip_soc_crosswalk(_xlsx(tmp_path / "xw.xlsx", XW_HEADER, rows, sheet="CIP-SOC"))
    assert xw == {"52.0201": {"11-1021", "13-1111"}, "30.7102": set()}
    bad = _csv(tmp_path / "bad.csv", XW_HEADER, [["52.0201", "t", "111021", "t"]])
    with pytest.raises(ValueError, match="malformed SOC"):
        load_cip_soc_crosswalk(bad)


def test_onet_interests_keeps_only_riasec_oi(tmp_path):
    p = _csv(tmp_path / "Interests.txt", ONET_HEADER,
             _interest_rows("15-2051.00", [1, 7, 2, 3, 4, 5]), delimiter="\t")  # fmt: skip
    got = load_onet_interests(p)
    assert got == {"15-2051.00": dict(zip(RIASEC, [1.0, 7.0, 2.0, 3.0, 4.0, 5.0], strict=True))}
    bad = _csv(tmp_path / "bad.txt", ONET_HEADER,
               _interest_rows("15-2051.00", [1, 8, 2, 3, 4, 5]), delimiter="\t")  # fmt: skip
    with pytest.raises(ValueError, match="outside 1-7"):
        load_onet_interests(bad)
    part = _csv(tmp_path / "part.txt", ONET_HEADER,
                _interest_rows("15-2051.00", [1, 2, 3, 4, 5, 6])[:3], delimiter="\t")  # fmt: skip
    with pytest.raises(ValueError, match="missing some RIASEC"):
        load_onet_interests(part)


def test_occupation_profile_averages_and_rescales():
    interests = {
        "11-1021.00": dict(zip(RIASEC, [1, 1, 1, 1, 7, 4], strict=True)),
        "13-1111.00": dict(zip(RIASEC, [1, 5, 1, 1, 5, 7], strict=True)),
        "13-1111.01": dict(zip(RIASEC, [3, 7, 1, 1, 5, 7], strict=True)),  # detail occupation
    }
    socs = soc_profiles(interests)
    assert socs["13-1111"]["Realistic"] == 2.0 and socs["13-1111"]["Investigative"] == 6.0
    xw = {"52.0201": {"11-1021", "13-1111", "11-9999"}, "30.7102": set()}
    prof = occupation_profile("52.0201", xw, socs)
    # equal weight over the two SOCs with O*NET data, then (v - 1) / 6
    assert prof.vector["riasec_r"] == pytest.approx(((1 + 2) / 2 - 1) / 6)
    assert prof.vector["riasec_e"] == pytest.approx(((7 + 5) / 2 - 1) / 6)
    assert prof.vector["riasec_c"] == pytest.approx(((4 + 7) / 2 - 1) / 6)
    assert prof.missing == ("11-9999",) and prof.coverage == pytest.approx(2 / 3)
    assert all(0 <= v <= 1 for v in prof.vector.values())
    empty = occupation_profile("30.7102", xw, socs)
    assert empty.vector == {} and empty.coverage == 0.0
    with pytest.raises(ValueError, match="malformed O"):
        soc_profiles({"131111": interests["13-1111.00"]})


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C0-1: the occupation bridge cannot tell apart programs that share a CIP code",
)
def test_probe_same_cip_programs_get_different_occupation_vectors():
    """A theory-heavy and an applied program both reported under one CIP code
    should not get the same x_occ (C0.3 curriculum data is needed to separate them)."""
    socs = soc_profiles({"15-2041.00": dict(zip(RIASEC, [1, 7, 2, 2, 3, 5], strict=True))})
    xw = {"27.0501": {"15-2041"}}
    theory = occupation_profile("27.0501", xw, socs)  # e.g. MS Statistics (theory track)
    applied = occupation_profile("27.0501", xw, socs)  # e.g. MS Applied Statistics
    assert theory.vector != applied.vector


def test_build_reference_script_end_to_end(tmp_path, cip_file):
    import subprocess
    import sys

    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps({"version": "fixture", "award_level": 7, "majornum": 1,
                                 "families": [{"name": "mgmt", "prefixes": ["52.02"]},
                                              {"name": "an", "prefixes": ["30.71"]}]}))  # fmt: skip
    xw = _xlsx(tmp_path / "xw.xlsx", XW_HEADER, [
        ["52.0201", "t", "11-1021", "t"], ["52.0205", "t", "11-3071", "t"],
        ["30.7102", "t", "99-9999", "NO MATCH"],
    ])  # fmt: skip
    onet = _csv(tmp_path / "Interests.txt", ONET_HEADER,
                _interest_rows("11-1021.00", [1, 2, 1, 2, 7, 4]), delimiter="\t")  # fmt: skip
    comp = tmp_path / "C2024_A.zip"
    with zipfile.ZipFile(comp, "w") as zf:
        zf.writestr("c2024_a.csv", "UNITID,CIPCODE,MAJORNUM,AWLEVEL,CTOTALT\n"
                    "100,52.0201,1,7,40\n100,52.0205,1,7,5\n200,30.7102,1,7,9\n")  # fmt: skip
    hd = tmp_path / "HD2024.zip"
    with zipfile.ZipFile(hd, "w") as zf:
        zf.writestr("hd2024.csv", "UNITID,INSTNM,CITY,STABBR,WEBADDR\n"
                    "100,Fixture U,A,NY,a.example\n200,Fixture State,B,TX,b.example\n")  # fmt: skip
    out, report = tmp_path / "ref", tmp_path / "report.md"
    cmd = [sys.executable, str(ROOT / "scripts" / "build_reference.py"),
           "--cip", str(cip_file), "--crosswalk", str(xw), "--interests", str(onet),
           "--completions", str(comp), "--directory", str(hd), "--retrieved-at", "2026-10-26",
           "--onet-release", "fixture", "--scope", str(scope), "--out", str(out),
           "--report", str(report)]  # fmt: skip
    import os

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0, res.stderr
    assert "3 CIPs in scope, 3 offered, 3 school x CIP rows" in res.stdout
    src = json.loads((out / "SOURCES.json").read_text())
    assert src["ipeds_year"] == "2024" and len(src["inputs"]) == 5
    assert all(v["source_url"].startswith("https://") for v in src["inputs"].values())
    with open(out / "occupation_profiles_mvp.csv", encoding="utf-8") as fh:
        profiles = list(csv.DictReader(fh))
    by_cip = {r["cip"]: r for r in profiles}
    assert by_cip["52.0201"]["riasec_e"] == "1.0000"  # (7 - 1) / 6
    assert by_cip["52.0205"]["riasec_e"] == "" and by_cip["52.0205"]["missing_soc"] == "11-3071"
    assert by_cip["30.7102"]["n_soc"] == "0"
    text = report.read_text(encoding="utf-8")
    assert "30.7102" in text and "52.0205" in text  # CIPs without a profile are listed


def test_build_reference_all_fields_scope(tmp_path, cip_file):
    """Rollout v3 path: several award levels, families = 2-digit series, graduate-only."""
    import os
    import subprocess
    import sys

    scope = tmp_path / "all.json"
    scope.write_text(json.dumps({"version": "fx-v3", "award_levels": [7, 17, 18], "majornum": 1,
                                 "families": "cip_series", "graduate_only": True,
                                 "output_suffix": "all"}))  # fmt: skip
    xw = _xlsx(tmp_path / "xw.xlsx", XW_HEADER, [["52.0201", "t", "11-1021", "t"]])
    onet = _csv(tmp_path / "Interests.txt", ONET_HEADER,
                _interest_rows("11-1021.00", [1, 2, 1, 2, 7, 4]), delimiter="\t")  # fmt: skip
    comp = _csv(tmp_path / "c2024_a.csv", ["UNITID", "CIPCODE", "MAJORNUM", "AWLEVEL", "CTOTALT"],
                [["100", "52.0201", "1", "7", "40"], ["100", "52.0201", "1", "17", "3"],
                 ["200", "30.7102", "1", "5", "9"], ["100", "52.0205", "2", "7", "4"]])  # fmt: skip
    hd = _csv(tmp_path / "hd2024.csv", ["UNITID", "INSTNM", "CITY", "STABBR", "WEBADDR"],
              [["100", "Fixture U", "A", "NY", "a"],
               ["200", "Fixture State", "B", "TX", "b"]])  # fmt: skip
    out = tmp_path / "ref"
    cmd = [sys.executable, str(ROOT / "scripts" / "build_reference.py"),
           "--cip", str(cip_file), "--crosswalk", str(xw), "--interests", str(onet),
           "--completions", str(comp), "--directory", str(hd), "--retrieved-at", "2026-10-26",
           "--onet-release", "fixture", "--scope", str(scope), "--out", str(out),
           "--report", str(tmp_path / "r.md")]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert res.returncode == 0, res.stderr
    # 30.7102 has only a level-5 award and 52.0205 only a second major: both out
    assert "1 CIPs in scope, 2 school x CIP x level rows" in res.stdout
    with open(out / "cip2020_all.csv", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert (row["cip"], row["family"], row["awards_level_7"], row["awards_level_17"],
            row["awards_level_18"]) == ("52.0201", "52", "40", "3", "0")  # fmt: skip
    assert (out / "SOURCES_all.json").exists() and not (out / "SOURCES.json").exists()


# --- probes on the committed real reference tables -----------------------------------


def _ref(name):
    with open(ROOT / "data" / "reference" / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.probe
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="B-C0-8: postsecondary-teacher SOCs in the crosswalk pull x_occ toward Social",
)
def test_probe_teacher_occupations_do_not_dominate_occupation_profiles():
    """Most master's graduates do not become professors, yet the crosswalk maps most
    academic CIPs to '... Teachers, Postsecondary' (25-1xxx). Dropping those SOCs
    should not move any profile's Social score by more than 0.10."""
    soc = {r["soc"]: float(r["social"]) for r in _ref("soc_profiles_mvp.csv")}
    offered = {r["cip"] for r in _ref("ipeds_masters_mvp.csv")}
    pairs: dict[str, list[str]] = {}
    for r in _ref("cip_soc_mvp.csv"):
        if r["cip"] in offered and r["soc"] in soc:
            pairs.setdefault(r["cip"], []).append(r["soc"])
    shifts = []
    for socs in pairs.values():
        rest = [s for s in socs if not s.startswith("25-1")]
        if rest and len(rest) < len(socs):
            all_s = sum(soc[s] for s in socs) / len(socs)
            shifts.append(abs(all_s - sum(soc[s] for s in rest) / len(rest)) / 6)
    assert shifts, "expected CIPs mixing teacher and non-teacher SOCs"
    assert max(shifts) <= 0.10


def test_committed_reference_tables_are_consistent():
    """L2 on the committed tables: provenance present, joins closed, ranges valid."""
    src = json.loads((ROOT / "data" / "reference" / "SOURCES.json").read_text())
    assert {"cip2020", "cip_soc_crosswalk", "onet_interests", "ipeds_completions",
            "ipeds_directory"} <= set(src["inputs"])  # fmt: skip
    for v in src["inputs"].values():
        assert v["source_url"].startswith("https://") and len(v["sha256"]) == 64
    cips = {r["cip"]: r for r in _ref("cip2020_mvp.csv")}
    rows = _ref("ipeds_masters_mvp.csv")
    assert {r["cip"] for r in rows} <= set(cips)
    assert all(int(r["masters_awards"]) > 0 for r in rows)
    assert len({(r["unitid"], r["cip"]) for r in rows}) == len(rows)
    for r in _ref("occupation_profiles_mvp.csv"):
        assert r["cip"] in cips
        vals = [
            r[k] for k in ("riasec_r", "riasec_i", "riasec_a", "riasec_s", "riasec_e", "riasec_c")
        ]
        assert all(v == "" for v in vals) or all(0 <= float(v) <= 1 for v in vals)


def test_scope_covers_other_graduate_award_levels():
    """B-C0-10 (fixed by rollout v3, W-65/W-70): the all-fields scope includes
    professional and research doctorates and post-baccalaureate certificates."""
    spec = json.loads((ROOT / "data" / "reference" / "all_scope.json").read_text())
    levels = spec.get("award_levels", [spec.get("award_level")])
    assert {6, 7, 8, 17, 18} <= set(levels)


def test_committed_all_field_tables_are_consistent():
    """L2 on rollout v3 (W-70): provenance, joins, ranges, and agreement with v1."""
    src_all = json.loads((ROOT / "data" / "reference" / "SOURCES_all.json").read_text())
    src_mvp = json.loads((ROOT / "data" / "reference" / "SOURCES.json").read_text())
    for key, v in src_mvp["inputs"].items():  # same official files as rollout v1
        assert src_all["inputs"][key]["sha256"] == v["sha256"]
    spec = json.loads((ROOT / "data" / "reference" / "all_scope.json").read_text())
    cips = {r["cip"]: r for r in _ref("cip2020_all.csv")}
    level_cols = [f"awards_level_{lv}" for lv in spec["award_levels"]]
    assert all(sum(int(r[c]) for c in level_cols) > 0 for r in cips.values())
    assert all(r["family"] == r["cip"][:2] for r in cips.values())
    rows = _ref("ipeds_graduate_all.csv")
    assert {r["cip"] for r in rows} == set(cips)
    assert {int(r["award_level"]) for r in rows} <= set(spec["award_levels"])
    assert len({(r["unitid"], r["cip"], r["award_level"]) for r in rows}) == len(rows)
    # rollout v1 is a subset: same master's rows, same occupation profiles
    v1 = {(r["unitid"], r["cip"], int(r["masters_awards"])) for r in _ref("ipeds_masters_mvp.csv")}
    v3 = {(r["unitid"], r["cip"], int(r["awards"])) for r in rows
          if r["award_level"] == "7" and r["cip"] in {c for _, c, _ in v1}}  # fmt: skip
    assert v1 == v3
    occ_all = {r["cip"]: r for r in _ref("occupation_profiles_all.csv")}
    for r in _ref("occupation_profiles_mvp.csv"):
        if r["cip"] in occ_all:
            assert {k: r[k] for k in r if k.startswith("riasec")} == {
                k: occ_all[r["cip"]][k] for k in r if k.startswith("riasec")
            }
    keys = [r["key"] for r in _ref("basic_interest_map.csv")]
    for r in _ref("occupation_basic_interests_all.csv"):
        assert r["cip"] in cips
        vals = [r[k] for k in keys]
        assert all(v == "" for v in vals) or all(0 <= float(v) <= 1 for v in vals)


def test_committed_basic_interest_tables():
    """W-64: 41 specific interest areas, each linked to one or two RIASEC types."""
    bmap = _ref("basic_interest_map.csv")
    assert len(bmap) == 41 and len({r["key"] for r in bmap}) == 41
    assert all(1 <= len(r["riasec"].split()) <= 2 for r in bmap)
    keys = [r["key"] for r in bmap]
    for r in _ref("occupation_basic_interests_mvp.csv"):
        vals = [r[k] for k in keys]
        assert all(v == "" for v in vals) or all(0 <= float(v) <= 1 for v in vals)
    src = json.loads((ROOT / "data" / "reference" / "SOURCES.json").read_text())
    assert "onet_specific_interest_areas" in src["inputs"]


def test_basic_interest_loaders(tmp_path):
    from aisc.catalog.occupation import (
        basic_interest_key,
        load_basic_interest_map,
        load_onet_basic_interests,
    )

    assert basic_interest_key("Applied Arts and Design") == "bi_applied_arts_and_design"
    assert basic_interest_key("Mechanics/Electronics") == "bi_mechanics_electronics"
    mp = _csv(tmp_path / "map.csv",
              ["Specific Interest Areas Element ID", "Specific Interest Areas Element Name",
               "Career Interest Types Element ID", "Career Interest Types Element Name"],
              [["1.B.3.a", "Fixture Area A", "1.B.1.a", "Realistic"],
               ["1.B.3.b", "Fixture Area B", "1.B.1.b", "Investigative"],
               ["1.B.3.b", "Fixture Area B", "1.B.1.d", "Social"]])  # fmt: skip
    bmap = load_basic_interest_map(mp)
    assert bmap == {"Fixture Area A": ("Realistic",), "Fixture Area B": ("Investigative", "Social")}
    rows = [["15-2051.00", "t", "1.B.3.a", "Fixture Area A", "OI", "Occupational Interests", "2.50",
             "05/2026", "AI/Expert"],
            ["15-2051.00", "t", "1.B.3.a", "Fixture Area A", "DS", "Summary Display Rank", "1.00",
             "05/2026", "AI/Expert"],
            ["15-2051.00", "t", "1.B.3.b", "Fixture Area B", "OI", "Occupational Interests", "6.00",
             "05/2026", "AI/Expert"]]  # fmt: skip
    hdr = ["O*NET-SOC Code", "Title", "Element ID", "Element Name", "Scale ID", "Scale Name",
           "Data Value", "Date", "Domain Source"]  # fmt: skip
    got = load_onet_basic_interests(_csv(tmp_path / "sia.csv", hdr, rows), tuple(bmap))
    assert got == {"15-2051.00": {"Fixture Area A": 2.5, "Fixture Area B": 6.0}}
    socs = soc_profiles(got, tuple(bmap))
    prof = occupation_profile("27.0501", {"27.0501": {"15-2051"}}, socs, ("a", "b"), tuple(bmap))
    assert prof.vector == {"a": pytest.approx(0.25), "b": pytest.approx(5 / 6)}
    with pytest.raises(ValueError, match="same length"):
        occupation_profile("27.0501", {}, socs, ("a",), tuple(bmap))


def test_interest_area_clusters_cover_all_41_areas_once():
    spec = json.loads((ROOT / "data" / "reference" / "interest_area_clusters.json").read_text())
    keys = [a["key"] for c in spec["clusters"] for a in c["areas"]]
    assert len(keys) == len(set(keys)) == 41
    assert set(keys) == {r["key"] for r in _ref("basic_interest_map.csv")}
    assert spec["k"] == len(spec["clusters"]) and spec["bootstrap_mean_ari"] >= 0.60
