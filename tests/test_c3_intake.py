"""C3.1 data intake and the time-split validation script (L2). All data here are
synthetic and written to pytest's temporary directory."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aisc.feasibility.intake import fit_from_files, load_admission_records, load_official_rates

ROOT = Path(__file__).resolve().parents[1]
HEAD = ["record_id", "year", "program_id", "cip", "program_type", "source", "outcome",
        "group", "x_gpa_pct", "x_toefl"]  # fmt: skip


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _synthetic_records(tmp_path, n_prog=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for i in range(n_prog):
        cip = ("52.1301", "14.3501", "9.0101")[i % 3]
        a = (-1.0, 0.0, 0.5)[i % 3] + rng.normal(0, 0.3)
        for year in (2024, 2025):
            for _ in range(25):
                gpa, toefl = rng.normal(80, 10), rng.normal(100, 8)
                z = a + 0.08 * (gpa - 80) + 0.05 * (toefl - 100)
                y = rng.random() < 1 / (1 + np.exp(-z))
                k += 1
                rows.append([f"R{k:04d}", year, f"P{i:02d}", cip,
                             "interview" if i % 2 else "standard", "complete",
                             "admit" if y else "reject", "taiwan" if k % 3 else "intl",
                             "" if k % 17 == 0 else f"{gpa:.1f}", f"{toefl:.0f}"])  # fmt: skip
    return _write(tmp_path / "records.csv", HEAD, rows)


def test_load_standardises_imputes_and_groups(tmp_path):
    data = load_admission_records(_synthetic_records(tmp_path))
    rep = data.report
    assert rep.n_rows == 600 and rep.by_source == {"complete": 600, "self_report": 0}
    assert rep.features == ("x_gpa_pct", "x_toefl", "x_gpa_pct_missing")
    assert rep.missing["x_gpa_pct"] == 600 // 17 and rep.standardized_on == "complete records"
    assert data.families["P02"] == "09.01"  # unpadded CIP normalised
    assert data.types["P01"] == "interview" and set(rep.groups) == {"taiwan", "intl"}
    z = np.array([r.features[1] for r in data.records])
    assert abs(z.mean()) < 1e-9 and abs(z.std(ddof=1) - 1) < 1e-9


@pytest.mark.parametrize("bad_header", ["name", "student_id", "Email", "學號"])
def test_identifier_columns_are_refused(tmp_path, bad_header):
    row = ["R1", 2025, "P1", "52.1301", "", "complete", "admit", "", "80", "100", "x"]
    p = _write(tmp_path / "r.csv", [*HEAD, bad_header], [row])
    with pytest.raises(ValueError, match="identifiers"):
        load_admission_records(p)


@pytest.mark.parametrize("cell", ["someone@example.com", "A123456789", "0912-345-678"])
def test_identifier_values_are_refused(tmp_path, cell):
    row = ["R1", 2025, "P1", "52.1301", "", "complete", "admit", cell, "80", "100"]
    p = _write(tmp_path / "r.csv", HEAD, [row])
    with pytest.raises(ValueError, match="de-identify"):
        load_admission_records(p)


def test_row_level_validation(tmp_path):
    def one(**kw):
        row = dict(zip(HEAD, ["R1", "2025", "P1", "52.1301", "", "complete", "admit", "",
                              "80", "100"], strict=True))  # fmt: skip
        row2 = dict(row, record_id="R2", outcome="reject", x_gpa_pct="70", x_toefl="90")
        row.update(kw)
        return _write(tmp_path / "r.csv", HEAD, [list(row.values()), list(row2.values())])

    for kw in ({"outcome": "waitlist"}, {"source": "gradcafe"}, {"cip": "52.13"},
               {"x_toefl": "high"}, {"year": "25"}, {"record_id": "R2"},
               {"program_type": "interview"}, {"year": ""}):  # fmt: skip
        with pytest.raises(ValueError):
            load_admission_records(one(**kw))
    assert load_admission_records(one(outcome="錄取")).records[0].admitted == 1


def test_official_rates(tmp_path):
    p = _write(tmp_path / "rates.csv",
               ["program_id", "year", "admit_rate", "applicants", "admitted", "source_url",
                "retrieved_at"],
               [["P00", 2024, "0.30", "", "", "https://example.edu/a", "2026-10-28"],
                ["P00", 2025, "", "400", "100", "https://example.edu/b", "2026-10-28"],
                ["P01", 2024, "0.5", "", "", "https://example.edu/c", "2026-10-28"]])  # fmt: skip
    assert load_official_rates(p) == {"P00": 0.25, "P01": 0.5}
    assert load_official_rates(p, 2024) == {"P00": 0.3, "P01": 0.5}
    bad = _write(tmp_path / "bad.csv", ["program_id", "year", "admit_rate", "source_url",
                                         "retrieved_at"],
                 [["P00", 2025, "0.3", "", "2026-10-28"]])  # fmt: skip
    with pytest.raises(ValueError, match="source_url"):
        load_official_rates(bad)


def test_fit_from_files_and_validation_script(tmp_path):
    rec = _synthetic_records(tmp_path)
    model, data = fit_from_files(rec)
    assert model.reference_year == 2025 and model.ptypes == ("interview", "standard")
    est = model.predict("P00", data.records[0].features)
    assert 0 < est.low <= est.mean <= est.high < 1
    out = tmp_path / "check.md"
    script = str(ROOT / "scripts/validate_admission_model.py")
    env = {"PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, script, "--records", str(rec), "--out", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    text = out.read_text(encoding="utf-8")
    assert "Brier skill" in text and "校準截距" in text and "| taiwan |" in text


# --- independent review of the intake (2026-10-28): B-C3-13..15 --------------------


def _probe(reason):
    def mark(fn):
        return pytest.mark.probe(pytest.mark.xfail(strict=True, raises=AssertionError,
                                                   reason=reason)(fn))  # fmt: skip

    return mark


def _refused(fn, *args, match=None):
    try:
        fn(*args)
    except ValueError as exc:
        return match is None or match in str(exc)
    except Exception:  # noqa: BLE001 - any other crash is a failure of the probe
        return False
    return False


ROW = ["R1", "2025", "P1", "52.1301", "", "complete", "admit", "", "80", "100"]
ROW2 = ["R2", "2025", "P1", "52.1301", "", "complete", "reject", "", "70", "90"]


@_probe("B-C3-13: identifier headers and cells in common formats get through")
def test_probe_common_identifier_formats_are_refused(tmp_path):
    slipped = []
    for h in ["Student Name", "Email Address", "Phone Number", "mobile", "DOB", "LINE ID",
              "Passport No", "學生姓名", "身分證字號", "手機號碼", "聯絡電話", "住址"]:  # fmt: skip
        p = _write(tmp_path / "h.csv", [*HEAD, h], [[*ROW, "x"], [*ROW2, "y"]])
        if not _refused(load_admission_records, p):
            slipped.append(h)
    for cell in ["0912 345 678", "+886 912 345 678", "a123456789", "A823456789",
                 "聯絡 0912345678"]:  # fmt: skip
        r = list(ROW)
        r[7] = cell
        p = _write(tmp_path / "c.csv", HEAD, [r, ROW2])
        if not _refused(load_admission_records, p):
            slipped.append(cell)
    assert not slipped, slipped


@_probe("B-C3-14: the time split fits a differently anchored, test-year-standardised model")
def test_probe_time_split_is_disclosed_and_leak_free(tmp_path):
    rec = _synthetic_records(tmp_path)
    rates = _write(tmp_path / "rates.csv",
                   ["program_id", "year", "admit_rate", "source_url", "retrieved_at"],
                   [[f"P{i:02d}", 2025, "0.4", "https://example.edu", "2026-10-28"]
                    for i in range(12)])  # fmt: skip
    out = tmp_path / "check.md"
    cmd = [sys.executable, str(ROOT / "scripts/validate_admission_model.py"), "--records",
           str(rec), "--rates", str(rates), "--out", str(out)]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src")})
    assert run.returncode == 0, run.stderr
    text = out.read_text(encoding="utf-8")
    assert "全部資料 12 個學程" in text and "訓練 0 個學程" in text
    try:
        train = load_admission_records(rec, years={2024})
        test = load_admission_records(rec, years={2025}, standardize_with=train.report)
    except TypeError:
        raise AssertionError(
            "no way to standardise the test year with training statistics"
        ) from None
    assert test.report.means == train.report.means
    assert test.report.features == train.report.features


@_probe("B-C3-15: ragged rows, blank lines, case and encoding are mishandled")
def test_probe_loader_robustness(tmp_path):
    p = tmp_path / "ragged.csv"
    p.write_text(",".join(HEAD) + "\n" + ",".join(ROW) + ",extra\n" + ",".join(ROW2) + "\n",
                 encoding="utf-8")  # fmt: skip
    assert _refused(load_admission_records, p, match=":2")
    p = tmp_path / "short.csv"
    p.write_text(",".join(HEAD) + "\n" + ",".join(ROW) + "\nR3,2025,P1\n", encoding="utf-8")
    assert _refused(load_admission_records, p, match=":3")
    p = tmp_path / "blank.csv"
    bad = list(ROW2)
    bad[6] = "waitlist"
    p.write_text(",".join(HEAD) + "\n" + ",".join(ROW) + "\n\n\n" + ",".join(bad) + "\n",
                 encoding="utf-8")  # fmt: skip
    assert _refused(load_admission_records, p, match=":5")
    p = tmp_path / "big5.csv"
    r = list(ROW)
    r[6] = "錄取"
    p.write_bytes((",".join(HEAD) + "\n" + ",".join(r) + "\n" + ",".join(ROW2) + "\n")
                  .encode("cp950"))  # fmt: skip
    assert _refused(load_admission_records, p, match="UTF-8")
    head = [" X_gpa_pct " if h == "x_gpa_pct" else h for h in HEAD]
    r1, r2 = list(ROW), list(ROW2)
    r1[4], r2[4] = "Interview", "interview"
    data = load_admission_records(_write(tmp_path / "case.csv", head, [r1, r2]))
    assert "x_gpa_pct" in data.report.features and data.types == {"P1": "interview"}
    # a feature only self-reports carry, with >= 30 complete records
    rows = [[f"C{k}", "2025", "P1", "52.1301", "", "complete", "admit" if k % 2 else "reject",
             "", str(70 + k % 9), ""] for k in range(30)]  # fmt: skip
    rows += [[f"S{k}", "2025", "P1", "52.1301", "", "self_report", "admit", "", "80",
              str(95 + k)] for k in range(5)]  # fmt: skip
    data = load_admission_records(_write(tmp_path / "selfonly.csv", HEAD, rows))
    assert "x_toefl" in data.report.features and data.report.warnings


@_probe("B-C3-15: conflicting, percent or undated official rates are accepted or crash")
def test_probe_rates_validation(tmp_path):
    head = ["program_id", "year", "admit_rate", "source_url", "retrieved_at"]
    ok = ["P1", "2025", "0.3", "https://example.edu", "2026-10-28"]
    for bad in (["P1", "2025", "0.4", "https://example.edu", "2026-10-28"],
                ["", "2025", "0.4", "https://example.edu", "2026-10-28"],
                ["P2", "2025", "45%", "https://example.edu", "2026-10-28"],
                ["P2", "2025", "0.4", "https://example.edu", "?"]):  # fmt: skip
        p = _write(tmp_path / "rates.csv", head, [ok, bad])
        assert _refused(load_official_rates, p, match=":3"), bad


@_probe("B-C3-15: the validation script crashes on a new type or an all-admit test year")
def test_probe_validation_script_survives_test_year_surprises(tmp_path):
    rec = _synthetic_records(tmp_path)
    lines = rec.read_text(encoding="utf-8").splitlines()
    extra = [f"N{k},2025,P99,52.1301,portfolio,complete,{'admit' if k % 2 else 'reject'},"
             f"taiwan,80,100" for k in range(5)]  # fmt: skip
    rec.write_text("\n".join(lines + extra) + "\n", encoding="utf-8")
    out = tmp_path / "check.md"
    cmd = [sys.executable, str(ROOT / "scripts/validate_admission_model.py"), "--records",
           str(rec), "--out", str(out)]  # fmt: skip
    env = {"PYTHONPATH": str(ROOT / "src")}
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-500:]
    assert "略過" in out.read_text(encoding="utf-8")
    lines = [ln.replace(",reject,", ",admit,") if ",2025," in ln else ln
             for ln in lines]  # fmt: skip
    rec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-500:]
