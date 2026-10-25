"""C0 catalog integrity (validation level L2, check A4)."""

import json

import pytest

from aisc.catalog import load_programs


def test_synthetic_catalog_loads_and_is_marked(catalog):
    assert len(catalog) == 12
    assert all(p.synthetic for p in catalog)


def _write(tmp_path, records):
    path = tmp_path / "programs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


BASE = {
    "program_id": "r1",
    "school": "X",
    "degree_name": "MS Y",
    "cip": "27.0501",
    "attributes": {"quant": 0.5},
}


def test_real_program_without_provenance_rejected(tmp_path):
    with pytest.raises(ValueError, match="source_url"):
        load_programs(_write(tmp_path, [BASE]))


def test_real_program_with_provenance_accepted(tmp_path):
    rec = {**BASE, "source_url": "https://example.edu/ms-y", "retrieved_at": "2026-10-25"}
    assert load_programs(_write(tmp_path, [rec]))[0].program_id == "r1"


def test_duplicate_ids_rejected(tmp_path):
    rec = {**BASE, "synthetic": True}
    with pytest.raises(ValueError, match="duplicate"):
        load_programs(_write(tmp_path, [rec, rec]))


def test_out_of_range_attribute_rejected(tmp_path):
    rec = {**BASE, "synthetic": True, "attributes": {"quant": 2.0}}
    with pytest.raises(ValueError):
        load_programs(_write(tmp_path, [rec]))
