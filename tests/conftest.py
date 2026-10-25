from pathlib import Path

import pytest

from aisc.catalog import load_programs

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "programs.jsonl"


@pytest.fixture(scope="session")
def catalog():
    return load_programs(SYNTHETIC)


@pytest.fixture(scope="session")
def by_id(catalog):
    return {p.program_id: p for p in catalog}
