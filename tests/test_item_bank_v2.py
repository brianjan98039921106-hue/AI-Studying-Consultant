"""W-41 draft item bank v2 (docs/item-bank-v2.md): structure and provenance, L2."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from aisc.dimensions import BASIC_INTEREST_KEYS, NEED_KEYS_V2, keys_in_layer
from aisc.profiling.hierarchy import AREA_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "items"


def _bank():
    return json.loads((ITEMS / "item_bank_v2.json").read_text(encoding="utf-8"))


def test_every_v2_dimension_and_group_has_items():
    items = _bank()["items"]
    per_dim = {}
    for it in items:
        if it["dimension"] is not None:
            per_dim.setdefault(it["dimension"], []).append(it)
    assert set(per_dim) == set(NEED_KEYS_V2)
    for key in NEED_KEYS_V2:
        expected = 1 if key in BASIC_INTEREST_KEYS else 2  # alpha needs two items
        assert len(per_dim[key]) == expected, key
    groups = [it for it in items if it["stage"] == "group"]
    assert sorted(it["group"] for it in groups) == sorted(GROUP_LABELS)
    for it in groups:  # every primary area of the group is represented
        n_primary = sum(1 for k in BASIC_INTEREST_KEYS if AREA_GROUPS[k][0] == it["group"])
        assert len(it["examples_zh"]) == max(n_primary, 3)
    assert len(items) == _bank()["n_items"] == 73


def test_items_are_traceable_to_their_sources():
    with open(ITEMS / "onet_illustrative_activities_zh.csv", encoding="utf-8") as fh:
        acts = {(r["element_id"], r["activity_en"]): r for r in csv.DictReader(fh)}
    ids = {eid for eid, _ in acts}
    for it in _bank()["items"]:
        assert len(it["scale"]["labels_zh"]) == 5
        if it["dimension"] in keys_in_layer("intensity", "v2") + keys_in_layer("orientation", "v2"):
            continue
        assert set(it["source"]) <= ids
        assert len(it["examples_zh"]) == len(it["examples_en"])
        for en, eid in zip(it["examples_en"], it["source"], strict=True):
            assert (eid, en) in acts
    src = _bank()["sources"]["onet_illustrative_activities"]
    assert src["url"].startswith("https://www.onetcenter.org/") and src["release"] == "O*NET 31.0"


def test_committed_bank_matches_its_sources(tmp_path):
    out = tmp_path / "bank.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_item_bank_v2.py"),
                          "--out", str(out)], capture_output=True, text=True, env=env)  # fmt: skip
    assert res.returncode == 0, res.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == _bank()
