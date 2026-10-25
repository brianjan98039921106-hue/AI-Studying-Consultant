"""Dimension registry v1 (W-16b) — must match docs/rubric.md §2."""

import pytest

from aisc.dimensions import LAYERS, N_NEEDS, NEED_DIMENSIONS, NEED_KEYS, RIASEC_KEYS, keys_in_layer


def test_fifteen_unique_dimensions_in_four_layers():
    assert N_NEEDS == 15 and len(set(NEED_KEYS)) == 15
    assert {d.layer for d in NEED_DIMENSIONS} == set(LAYERS)


def test_layer_sizes_match_rubric():
    assert [len(keys_in_layer(layer)) for layer in LAYERS] == [6, 2, 5, 2]


def test_riasec_order_is_holland_hexagon_order():
    assert tuple(f"riasec_{c}" for c in "riasec") == RIASEC_KEYS


def test_v0_business_dimension_is_gone():
    # Q4: "business" folded into Enterprising interest.
    assert "business" not in NEED_KEYS and "riasec_e" in NEED_KEYS


def test_every_dimension_documents_both_measurement_sources():
    for d in NEED_DIMENSIONS:
        assert d.student_source and d.program_source and d.label_zh


def test_unknown_layer_rejected():
    with pytest.raises(ValueError):
        keys_in_layer("vibes")
