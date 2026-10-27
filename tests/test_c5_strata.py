"""C5 stratified evaluation (W-68, D12): verification (L2)."""

import math

import numpy as np
import pytest

from aisc.simulation.catalog_gen import synthetic_catalog
from aisc.simulation.evaluate import (
    c2_pipeline,
    evaluate_lists,
    evaluate_lists_per_student,
)
from aisc.simulation.population import SimConfig, generate_population
from aisc.simulation.strata import (
    ALL,
    StratumStat,
    format_cell,
    maximin,
    paired_stratified,
    stratified,
    worst_case,
    worst_stratum,
)


def test_stratified_means_and_support():
    vals = [1.0, 2.0, 3.0, 4.0, float("nan")]
    labs = ["a", "a", "b", "b", "b"]
    st = stratified(vals, labs, min_n=2)
    assert st[ALL].mean == pytest.approx(2.5) and st[ALL].n == 4
    assert st["a"].mean == pytest.approx(1.5) and st["a"].supported
    assert st["b"].mean == pytest.approx(3.5) and st["b"].n == 2
    st3 = stratified(vals, labs, min_n=3)
    assert not st3["a"].supported and not st3["b"].supported
    with pytest.raises(ValueError):
        stratified([1.0], ["a", "b"])
    with pytest.raises(ValueError):
        stratified([1.0], ["a"], min_n=1)


def test_paired_and_worst():
    a = [0.5] * 40 + [0.2] * 40
    b = [0.4] * 40 + [0.3] * 40
    labs = ["x"] * 40 + ["y"] * 40
    d = paired_stratified(a, b, labs)
    assert d["x"].mean == pytest.approx(0.1) and d["y"].mean == pytest.approx(-0.1)
    assert worst_stratum(d) == "y" and worst_stratum(d, higher_is_better=False) == "x"
    noisy = paired_stratified(list(np.r_[np.full(40, -0.1), np.linspace(-1, 1, 40)]),
                              [0.0] * 80, labs)  # fmt: skip
    assert noisy["x"].significantly_below()  # deterministic -0.1 everywhere
    assert StratumStat(-0.5, 0.1, 50, True).significantly_below()
    assert not StratumStat(-0.5, 0.1, 5, False).significantly_below()
    assert (
        worst_stratum({ALL: StratumStat(0, 0, 10, True), "z": StratumStat(0, 0, 3, False)}) is None
    )


def test_maximin_uses_worst_cell():
    table = {"m1": {"c1": 0.9, "c2": 0.1}, "m2": {"c1": 0.5, "c2": 0.4}}
    assert worst_case(table) == {"m1": 0.1, "m2": 0.4}
    assert maximin(table) == "m2"
    assert maximin(table, higher_is_better=False) == "m2"  # worst = largest: 0.9 vs 0.5
    with pytest.raises(ValueError):
        maximin({})


def test_format_cell():
    assert format_cell(StratumStat(0.1234, 0.0101, 40, True)) == "0.123±0.010 (n=40)"
    assert format_cell(StratumStat(0.1, 0.01, 4, False), signed=True) == "[+0.100±0.010 (n=4)]"
    assert math.isnan(stratified([float("nan")] * 3, ["a"] * 3, min_n=2)["a"].mean)


def test_background_stratum_and_per_student_rows_match_summary():
    cat = synthetic_catalog(np.random.default_rng(0), n_per_family=3)
    pop = generate_population(cat, SimConfig(n_students=60), np.random.default_rng(1))
    labels = {st.background_stratum for st in pop}
    assert labels <= {"continuing", "switching"} and len(labels) == 2
    for st in pop:
        same = st.background_family == st.natural_family
        assert (st.background_stratum == "continuing") == same
    recs = {"c2": c2_pipeline()}
    per = evaluate_lists_per_student(pop, cat, recs, k=5)["c2"]
    summ = evaluate_lists(pop, cat, recs, k=5)["c2"]
    for key in ("relevant_rate", "novel_relevant_rate", "best_fit_retained"):
        assert len(per[key]) == len(pop)
        assert np.nanmean(per[key]) == pytest.approx(summ[key][0])
