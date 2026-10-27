"""C5 stratified evaluation (W-68, D12): report every metric per background stratum
and judge methods by their worst stratum, not only by the average.

A stratum smaller than `min_n` students is reported but marked unsupported and
never used to pick a winner (too noisy to trust either way).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from aisc.simulation.metrics import mean_se

ALL = "all"
MIN_N = 30


@dataclass(frozen=True)
class StratumStat:
    mean: float
    se: float
    n: int
    supported: bool

    def significantly_below(self, other: float = 0.0) -> bool:
        """True when mean is more than 2 SE below `other` (used for paired differences)."""
        return self.supported and not math.isnan(self.se) and self.mean < other - 2 * self.se


def stratified(
    values: Sequence[float], labels: Sequence[str], min_n: int = MIN_N
) -> dict[str, StratumStat]:
    """Mean ± SE per stratum plus 'all'. NaN values are ignored (as in mean_se)."""
    if len(values) != len(labels):
        raise ValueError("values and labels must have the same length")
    if min_n < 2:
        raise ValueError("min_n must be at least 2")
    v = np.asarray(values, float)
    lab = np.asarray(labels, dtype=object)
    out: dict[str, StratumStat] = {}
    for name in [ALL, *sorted(set(labels))]:
        sel = v if name == ALL else v[lab == name]
        n = int(np.sum(~np.isnan(sel)))
        m, se = mean_se(list(sel))
        out[name] = StratumStat(m, se, n, n >= min_n)
    return out


def paired_stratified(
    a: Sequence[float], b: Sequence[float], labels: Sequence[str], min_n: int = MIN_N
) -> dict[str, StratumStat]:
    """Stratified paired difference a - b (same students, same order)."""
    if len(a) != len(b):
        raise ValueError("paired sequences must have the same length")
    return stratified(list(np.asarray(a, float) - np.asarray(b, float)), labels, min_n)


def worst_stratum(stats: Mapping[str, StratumStat], higher_is_better: bool = True) -> str | None:
    """Supported stratum with the worst mean ('all' excluded); None if none is supported."""
    cands = [(k, s.mean) for k, s in stats.items() if k != ALL and s.supported]
    if not cands:
        return None
    return (min if higher_is_better else max)(cands, key=lambda kv: kv[1])[0]


def worst_case(
    table: Mapping[str, Mapping[str, float]], higher_is_better: bool = True
) -> dict[str, float]:
    """For each method, its worst value over conditions/strata (maximin input, D12)."""
    pick = min if higher_is_better else max
    return {name: pick(row.values()) for name, row in table.items() if row}


def maximin(table: Mapping[str, Mapping[str, float]], higher_is_better: bool = True) -> str:
    """Method whose worst condition/stratum is best."""
    worst = worst_case(table, higher_is_better)
    if not worst:
        raise ValueError("empty table")
    return (max if higher_is_better else min)(sorted(worst), key=lambda k: worst[k])


def format_cell(s: StratumStat, digits: int = 3, signed: bool = False) -> str:
    """'0.123±0.004 (n=150)'; unsupported strata are bracketed."""
    sign = "+" if signed else ""
    cell = f"{s.mean:{sign}.{digits}f}±{s.se:.{digits}f} (n={s.n})"
    return cell if s.supported else f"[{cell}]"


def labels_for(population: Sequence, stratum_fn: Callable[[object], str]) -> list[str]:
    return [stratum_fn(st) for st in population]
