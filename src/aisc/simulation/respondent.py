"""Simulated respondent for the C1.4 familiarity check (A-C5-8).

Signal-detection model: a truly familiar program is reported "heard" with
probability 1 - miss_rate; an unfamiliar one is reported "heard" with
probability false_alarm_rate (name confusion, B-C1-12). Errors are
independent across programs — real false alarms probably cluster on
similarly named programs, which this model does not capture.
"""

from __future__ import annotations

import numpy as np

from aisc.simulation.population import SyntheticStudent


def make_ask(
    student: SyntheticStudent,
    rng: np.random.Generator,
    miss_rate: float = 0.0,
    false_alarm_rate: float = 0.0,
):
    if not 0 <= miss_rate <= 1 or not 0 <= false_alarm_rate <= 1:
        raise ValueError("Rates must lie in [0, 1]")

    def ask(program_ids: list[str]) -> dict[str, str]:
        report: dict[str, str] = {}
        for pid in program_ids:
            truly = pid in student.familiar
            says = rng.random() >= miss_rate if truly else rng.random() < false_alarm_rate
            report[pid] = "heard" if says else "unknown"
        return report

    return ask
