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


def make_item_responder(
    student: SyntheticStudent,
    rng: np.random.Generator,
    noise_sd: float = 0.25,
    acquiescence: float = 0.0,
    desirability: dict[str, float] | None = None,
):
    """Answers C1.3 items: y = clip(w_d + acquiescence + desirability_d + N(0, noise_sd)).

    Setting `noise_sd` different from the estimator's assumed item noise, or
    adding biases, simulates a misspecified response model (V-C1-4, B-C1-5).
    """
    from aisc.dimensions import NEED_KEYS

    if noise_sd < 0:
        raise ValueError("noise_sd must be non-negative")
    shift = desirability or {}

    def answer(item) -> float:
        d = NEED_KEYS.index(item.dimension)
        y = student.true_needs[d] + acquiescence + shift.get(item.dimension, 0.0)
        return float(np.clip(y + rng.normal(0.0, noise_sd), 0.0, 1.0))

    return answer
