"""W-52: the student self-report questionnaire on the v2 item bank (D14; THEORY
"學生自填問卷"; A-C1-60, A-C1-61).

The bank (data/items/item_bank_v2.json, 73 items) is asked in full, in three pages:
    core   12 RIASEC items, 4 intensity items, 4 orientation items
    group  12 interest-area group items (stage 1 of the two-stage design)
    area   41 interest-area items, ordered by the student's group answers
           (`areas_to_ask` with every group); the order changes nothing in scoring
Every item can be skipped. Scoring: answer 1..5 -> (answer - 1) / 4, the mean of a
dimension's answered items; group items become `group_scores`. Skipped dimensions
stay out of `needs` so the usual completion (C1.2, pipeline.prepare_profile) fills
them. Asking everything follows F180-F183: no shorter two-stage variant was
non-inferior to answering all items.

UI-agnostic: a web page, a terminal or a test drives `Questionnaire` with
`next_page()` / `answer()` / `profile()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aisc.contracts import StudentProfile
from aisc.dimensions import BASIC_INTEREST_KEYS, NEED_KEYS_V2
from aisc.profiling.hierarchy import GROUP_LABELS, areas_to_ask

DEFAULT_BANK = Path("data/items/item_bank_v2.json")
STAGES: tuple[str, ...] = ("core", "group", "area")
ANSWERS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Question:
    item_id: str
    stage: str  # "core", "group" or "area"
    dimension: str | None  # None for group items
    group: int | None  # group id for group items
    stem_zh: str
    examples_zh: tuple[str, ...]
    labels_zh: tuple[str, ...]  # the five answer options, low to high


def load_questions(path: str | Path = DEFAULT_BANK) -> list[Question]:
    """Read the v2 item bank and check that it covers every v2 dimension and group."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for it in spec["items"]:
        labels = tuple(it["scale"]["labels_zh"])
        if len(labels) != len(ANSWERS):
            raise ValueError(f"{it['item_id']}: expected 5 answer labels")
        out.append(Question(it["item_id"], it["stage"], it.get("dimension"), it.get("group"),
                            it["stem_zh"], tuple(it.get("examples_zh") or ()), labels))  # fmt: skip
    if len({q.item_id for q in out}) != len(out):
        raise ValueError("item ids must be unique")
    if {q.stage for q in out} - set(STAGES):
        raise ValueError(f"unknown stages {sorted({q.stage for q in out} - set(STAGES))}")
    dims = {q.dimension for q in out if q.dimension}
    if dims != set(NEED_KEYS_V2):
        raise ValueError(
            f"bank does not cover the v2 dimensions: {sorted(set(NEED_KEYS_V2) ^ dims)}"
        )
    if {q.group for q in out if q.stage == "group"} != set(GROUP_LABELS):
        raise ValueError("bank must have one item per interest-area group")
    return out


@dataclass
class Questionnaire:
    """One student's run through the bank. `answers` maps item_id -> 1..5 or None
    (skipped)."""

    questions: list[Question] = field(default_factory=load_questions)
    answers: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {q.item_id: q for q in self.questions}
        self._area_item = {q.dimension: q.item_id for q in self.questions if q.stage == "area"}

    # flow ------------------------------------------------------------------------------
    def stage(self) -> str | None:
        """First stage with unanswered items, or None when every item is answered or
        skipped."""
        for s in STAGES:
            if any(q.item_id not in self.answers for q in self.page(s)):
                return s
        return None

    def page(self, stage: str) -> list[Question]:
        """All items of a stage in asking order (area items follow the group answers)."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        items = [q for q in self.questions if q.stage == stage]
        if stage != "area":
            return items
        order = areas_to_ask(self.group_scores(), m=len(GROUP_LABELS))
        rest = [k for k in BASIC_INTEREST_KEYS if k not in order]
        return [self._by_id[self._area_item[k]] for k in order + rest if k in self._area_item]

    def next_page(self) -> list[Question]:
        """Unanswered items of the current stage ([] when finished)."""
        s = self.stage()
        return [] if s is None else [q for q in self.page(s) if q.item_id not in self.answers]

    def answer(self, item_id: str, value: int | None) -> None:
        if item_id not in self._by_id:
            raise ValueError(f"unknown item {item_id!r}")
        if value is not None and value not in ANSWERS:
            raise ValueError(f"{item_id}: answer must be 1-5 or None (skip), got {value!r}")
        self.answers[item_id] = value

    def progress(self) -> tuple[int, int]:
        """(items answered or skipped, total items)."""
        return len(self.answers), len(self.questions)

    # scoring ---------------------------------------------------------------------------
    def group_scores(self) -> dict[int, float]:
        return {q.group: (self.answers[q.item_id] - 1) / 4 for q in self.questions
                if q.stage == "group" and self.answers.get(q.item_id) is not None}  # fmt: skip

    def needs(self) -> dict[str, float]:
        """Mean of answered items per dimension, on [0, 1]; skipped dimensions absent."""
        acc: dict[str, list[float]] = {}
        for q in self.questions:
            v = self.answers.get(q.item_id)
            if q.dimension and v is not None:
                acc.setdefault(q.dimension, []).append((v - 1) / 4)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def skipped(self) -> list[str]:
        return [i for i, v in self.answers.items() if v is None]

    def profile(self, student_id: str, anchors: list[str] | None = None) -> StudentProfile:
        """A v2 profile of the answers so far; complete it with pipeline.prepare_profile
        before ranking."""
        return StudentProfile(student_id, self.needs(), anchors=list(anchors or []), dims="v2",
                              group_scores=self.group_scores())  # fmt: skip
