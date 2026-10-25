"""Load and integrity-check program catalogues (JSON Lines)."""

from __future__ import annotations

import json
from pathlib import Path

from aisc.contracts import Program


def load_programs(path: str | Path) -> list[Program]:
    """Load programs from a JSON Lines file (one program per line).

    Integrity rules (A4 in docs/VALIDATION.md):
    - every record parses into a Program and its vector lies in [0, 1];
    - program_id is unique;
    - non-synthetic records carry source_url and retrieved_at.
    """
    programs: list[Program] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                program = Program(**json.loads(line))
                _ = program.vector  # validates dimensions and range
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if not program.synthetic and not (program.source_url and program.retrieved_at):
                raise ValueError(
                    f"{path}:{line_no}: real program {program.program_id} needs "
                    "source_url and retrieved_at"
                )
            programs.append(program)
    ids = [p.program_id for p in programs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate program_id values")
    return programs
