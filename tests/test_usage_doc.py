"""The user guide must stay runnable (W-05).

Every ```python block in docs/USAGE.md is executed in order, in one namespace,
from the repo root. If an API changes and the guide is not updated, CI fails.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs" / "USAGE.md"


def _python_blocks() -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", USAGE.read_text(encoding="utf-8"), re.M | re.S)


def test_usage_doc_has_examples():
    assert len(_python_blocks()) >= 5


def test_usage_doc_examples_run(capsys):
    namespace: dict[str, object] = {"__name__": "usage_doc"}
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for i, block in enumerate(_python_blocks(), start=1):
            try:
                exec(compile(block, f"docs/USAGE.md block {i}", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - message is the point
                raise AssertionError(f"docs/USAGE.md python block {i} failed: {exc!r}") from exc
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "可能沒想過" in out
