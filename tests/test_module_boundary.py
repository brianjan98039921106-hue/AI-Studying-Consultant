"""Scope boundary (ARCHITECTURE D13): the core (needs profiling, recommendation and
requirement gaps: C0, C1, C2, C6, the pipeline and the report) must not load the
paused components C3 (aisc.feasibility) and C4 (aisc.portfolio).

Checked in a fresh interpreter so that modules imported by other tests do not hide a
dependency. The static check in test_interface_freeze.py covers the component
packages; this one also covers the pipeline, the report and the demo, where
`plan_applications` (the frozen interface) imports C4 lazily.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = [
    "aisc",
    "aisc.contracts",
    "aisc.dimensions",
    "aisc.catalog",
    "aisc.profiling",
    "aisc.discovery",
    "aisc.requirements",
    "aisc.pipeline",
    "aisc.report",
]
PAUSED = ("aisc.feasibility", "aisc.portfolio")


def _loaded_paused(setup: str) -> list[str]:
    code = (
        f"{setup}\n"
        "import sys\n"
        f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith({PAUSED!r}))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.strip()
    return out.splitlines()


def test_core_does_not_load_paused_components():
    setup = f"import importlib\nfor m in {CORE!r}: importlib.import_module(m)"
    assert _loaded_paused(setup) == []


def test_demo_does_not_load_paused_components():
    """The default flow (examples/quickstart.py) runs without C3 or C4 (D13)."""
    setup = (
        "import contextlib, io, runpy\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    runpy.run_path('examples/quickstart.py', run_name='__main__')"
    )
    assert _loaded_paused(setup) == []
