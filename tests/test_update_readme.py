"""scripts/update_readme.py (W-06): the README sections that track development
are generated from their sources. Freshness is not tested here (that would fail
every PR that edits a source); the README workflow regenerates on main."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("update_readme", ROOT / "scripts/update_readme.py")
ur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ur)


def test_rebase_links_from_docs_to_root():
    src = ROOT / "docs" / "VALIDATION.md"
    text = "[a](../results/x.md) [b](USAGE.md#1-x) [c](https://e.com) [d](#local)"
    assert ur.rebase_links(text, src) == (
        "[a](results/x.md) [b](docs/USAGE.md#1-x) [c](https://e.com) [d](#local)"
    )


def test_every_block_builds_and_has_content():
    blocks = ur.build_blocks()
    assert set(blocks) == {"version", "status", "latest-results", "next-steps", "changelog"}
    assert all(b.strip() for b in blocks.values())
    assert f"v{ur.current_version()}" in blocks["version"]


def test_render_is_idempotent_and_keeps_static_text():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    once = ur.render(readme, ur.build_blocks())
    assert ur.render(once, ur.build_blocks()) == once
    assert "## 要解決的問題" in once and "## License" in once


def test_render_rejects_missing_markers():
    with pytest.raises(SystemExit, match="AUTO:status"):
        ur.render("# no markers\n", {"status": "x"})


def test_version_sync_rewrites_only_the_version_line():
    text = 'name = "aisc"\nversion = "0.1.0"\nother = "0.1.0"\n'
    out = ur.sync_version(text, r"^(version = )\"[^\"]*\"", "0.3.2")
    assert out == 'name = "aisc"\nversion = "0.3.2"\nother = "0.1.0"\n'
