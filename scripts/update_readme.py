"""Regenerate the README sections that track development (W-06).

The README copies nothing by hand. Each block between
    <!-- AUTO:<name> START ... -->  and  <!-- AUTO:<name> END -->
is rebuilt from its single source of truth (CONTRIBUTING.md):

    version         latest row of docs/USAGE.md §8 changelog
                    (also synced into pyproject.toml, src/aisc/__init__.py and the
                    "最後更新" line at the top of docs/USAGE.md)
    status          docs/USAGE.md §1 feature table
    latest-results  docs/VALIDATION.md §5 "最新 L3 結果" links
    next-steps      docs/ROADMAP.md "下一步" table
    changelog       last rows of docs/USAGE.md §8

Relative links are rewritten so they resolve from the repo root.
Standard library only, so CI can run it without installing the package.

    python scripts/update_readme.py           # rewrite README (and version files)
    python scripts/update_readme.py --check   # exit 1 if anything is out of date
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
USAGE = ROOT / "docs" / "USAGE.md"
VALIDATION = ROOT / "docs" / "VALIDATION.md"
ROADMAP = ROOT / "docs" / "ROADMAP.md"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "aisc" / "__init__.py"

CHANGELOG_ROWS = 3
NOTE = "由 scripts/update_readme.py 自動產生，請修改來源檔，不要直接改這裡"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rebase_links(text: str, source: Path) -> str:
    """Rewrite relative markdown links in `text` (written in `source`) to repo-root form."""
    src_dir = source.parent.relative_to(ROOT).as_posix()

    def fix(m: re.Match[str]) -> str:
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        path, _, frag = target.partition("#")
        new = posixpath.normpath(posixpath.join(src_dir, path)) if path else ""
        return f"]({new}{'#' + frag if frag else ''})"

    return re.sub(r"\]\(([^)\s]+)\)", fix, text)


def _section(text: str, heading_prefix: str) -> str:
    """Body of the first '## ' section whose heading starts with `heading_prefix`."""
    m = re.search(rf"^## {re.escape(heading_prefix)}.*?$\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"section '## {heading_prefix}' not found")
    return m.group(1)


def _first_table(body: str) -> str:
    lines, started = [], False
    for line in body.splitlines():
        if line.startswith("|"):
            lines.append(line)
            started = True
        elif started:
            break
    if not lines:
        raise SystemExit("expected a markdown table")
    return "\n".join(lines)


def changelog_rows() -> list[str]:
    table = _first_table(_section(_read(USAGE), "8."))
    rows = [r for r in table.splitlines()[2:] if r.strip()]
    if not rows:
        raise SystemExit("docs/USAGE.md §8 changelog is empty")
    return rows


def usage_header(text: str) -> str:
    """Sync the '最後更新：<date> · 對應版本 v<x>' line with the latest changelog row."""
    date = changelog_rows()[-1].strip("|").split("|")[0].strip()
    line = f"最後更新：{date} · 對應版本 v{current_version()}（研究原型）"
    new, n = re.subn(r"^最後更新：.*$", line, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("docs/USAGE.md: '最後更新：' line not found")
    return new


def current_version() -> str:
    cell = changelog_rows()[-1].strip("|").split("|")[1].strip()
    m = re.fullmatch(r"v?(\d+(?:\.\d+){1,2})", cell)
    if not m:
        raise SystemExit(f"latest changelog version {cell!r} is not like v0.3.2")
    return m.group(1)


def build_blocks() -> dict[str, str]:
    version = current_version()
    usage = _read(USAGE)

    status_body = _section(usage, "1.")
    status = rebase_links(_first_table(status_body), USAGE)
    summary = re.search(r"^\*\*一句話：\*\*.*$", status_body, re.M)
    status_block = (
        "來源：[docs/USAGE.md §1](docs/USAGE.md#1-現在能用什麼)"
        "（各部件的驗證關卡見 [docs/VALIDATION.md §5](docs/VALIDATION.md)）\n\n" + status
    )
    if summary:
        status_block += "\n\n" + summary.group(0)

    validation = _read(VALIDATION)
    m = re.search(r"最新 L3 結果：(.*)$", validation, re.M)
    if not m:
        raise SystemExit("docs/VALIDATION.md: line with '最新 L3 結果：' not found")
    latest = rebase_links(m.group(1).strip(), VALIDATION)

    roadmap = _read(ROADMAP)
    h = re.search(r"^## 下一步(.*)$", roadmap, re.M)
    if not h:
        raise SystemExit("docs/ROADMAP.md: '## 下一步' not found")
    next_table = rebase_links(_first_table(_section(roadmap, "下一步")), ROADMAP)
    when = h.group(1).strip()
    next_block = f"來源：[docs/ROADMAP.md](docs/ROADMAP.md) {when}\n\n{next_table}"

    rows = changelog_rows()[-CHANGELOG_ROWS:]
    header = "| 日期 | 版本 | 改變 | 工項 |\n|---|---|---|---|"
    changelog = (
        "來源：[docs/USAGE.md §8](docs/USAGE.md#8-變更紀錄使用方式)（最近 "
        f"{len(rows)} 筆，新到舊）\n\n"
        + header
        + "\n"
        + rebase_links("\n".join(reversed(rows)), USAGE)
    )

    return {
        "version": (
            f"**Status:** v{version} · research prototype · program attributes used in "
            "simulations are **synthetic**; reference tables in `data/reference/` "
            "(CIP 2020, IPEDS 2025, O*NET) are real and sourced."
        ),
        "status": status_block,
        "latest-results": (
            "最新 L3 合成驗證結論（來源：[docs/VALIDATION.md §5](docs/VALIDATION.md)）：" + latest
        ),
        "next-steps": next_block,
        "changelog": changelog,
    }


def render(readme: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        tag = re.escape(name)
        pattern = re.compile(
            rf"(<!-- AUTO:{tag} START.*?-->\n).*?(\n<!-- AUTO:{tag} END -->)", re.S
        )
        if not pattern.search(readme):
            raise SystemExit(f"README.md is missing the AUTO:{name} markers")
        start = f"<!-- AUTO:{name} START · {NOTE} -->\n"
        readme = pattern.sub(lambda m, s=start, b=body: s + b + m.group(2), readme, count=1)
    return readme


def sync_version(text: str, pattern: str, version: str) -> str:
    new, n = re.subn(pattern, lambda m: f'{m.group(1)}"{version}"', text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"version line not found ({pattern})")
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report staleness, write nothing")
    args = ap.parse_args(argv)

    version = current_version()
    targets = {
        README: render(_read(README), build_blocks()),
        PYPROJECT: sync_version(_read(PYPROJECT), r"^(version = )\"[^\"]*\"", version),
        INIT: sync_version(_read(INIT), r"^(__version__ = )\"[^\"]*\"", version),
        USAGE: usage_header(_read(USAGE)),
    }
    stale = [p for p, new in targets.items() if _read(p) != new]
    if args.check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)}")
        if stale:
            print("run: python scripts/update_readme.py")
        return 1 if stale else 0
    for p in stale:
        p.write_text(targets[p], encoding="utf-8")
        print(f"updated: {p.relative_to(ROOT)}")
    if not stale:
        print("README already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
