# Notes for AI assistants working in this repo

## What this is
A recommender that profiles Taiwanese grad-school applicants by *needs* (not program names) and surfaces US programs they would not have found, then optimises the application portfolio. Read `docs/ARCHITECTURE.md` first.

## Commands
- Test: `pytest`
- Lint / format: `ruff check . && ruff format .`
- Demo: `python examples/quickstart.py`

## Rules
- Never commit real student data. `data/private/` is gitignored for that.
- Never present synthetic data as real. Synthetic records carry `"synthetic": true` and fictional school names.
- Real program facts need `source_url` and `retrieved_at`. Do not fill attributes from memory.
- Optimisation changes must keep the brute-force tests in `tests/test_portfolio.py` passing.
- Need dimensions live only in `src/aisc/needs.py`; adding one means updating the rubric, synthetic data and tests together.
- Docs are written in Traditional Chinese with English technical terms; code and docstrings in English.
- Use Conventional Commits.
- When a TODO in `docs/ROADMAP.md` is finished, tick it in the same commit.
