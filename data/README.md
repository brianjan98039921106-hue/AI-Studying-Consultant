# Data

| Folder | Content | Committed? |
|---|---|---|
| `synthetic/` | Fictional programs for tests and demos (`"synthetic": true`) | Yes |
| `reference/` | Public reference tables (NCES CIP 2020, IPEDS Completions extracts) | Yes, with source + date below |
| `programs/` | Real program attributes; each record has `source_url`, `retrieved_at` | Yes |
| `private/` | Student profiles, survey answers, consultant cases | **Never** (gitignored) |

## Synthetic catalogue
`synthetic/programs.jsonl` — 12 fictional programs. School names are placeholders. CIP codes are used only to exercise label-space novelty and must be checked against the official CIP 2020 list before any real use. Attribute scores and `prestige` values are hand-set for testing, not measurements. The simulator (C5) redraws prestige for its own catalogue; the prototype prestige values are used only by unit tests such as the B-C1-2 probe.

## Reference tables (W-14, W-15)
Built by `scripts/build_reference.py` from the official files as downloaded; never edited by hand. `reference/SOURCES.json` records, for every input, the source URL, the download date and the SHA-256 of the file.

| File | Content | Source |
|---|---|---|
| `reference/mvp_scope.json` | CIP 2020 prefixes of the MVP (hand-written decision, versioned; B-C0-6) | — |
| `reference/cip2020_mvp.csv` | 6-digit CIP codes in scope, official titles and definitions | NCES CIP 2020 |
| `reference/ipeds_masters_mvp.csv` | school x CIP rows with master's awards (first major) — not yet programs (B-C0-7) | IPEDS Completions + Directory |
| `reference/cip_soc_mvp.csv` | CIP -> SOC 2018 pairs for codes in scope | NCES/BLS CIP2020-SOC2018 crosswalk |
| `reference/occupation_profiles_mvp.csv` | x_occ: RIASEC in [0, 1] per CIP, with SOC coverage | O*NET Interests (OI) |
