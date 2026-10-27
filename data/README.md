# Data

| Folder | Content | Committed? |
|---|---|---|
| `synthetic/` | Fictional programs for tests and demos (`"synthetic": true`) | Yes |
| `reference/` | Public reference tables (NCES CIP 2020, IPEDS Completions extracts) | Yes, with source + date below |
| `programs/` | Real program attributes; each record has `source_url`, `retrieved_at` | Yes |
| `private/` | Student profiles, survey answers, consultant cases | **Never** (gitignored) |
| `templates/` | Empty CSV templates: admission outcomes and official rates ([docs/data-intake.md](../docs/data-intake.md)); W-41 study answers `study_*.csv` ([docs/study-protocol.md](../docs/study-protocol.md) §4). Copy to `private/` before filling | Yes (headers only) |

## Synthetic catalogue

The all-fields simulator (`src/aisc/simulation/fields.py`, W-67) adds six far-field program prototypes (fictional schools; CIP labels are modelling choices to be checked against CIP 2020 before any real use) and hidden-domain values that exist only inside the simulator.

`synthetic/programs.jsonl` — 12 fictional programs. School names are placeholders. CIP codes are used only to exercise label-space novelty and must be checked against the official CIP 2020 list before any real use. Attribute scores and `prestige` values are hand-set for testing, not measurements. The simulator (C5) redraws prestige for its own catalogue; the prototype prestige values are used only by unit tests such as the B-C1-2 probe.

## Reference tables (W-14, W-15)
Built by `scripts/build_reference.py` from the official files as downloaded; never edited by hand. `reference/SOURCES.json` records, for every input, the source URL, the download date and the SHA-256 of the file.

| File | Content | Source |
|---|---|---|
| `reference/mvp_scope.json` | CIP 2020 prefixes of data rollout v1 (hand-written decision, versioned). The product covers all fields (D10); this only limits the first rollout (B-C0-6) | — |
| `reference/requirement_taxonomy.json` | C6 requirement categories across all fields (draft-1; audited by V-C6-4) | — |
| `reference/cip2020_mvp.csv` | 6-digit CIP codes in scope, official titles and definitions | NCES CIP 2020 |
| `reference/ipeds_masters_mvp.csv` | school x CIP rows with master's awards (first major) — not yet programs (B-C0-7) | IPEDS Completions + Directory |
| `reference/cip_soc_mvp.csv` | CIP -> SOC 2018 pairs for codes in scope | NCES/BLS CIP2020-SOC2018 crosswalk |
| `reference/occupation_profiles_mvp.csv` | x_occ: RIASEC in [0, 1] per CIP, with SOC coverage | O*NET Interests (OI) |

## Third-party data and attribution

**O\*NET.** The following files are built from the O\*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license: `reference/occupation_profiles_mvp.csv`, `reference/occupation_basic_interests_mvp.csv`, `reference/soc_profiles_mvp.csv`, `reference/soc_basic_interests_mvp.csv`, `reference/basic_interest_map.csv` and `reference/interest_area_clusters.json`. Po-An (Brian) Chan has modified all or some of this information: occupation scores are rescaled to [0, 1], averaged across O\*NET-SOC codes and mapped to CIP codes, and interest areas are clustered (the exact transformations are in `scripts/build_reference.py`, `scripts/cluster_interest_areas.py` and `docs/THEORY.md`). USDOL/ETA has not approved, endorsed, or tested these modifications. O\*NET® is a trademark of USDOL/ETA.

**NCES / IPEDS.** `reference/cip2020_mvp.csv`, `reference/cip_soc_mvp.csv` and `reference/ipeds_masters_mvp.csv` are built from public U.S. government data (NCES CIP 2020, the NCES/BLS CIP 2020–SOC 2018 crosswalk, and IPEDS 2025 Completions and Directory files).

The MIT license of this repository covers the code and documentation; the third-party data above stays under its own terms.
