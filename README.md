# ABS-Umpire-Accuracy

Evaluating whether MLB's Automated Ball-Strike (ABS) challenge system
improves umpire accuracy, using pitch-level and challenge data from
2021–2026 built from MLB's public Stats API.

**Status (7 Nov 2026):** data complete — 4,270,416 pitches in 14,577
regular-season games, every 2026 challenge reconstructed and verified
against MLB's published totals. Analysis phase starts 11 Nov.

## Setup

    git clone https://github.com/jorgearmenterosr/ABS-Umpire-Accuracy.git
    cd ABS-Umpire-Accuracy
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Every terminal session starts with `source .venv/bin/activate`.
Scripts are run from the project root as `python src/<script>.py`.

## How the data is organised

Three layers (details in `docs/conventions.md`):

| Layer | Where | What |
| --- | --- | --- |
| raw | `data/raw/` | MLB's JSON responses, saved untouched (13 GB, not in git) |
| tables | `data/tables/<table>/season=<yyyy>/` | clean tables, one row per thing, Parquet + CSV (4 GB, not in git; shared via Drive) |
| analysis | `notebooks/` | joins and results built from tables |

Tables: `games`, `umpire_assignments`, `crews`, `umpires`, `pitches`
(52 columns), `challenges`, `challenges_remaining`, `teams`,
`challenges_enriched`. Keys and columns: `docs/tables_readme.md`.

## Build the dataset from scratch

Run in this order, once per season (2021–2026). A season takes about
45 minutes, almost all of it the feed download.

    python src/fetch_season_schedule.py 2026   # games + umpire crews (2 min)
    python src/fetch_feeds.py 2026             # every game's live feed (~40 min, resumable)
    python src/build_pitches.py 2026           # pitches table
    python src/build_challenges.py 2026        # challenges + challenges_remaining
    python src/build_crews.py 2026             # crews + umpire workload report

Then, once for all seasons:

    python src/build_teams.py                  # 30-club lookup
    python src/combine_seasons.py              # all_seasons.parquet per table
    python src/enrich_challenges.py            # challenges + game context + team names
    python src/build_umpires.py 2021 2022 2023 2024 2025 2026   # needs data/raw/retrosheet/biofile0.csv

Downloads skip files that already exist, so a stopped run can be
restarted with the same command. Use `caffeinate -i python ...` on a
Mac so it does not sleep mid-download.

## Scripts

| Script | What it does| 
| --- | --- |
| `fetch_schedule.py` | one day of the schedule with umpires (building block)|
| `fetch_season_schedule.py` | whole season → `games`, `umpire_assignments`|
| `fetch_feed.py` | one game's live feed (building block)|
| `fetch_feeds.py` | every finished game's feed for a season, resumable|
| `build_pitches.py` | feeds → `pitches`, with the umpire's original call reconstructed| 
| `build_challenges.py` | feeds → `challenges`, `challenges_remaining`, checked against MLB's counters| 
| `build_crews.py` | assignments → `crews`, plate-games histogram|
| `build_umpires.py` | `umpires` with Retrosheet birth and debut dates| 
| `build_teams.py` | `teams` lookup| 
| `combine_seasons.py` | per-season files → one file per table| 
| `enrich_challenges.py` | challenges joined to games and teams|
| `explore_feed.py` | scratch: readable report of one feed|


## Documents

- `docs/conventions.md` — layers, keys, naming, the rules found on real feeds
- `docs/tables_readme.md` — one-line description and key of every table
- Research plan and methodology memo: shared Claude Docs (links in the project notes)
