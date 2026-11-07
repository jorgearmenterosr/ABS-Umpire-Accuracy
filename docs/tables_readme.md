# ABS Research — data tables

Clean tables built from MLB's public Stats API (statsapi.mlb.com), one
folder per table, one subfolder per regular season 2021–2026. Each
season folder holds `part.parquet` (typed; use this in pandas/R) and
`part.csv` (same rows, for Excel). `all_seasons.parquet` in each table
folder is every season stacked (pitches: Parquet only, 4.27M rows).

Built by Owen Purnomo and Jorge Armenteros. Code and column
definitions: https://github.com/jorgearmenterosr/ABS-Umpire-Accuracy
(see `docs/conventions.md`). Coverage and known issues:
`coverage_report.md` beside this file.

| Table | One row per | Key | Notes |
| --- | --- | --- | --- |
| games | game | game_pk | status, teams, venue, plate umpire. 2,430 scheduled per season; Postponed/Cancelled kept here but excluded downstream |
| umpire_assignments | game × umpire | game_pk, official_id | position (Home Plate, First/Second/Third Base). `NO UMPIRE` placeholders removed |
| crews | game × umpire | game_pk, official_id | crew id per distinct set of officials, crew size, position change vs previous game |
| umpires | umpire | official_id | name, seasons active, Retrosheet birth date and MLB debut where matched |
| pitches | pitch | game_pk, at_bat_index, pitch_number | 52 columns: call, pitch type, location at front of plate (ft), batter zone (ft), movement (in), spin, nine trajectory parameters, count, challenge flags, umpire's original call |
| challenges | ABS challenge | challenge_id | 2026 only (empty files for 2021–2025): team, player, role (batter/catcher/pitcher), original and final call, overturned |
| challenges_remaining | pitch | game_pk, at_bat_index, pitch_number | challenges each team had left before the pitch (2 to start; −1 on a failed challenge; +1 entering an extra inning with 0) |

Column conventions: snake_case; `_id` = MLB id; `is_` = true/false;
units in the name (`_ft`, `_in`, `_mph`, `_rpm`); `_raw` = copied
verbatim from the API; empty = unknown (never 0 or "").

Two things to know before comparing seasons:
- `sz_top_ft` / `sz_bot_ft` are per-pitch, stance-based values through
  2025 and fixed per batter (ABS, from measured height) in 2026. Do not
  compare accuracy across seasons with these columns as-is; use the
  trajectory columns to recompute location on one plane.
- `plate_x_ft` / `plate_z_ft` are at the front of the plate in every
  season (the ABS zone is judged at the plate midpoint, 8.5 in back).
