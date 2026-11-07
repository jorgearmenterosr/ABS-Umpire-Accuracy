# How we organise the data

## Three layers

1. **Raw** — `data/raw/` contains the exact JSON responses from MLB.
   Save each response before parsing it. Never edit these files.
2. **Tables** — `data/tables/` contains clean tables, one Parquet file per
   table per season, with a matching CSV copy beside it. These tables are
   used for research.
3. **Analysis** — notebooks contain joins, filters, and results. They are
   built from the tables; nothing in a notebook is saved as "the data."

## Tables and their keys

A key identifies each row so rows can be joined and not counted twice.

| Table | One row per | Key |
| --- | --- | --- |
| games | game | game_pk |
| umpire_assignments | game and umpire | game_pk, official_id |
| pitches | pitch | game_pk, at_bat_index, pitch_number |
| challenges | ABS challenge | challenge_id |
| players | player | player_id |
| umpires | umpire | official_id |
| crews | game and assigned umpire | game_pk, official_id |

Before saving, each table must check that its key is unique.

## Naming and values

- Use lowercase names with underscores, such as `plate_umpire_id`.
- ID columns end in `_id`. True/false columns start with `is_`.
- Include units in measurement names: `plate_z_ft`, `release_speed_mph`,
  `distance_in`.
- Values copied verbatim from the API end in `_raw`: `pitch_type_raw`.
- Unknown values are null. Never `0`, an empty string, or `False` for an
  unknown value.
- Every table includes `season` and `source_file` so rows can be traced
  to their source.
- Use integer types for IDs, nullable `Int64` when an ID may be missing,
  booleans for true/false values, and strings for text.
- `game_pk` and every `_id` column are nullable integers (`Int64`) in
  every table. A table's columns and types must be identical across
  seasons; rebuild every season with the same script version before
  combining.

## Season table files

Save each table under `data/tables/<table>/season=<year>/`. Each season
folder contains `part.parquet` and a matching `part.csv`, written with
`index=False`. The `data/` and `reports/` folders are excluded from Git;
do not commit them.

## Repeated schedule entries

MLB may return a game under both its original schedule date and its
rescheduled date with the same `gamePk`. In the games and umpire
assignment tables, keep the occurrence whose schedule date matches the
game's `officialDate`. Report the status counts before resolving repeats
so postponed occurrences remain visible in the run summary.

## Known season differences (verified on real feeds, 2 Nov 2026)

- 2021: `sz_top_ft` / `sz_bot_ft` change pitch to pitch for the same
  batter (set from the stance). 2026: one fixed value per batter (ABS,
  from height). Accuracy across seasons must use one common zone, not
  these columns as-is.
- Game status can be "Completed Early" (shortened game), not only
  "Final". Decide before the full build which statuses count.
- `gameData.absChallenges.<side>.remaining` does not include the
  extra-inning grant (a team at 0 entering the 10th gets 1). Check
  `usedSuccessful` / `usedFailed` against our counts instead.
- `reviewDetails` records every replay review (tag plays, home runs,
  hit-by-pitch, ...). Only `reviewType == "MJ"` is an ABS ball/strike
  challenge; only those carry `player`.
- Game 822717 (2026-06-25, PHI@WSH): counters say WSH won 1 / lost 1,
  but the feed records only one WSH challenge (upheld). One known
  counter mismatch; left as-is and flagged.

## Crews and umpire rotations

The `crews` table has one row per worked game and assigned umpire. A
`crew_id` represents the same set of officials within a season, regardless
of their positions. The `official_type` column records each umpire's
position in that game.

`previous_game_pk` and `previous_official_type` identify the umpire's
previous worked game and position. `is_position_change` is null when the
umpire has no previous game in that season.

The pipeline counts `Final` and `Completed Early` games as worked.
Postponed and cancelled games are excluded. Crews with fewer than four
listed umpires are retained and marked with `is_complete_crew = False`.
For example, game 632512 in 2021 had three listed umpires and no
Second Base umpire.

The season summary CSV and plate-game histogram are saved under
`reports/`. The reports folder is git-ignored.

## Placeholder officials in schedule responses

When the schedule response lists an official with the full name `NO UMPIRE`,
the raw JSON remains unchanged, but that placeholder is omitted from
`umpire_assignments`. The position is treated as unassigned and appears as
an incomplete crew in the crews table.

## Global umpires table

The `umpires` table has one row per `official_id` across all included seasons.
It is stored at `data/tables/umpires/part.parquet` and `part.csv`, without a
season partition or a single `season` column. The `seasons_active` column lists
the seasons in which each umpire worked. Each row's `source_file` lists the
assignment table files for those seasons and the Retrosheet biofile.

Retrosheet matches use exact normalized names from `fullname` or
`usename` plus `lastname`. Names without a unique match remain unmatched and
are listed in `reports/umpire_match_issues.csv`; they are not guessed.