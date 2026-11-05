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