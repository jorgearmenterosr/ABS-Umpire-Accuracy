# How we organise the data

## Three layers

1. **raw** – `data/raw/`. The exact answer MLB's server gave us, saved as JSON.
   We never edit these files. If something looks wrong later, we come back here.
2. **tables** – `data/tables/`. Clean tables built from raw. One table per kind
   of thing (games, pitches, ...), one row per thing, one folder per season,
   saved as Parquet. This is what we do research on.
3. **analysis** – notebooks. Joins, filters and results built from tables.
   Nothing here is saved as "the data"; it can always be rebuilt.

## Tables and their keys

A key is the column (or columns) that identifies one row, so rows can be
joined and never counted twice.

| table | one row per | key |
| --- | --- | --- |
| games | game | game_pk |
| umpire_assignments | game and umpire | game_pk, official_id |
| pitches | pitch | game_pk, at_bat_index, pitch_number |
| challenges | ABS challenge | challenge_id |
| players | player | player_id |
| umpires | umpire | official_id |

## Naming

- Column names are lower case with underscores: `plate_umpire_id`.
- Id columns end in `_id`. True/false columns start with `is_`.
- Units go in the name: `plate_z_ft`, `release_speed_mph`, `distance_in`.
- A value copied straight from MLB keeps the ending `_raw`: `pitch_type_raw`.
- If we do not know a value, it is empty (null). Never 0, never "" , never False.
- Every table has `season` and `source_file` so any number can be traced back.