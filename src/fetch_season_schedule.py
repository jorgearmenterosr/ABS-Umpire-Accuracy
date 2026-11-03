"""Fetch a season of regular-season schedules and umpire assignments."""

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from requests.exceptions import RequestException

from fetch_schedule import fetch_schedule, save_raw, save_table

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "schedule"
REQUEST_PAUSE_SECONDS = 0.5

# Season dates are based on MLB's regular-season schedule announcements.
# 2021: https://www.mlb.com/press-release/press-release-mlb-announces-2021-regular-season-schedule
# 2022: https://www.mlb.com/press-release/press-release-mlb-announces-revised-2022-regular-season-schedule
# 2023: https://www.mlb.com/press-release/press-release-mlb-announces-2023-regular-season-schedule
# 2024: https://www.mlb.com/amp/press-release/press-release-mlb-announces-2024-regular-season-schedule.html
# 2025: https://www.mlb.com/press-release/press-release-mlb-announces-2025-regular-season-schedule
# 2026 dates are from the Research Project briefing.
SEASON_DATES = {
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-30"),
    2025: ("2025-03-18", "2025-09-28"),
    2026: ("2026-03-25", "2026-09-27"),
}

GAME_COLUMNS = [
    "game_pk",
    "season",
    "game_date",
    "status",
    "home_team_id",
    "away_team_id",
    "plate_umpire_id",
    "plate_umpire_name",
    "game_type",
    "is_doubleheader",
    "game_number",
    "venue_id",
    "source_file",
]

ASSIGNMENT_COLUMNS = [
    "game_pk",
    "season",
    "official_id",
    "official_name",
    "official_type",
    "source_file",
]


def season_dates(season: int) -> list[str]:
    """Return every date in the season's regular-season range."""
    start_text, end_text = SEASON_DATES[season]
    start_date = date.fromisoformat(start_text)
    end_date = date.fromisoformat(end_text)
    day_count = (end_date - start_date).days
    return [
        (start_date + timedelta(days=offset)).isoformat()
        for offset in range(day_count + 1)
    ]


def load_or_fetch_date(date_text: str) -> tuple[dict, str, bool]:
    """Load a saved schedule response or fetch and save it."""
    raw_path = RAW_DIR / f"{date_text}.json"
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as raw_file:
            payload = json.load(raw_file)
        source_file = raw_path.relative_to(PROJECT_DIR).as_posix()
        return payload, source_file, False

    payload = fetch_schedule(date_text)
    saved_path = save_raw(payload, date_text)
    source_file = saved_path.relative_to(PROJECT_DIR).as_posix()
    return payload, source_file, True


def extract_rows(
    payload: dict, source_file: str
) -> tuple[list[dict], list[dict]]:
    """Extract game and umpire assignment rows from one response."""
    games = []
    assignments = []

    for day in payload.get("dates", []):
        schedule_date = day.get("date")

        for game in day.get("games", []):
            officials = game.get("officials") or []
            plate_umpire = {}
            for entry in officials:
                if entry.get("officialType") == "Home Plate":
                    plate_umpire = entry.get("official") or {}
                    break

            doubleheader = game.get("doubleHeader")
            if doubleheader == "Y":
                is_doubleheader = True
            elif doubleheader == "N":
                is_doubleheader = False
            else:
                is_doubleheader = None

            teams = game.get("teams") or {}
            home = (teams.get("home") or {}).get("team") or {}
            away = (teams.get("away") or {}).get("team") or {}
            venue = game.get("venue") or {}
            status = game.get("status") or {}

            games.append(
                {
                    "game_pk": game["gamePk"],
                    "season": int(game["season"]),
                    "_schedule_date": schedule_date,
                    "game_date": game.get("officialDate"),
                    "status": status.get("detailedState"),
                    "home_team_id": home.get("id"),
                    "away_team_id": away.get("id"),
                    "plate_umpire_id": plate_umpire.get("id"),
                    "plate_umpire_name": plate_umpire.get("fullName"),
                    "game_type": game.get("gameType"),
                    "is_doubleheader": is_doubleheader,
                    "game_number": game.get("gameNumber"),
                    "venue_id": venue.get("id"),
                    "source_file": source_file,
                }
            )

            for entry in officials:
                official = entry.get("official") or {}
                assignments.append(
                    {
                        "game_pk": game["gamePk"],
                        "season": int(game["season"]),
                        "_schedule_date": schedule_date,
                        "official_id": official.get("id"),
                        "official_name": official.get("fullName"),
                        "official_type": entry.get("officialType"),
                        "source_file": source_file,
                    }
                )

    return games, assignments


def resolve_repeated_games(
    game_rows: list[dict], assignment_rows: list[dict]
) -> tuple[list[dict], list[dict], dict[str, int], int]:
    """Keep the occurrence whose schedule date matches officialDate."""
    rows_by_game = {}
    for row in game_rows:
        rows_by_game.setdefault(row["game_pk"], []).append(row)

    kept_games = []
    kept_sources = {}
    repeated_count = 0

    for game_pk, rows in rows_by_game.items():
        if len(rows) == 1:
            selected = rows[0]
        else:
            repeated_count += 1
            matches = [
                row
                for row in rows
                if row["_schedule_date"] == row["game_date"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Game {game_pk} has no unique schedule-date match. "
                    "Inspect its raw JSON before changing the selection rule."
                )
            selected = matches[0]

        selected = dict(selected)
        selected.pop("_schedule_date")
        kept_games.append(selected)
        kept_sources[game_pk] = selected["source_file"]

    status_counts = {}
    for row in game_rows:
        status = row["status"]
        label = "<null>" if status is None else str(status)
        status_counts[label] = status_counts.get(label, 0) + 1

    kept_assignments = []
    for row in assignment_rows:
        if kept_sources.get(row["game_pk"]) == row["source_file"]:
            assignment = dict(row)
            assignment.pop("_schedule_date")
            kept_assignments.append(assignment)

    return kept_games, kept_assignments, status_counts, repeated_count


def make_table(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build a table and apply nullable integer and text column types."""
    table = pd.DataFrame(rows, columns=columns)

    integer_columns = [
        column
        for column in columns
        if column.endswith("_id") or column in {"season", "game_number"}
    ]
    text_columns = [
        column
        for column in columns
        if column not in integer_columns and column != "is_doubleheader"
    ]

    for column in integer_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype(
            "Int64"
        )

    for column in text_columns:
        table[column] = table[column].astype("string")

    if "is_doubleheader" in table.columns:
        table["is_doubleheader"] = table["is_doubleheader"].astype("boolean")

    return table


def summarize(
    games: pd.DataFrame,
    assignments: pd.DataFrame,
    fetched_dates: int,
    raw_status_counts: dict[str, int],
    repeated_count: int,
) -> None:
    """Print coverage, status, and umpire assignment counts."""
    final_games = games[games["status"] == "Final"]
    final_missing = final_games[final_games["plate_umpire_id"].isna()]
    all_missing = games[games["plate_umpire_id"].isna()]

    if games.empty:
        average_assignments = 0.0
    else:
        assignment_counts = assignments.groupby("game_pk").size()
        average_assignments = (
            assignment_counts.reindex(games["game_pk"]).fillna(0).mean()
        )

    print(f"New schedule dates fetched this run: {fetched_dates}")
    print(f"Games in table: {len(games)}")
    print(f"Repeated gamePk values resolved: {repeated_count}")
    print(f"Final games: {len(final_games)}")
    print(f"Final games missing a plate umpire: {len(final_missing)}")
    print(f"All games missing a plate umpire: {len(all_missing)}")
    print(f"Average assignments per game: {average_assignments:.2f}")
    print(f"Statuses before resolving repeats: {raw_status_counts}")

    if not all_missing.empty:
        print("Games missing a plate umpire:")
        print(
            all_missing[["game_pk", "game_date", "status"]].to_string(
                index=False
            )
        )


def main() -> None:
    """Fetch, validate, and save schedule tables for one season."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/fetch_season_schedule.py <season>")

    season = int(sys.argv[1])
    if season not in SEASON_DATES:
        choices = ", ".join(str(year) for year in SEASON_DATES)
        raise SystemExit(f"Season must be one of: {choices}")

    game_rows = []
    assignment_rows = []
    failed_dates = []
    fetched_dates = 0

    for date_text in season_dates(season):
        try:
            payload, source_file, was_fetched = load_or_fetch_date(date_text)
        except RequestException as error:
            failed_dates.append(f"{date_text}: {error}")
            continue

        fetched_dates += int(was_fetched)
        games_for_date, assignments_for_date = extract_rows(
            payload, source_file
        )
        game_rows.extend(games_for_date)
        assignment_rows.extend(assignments_for_date)

        if was_fetched:
            time.sleep(REQUEST_PAUSE_SECONDS)

    (
        game_rows,
        assignment_rows,
        raw_status_counts,
        repeated_count,
    ) = resolve_repeated_games(game_rows, assignment_rows)

    games = make_table(game_rows, GAME_COLUMNS)
    assignments = make_table(assignment_rows, ASSIGNMENT_COLUMNS)

    assert games["game_pk"].is_unique, "game_pk is not unique in games"
    assert not assignments.duplicated(["game_pk", "official_id"]).any(), (
        "game_pk and official_id are not unique"
    )
    assert (games["season"] == season).all(), "Unexpected season in games"
    assert (assignments["season"] == season).all(), (
        "Unexpected season in umpire assignments"
    )

    print("Saved games:", save_table(games, "games", season))
    print(
        "Saved umpire assignments:",
        save_table(assignments, "umpire_assignments", season),
    )
    summarize(
        games,
        assignments,
        fetched_dates,
        raw_status_counts,
        repeated_count,
    )

    print(f"Schedule dates that failed to fetch: {len(failed_dates)}")
    for failure in failed_dates:
        print(" ", failure)


if __name__ == "__main__":
    main()
