"""Fetch postseason games and umpire assignments for 2021–2026."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "schedule"
TABLES_DIR = PROJECT_DIR / "data" / "tables"
REPORTS_DIR = PROJECT_DIR / "reports"
API_URL = "https://statsapi.mlb.com/api/v1/schedule"

SEASONS = tuple(range(2021, 2027))
POSTSEASON_GAME_TYPES = ("F", "D", "L", "W")
NO_UMPIRE_NAME = "NO UMPIRE"
MAX_RETRIES = 4
BACKOFF_SECONDS = 1.0
REQUEST_PAUSE_SECONDS = 0.25
SESSION = requests.Session()

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


def fetch_schedule(season, game_type):
    """Fetch one postseason type across the season's postseason date range."""
    params = {
        "sportId": 1,
        "season": season,
        "startDate": f"{season}-09-01",
        "endDate": f"{season}-12-31",
        "gameType": game_type,
        "hydrate": "officials",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response: requests.Response = SESSION.get(
                API_URL,
                params=params,
                timeout=30,
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                time.sleep(BACKOFF_SECONDS * (2**attempt))
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as error:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)

            if (
                status_code is not None
                and status_code != 429
                and status_code < 500
            ):
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            time.sleep(BACKOFF_SECONDS * (2**attempt))

    raise RuntimeError(
        f"Could not fetch {game_type} schedule for {season}"
    )


def load_or_fetch(season, game_type, refresh):
    """Load a cached response or fetch and cache it."""
    raw_path = RAW_DIR / f"postseason_{season}_{game_type}.json"

    if raw_path.exists() and not refresh:
        with raw_path.open(encoding="utf-8") as raw_file:
            payload = json.load(raw_file)
        return payload, raw_path.relative_to(PROJECT_DIR).as_posix(), False

    payload = fetch_schedule(season, game_type)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with raw_path.open("w", encoding="utf-8") as raw_file:
        json.dump(payload, raw_file, indent=2)

    return payload, raw_path.relative_to(PROJECT_DIR).as_posix(), True


def extract_rows(payload, season, requested_type, source_file):
    """Extract game and official rows from a schedule response."""
    game_rows = []
    assignment_rows = []

    for day in payload.get("dates", []):
        schedule_date = day.get("date")

        for game in day.get("games", []):
            game_type = game.get("gameType")
            if game_type != requested_type:
                raise ValueError(
                    f"Requested game type {requested_type}, "
                    f"but API returned {game_type}."
                )

            officials = game.get("officials") or []
            plate_umpire = {}

            for entry in officials:
                official = entry.get("official") or {}
                if (
                    entry.get("officialType") == "Home Plate"
                    and official.get("fullName") != NO_UMPIRE_NAME
                ):
                    plate_umpire = official
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

            game_rows.append(
                {
                    "game_pk": game["gamePk"],
                    "season": int(game.get("season", season)),
                    "_schedule_date": schedule_date,
                    "game_date": game.get("officialDate"),
                    "status": status.get("detailedState"),
                    "home_team_id": home.get("id"),
                    "away_team_id": away.get("id"),
                    "plate_umpire_id": plate_umpire.get("id"),
                    "plate_umpire_name": plate_umpire.get("fullName"),
                    "game_type": game_type,
                    "is_doubleheader": is_doubleheader,
                    "game_number": game.get("gameNumber"),
                    "venue_id": venue.get("id"),
                    "source_file": source_file,
                }
            )

            for entry in officials:
                official = entry.get("official") or {}
                official_name = official.get("fullName")

                if (
                    official_name == NO_UMPIRE_NAME
                    or official.get("id") is None
                ):
                    continue

                assignment_rows.append(
                    {
                        "game_pk": game["gamePk"],
                        "season": int(game.get("season", season)),
                        "_schedule_date": schedule_date,
                        "official_id": official.get("id"),
                        "official_name": official_name,
                        "official_type": entry.get("officialType"),
                        "source_file": source_file,
                    }
                )

    return game_rows, assignment_rows


def resolve_repeated_games(game_rows, assignment_rows):
    """Keep the schedule occurrence matching each game's official date."""
    rows_by_game = {}
    for row in game_rows:
        rows_by_game.setdefault(row["game_pk"], []).append(row)

    kept_games = []
    selected_dates = {}
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
                    "Inspect its cached raw response."
                )
            selected = matches[0]

        selected_game = dict(selected)
        selected_game.pop("_schedule_date")
        kept_games.append(selected_game)

        selected_dates[game_pk] = (
            selected["source_file"],
            selected["_schedule_date"],
        )

    status_counts = {}
    for row in game_rows:
        status = row["status"]
        label = "<null>" if status is None else str(status)
        status_counts[label] = status_counts.get(label, 0) + 1

    kept_assignments = []
    for row in assignment_rows:
        selected_source, selected_date = selected_dates[row["game_pk"]]
        if (
            row["source_file"] == selected_source
            and row["_schedule_date"] == selected_date
        ):
            assignment = dict(row)
            assignment.pop("_schedule_date")
            kept_assignments.append(assignment)

    return kept_games, kept_assignments, status_counts, repeated_count


def make_table(rows, columns):
    """Build a table with the project's nullable integer and text types."""
    table = pd.DataFrame(rows, columns=columns)

    integer_columns = [
        column
        for column in columns
        if column.endswith("_id")
        or column in {"game_pk", "season", "game_number"}
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


def save_season_table(table, table_name, season):
    """Save one season's table as Parquet and CSV."""
    output_dir = TABLES_DIR / table_name / f"season={season}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "part.parquet"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(output_dir / "part.csv", index=False)
    return parquet_path


def save_all_seasons(table, table_name):
    """Save the combined postseason table as Parquet and CSV."""
    output_dir = TABLES_DIR / table_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "all_seasons.parquet"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(output_dir / "all_seasons.csv", index=False)
    return parquet_path


def build_role_counts(assignments):
    """Count postseason games worked by each umpire and role."""
    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "official_id",
                "official_name",
                "total_postseason_games",
            ]
        )

    keys = ["season", "official_id", "official_name"]
    role_counts = assignments.pivot_table(
        index=keys,
        columns="official_type",
        values="game_pk",
        aggfunc="nunique",
        fill_value=0,
    )

    role_counts.columns = [
        f"{str(role).lower().replace(' ', '_')}_games"
        for role in role_counts.columns
    ]
    role_counts = role_counts.reset_index()

    total_games = (
        assignments.groupby(keys)["game_pk"]
        .nunique()
        .rename("total_postseason_games")
        .reset_index()
    )

    return role_counts.merge(total_games, on=keys, validate="one_to_one")


def main():
    """Fetch and save postseason schedules and umpire role counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh schedule responses instead of using cached JSON.",
    )
    args = parser.parse_args()

    all_season_games = []
    all_season_assignments = []

    for season in SEASONS:
        game_rows = []
        assignment_rows = []
        fetched_count = 0

        for game_type in POSTSEASON_GAME_TYPES:
            payload, source_file, was_fetched = load_or_fetch(
                season,
                game_type,
                args.refresh,
            )
            fetched_count += int(was_fetched)

            season_games, season_assignments = extract_rows(
                payload,
                season,
                game_type,
                source_file,
            )
            game_rows.extend(season_games)
            assignment_rows.extend(season_assignments)

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

        if games.empty:
            raise ValueError(
                f"No postseason games found for {season}. "
                "Check the saved raw responses before continuing."
            )

        if games["game_pk"].duplicated().any():
            raise ValueError(f"Postseason game_pk is not unique in {season}.")
        if assignments.duplicated(["game_pk", "official_id"]).any():
            raise ValueError(
                f"Postseason game and official keys are not unique in {season}."
            )
        if not (games["season"] == season).all():
            raise ValueError(f"Unexpected season in postseason games: {season}")
        if not assignments.empty and not (
            assignments["season"] == season
        ).all():
            raise ValueError(
                f"Unexpected season in postseason assignments: {season}"
            )

        game_path = save_season_table(
            games,
            "postseason_games",
            season,
        )
        assignment_path = save_season_table(
            assignments,
            "postseason_umpire_assignments",
            season,
        )

        print(
            f"{season}: {len(games):,} games; "
            f"{len(assignments):,} umpire assignments; "
            f"{fetched_count} fresh responses"
        )
        print(f"  Game types: {games['game_type'].value_counts().to_dict()}")
        print(f"  Repeated games resolved: {repeated_count}")
        print(f"  Statuses before resolving repeats: {raw_status_counts}")
        print(
            "  Games missing plate umpire: "
            f"{int(games['plate_umpire_id'].isna().sum())}"
        )
        print(f"  Saved games: {game_path}")
        print(f"  Saved assignments: {assignment_path}")

        all_season_games.append(games)
        all_season_assignments.append(assignments)

    combined_games = pd.concat(all_season_games, ignore_index=True)
    combined_assignments = pd.concat(
        all_season_assignments,
        ignore_index=True,
    )

    if combined_games["game_pk"].duplicated().any():
        raise ValueError("Combined postseason game_pk is not unique.")
    if combined_assignments.duplicated(["game_pk", "official_id"]).any():
        raise ValueError(
            "Combined postseason game and official keys are not unique."
        )

    all_games_path = save_all_seasons(
        combined_games,
        "postseason_games",
    )
    all_assignments_path = save_all_seasons(
        combined_assignments,
        "postseason_umpire_assignments",
    )

    role_counts = build_role_counts(combined_assignments)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    role_path = REPORTS_DIR / "tables" / "postseason_umpire_roles.csv"
    role_path.parent.mkdir(parents=True, exist_ok=True)
    role_counts.to_csv(role_path, index=False)

    games_by_type = (
        combined_games.groupby(["season", "game_type"])
        .size()
        .rename("games")
        .reset_index()
    )
    type_path = REPORTS_DIR / "tables" / "postseason_games_by_type.csv"
    games_by_type.to_csv(type_path, index=False)

    print(f"Saved all-season games: {all_games_path}")
    print(f"Saved all-season assignments: {all_assignments_path}")
    print(f"Saved umpire role counts: {role_path}")
    print(f"Saved game counts by round: {type_path}")
    print(f"Combined postseason games: {len(combined_games):,}")
    print(f"Umpire-season role rows: {len(role_counts):,}")
    print(
        "Unique postseason umpires: "
        f"{role_counts['official_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()