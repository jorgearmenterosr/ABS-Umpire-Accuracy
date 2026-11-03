"""Fetch one day of the MLB schedule with umpires.

Usage: python src/fetch_schedule.py 2026-06-10
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests import Response
from requests.exceptions import RequestException

BASE_URL = "https://statsapi.mlb.com"
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "schedule"
TABLES_DIR = PROJECT_DIR / "data" / "tables"
MAX_RETRIES = 4
BACKOFF_SECONDS = 1.0
SESSION = requests.Session()


def fetch_schedule(date: str) -> dict:
    """Return the schedule JSON for one date with officials."""
    params = {
        "sportId": 1,
        "startDate": date,
        "endDate": date,
        "gameType": "R",
        "hydrate": "officials",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response: Response = SESSION.get(
                BASE_URL + "/api/v1/schedule",
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

        except RequestException as error:
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

    raise RuntimeError(f"Could not fetch schedule for {date}")


def save_raw(payload: dict, date: str) -> Path:
    """Write the untouched JSON response to the raw schedule folder."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{date}.json"

    with path.open("w", encoding="utf-8") as raw_file:
        json.dump(payload, raw_file, indent=2)

    return path


def plate_umpire(game: dict) -> tuple:
    """Return the home-plate umpire ID and name, if available."""
    for entry in game["officials"]:
        if entry["officialType"] == "Home Plate":
            official = entry["official"]
            return official["id"], official["fullName"]

    return None, None


def games_table(payload: dict) -> pd.DataFrame:
    """Return one tidy row per game in a schedule response."""
    rows = []

    for day in payload["dates"]:
        for game in day["games"]:
            umpire_id, umpire_name = plate_umpire(game)
            rows.append(
                {
                    "game_pk": game["gamePk"],
                    "season": int(game["season"]),
                    "game_date": game["officialDate"],
                    "status": game["status"]["detailedState"],
                    "home_team_id": game["teams"]["home"]["team"]["id"],
                    "away_team_id": game["teams"]["away"]["team"]["id"],
                    "plate_umpire_id": umpire_id,
                    "plate_umpire_name": umpire_name,
                }
            )

    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, name: str, season: int) -> Path:
    """Write a season table as Parquet and CSV."""
    output_dir = TABLES_DIR / name / f"season={season}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "part.parquet"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(output_dir / "part.csv", index=False)

    return parquet_path


def main() -> None:
    """Fetch and save one day's schedule."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/fetch_schedule.py YYYY-MM-DD")

    date_text = sys.argv[1]
    payload = fetch_schedule(date_text)
    print("Saved raw response:", save_raw(payload, date_text))

    table = games_table(payload)
    assert table["game_pk"].is_unique, "A game appears more than once"

    if not table.empty:
        season = int(table["season"].iloc[0])
        print("Saved games table:", save_table(table, "games", season))
        print("Rows:", len(table))


if __name__ == "__main__":
    main()
