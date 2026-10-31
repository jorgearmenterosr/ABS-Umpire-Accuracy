"""Fetch one day of the MLB schedule with umpires.

Usage:  python src/fetch_schedule.py 2026-06-10
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com"
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "schedule"
TABLES_DIR = PROJECT_DIR / "data" / "tables"


def fetch_schedule(date: str) -> dict:
    """Return the schedule JSON for one date with officials."""
    params = {
        "sportId": 1,
        "startDate": date,
        "endDate": date,
        "gameType": "R",
        "hydrate": "officials",
    }
    response = requests.get(
        BASE_URL + "/api/v1/schedule", params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def save_raw(payload: dict, date: str) -> Path:
    """Write the untouched JSON to data/raw/schedule/<date>.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{date}.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def plate_umpire(game: dict) -> tuple:
    """Return (id, name) of the home-plate umpire, or (None, None)."""
    for entry in game["officials"]:
        if entry["officialType"] == "Home Plate":
            return entry["official"]["id"], entry["official"]["fullName"]
    return None, None


def games_table(payload: dict) -> pd.DataFrame:
    """Return one tidy row per game."""
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
    """Write a table partition to data/tables/<name>/season=<season>/."""
    out_dir = TABLES_DIR / name / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"
    table.to_parquet(path, index=False)
    return path


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-10"
    payload = fetch_schedule(date)
    print("saved", save_raw(payload, date))
    table = games_table(payload)
    assert table["game_pk"].is_unique, "a game appears twice"
    season = int(table["season"].iloc[0])
    print("saved", save_table(table, "games", season), "rows:", len(table))


if __name__ == "__main__":
    main()