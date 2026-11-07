"""Build the teams lookup table (30 MLB clubs).

Usage:  python src/build_teams.py
"""

import json
from pathlib import Path

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "teams.json"
OUT_DIR = PROJECT_DIR / "data" / "tables" / "teams"
URL = "https://statsapi.mlb.com/api/v1/teams"


def fetch_teams() -> dict:
    """Return the MLB teams JSON."""
    response = requests.get(URL, params={"sportId": 1}, timeout=30)
    response.raise_for_status()
    return response.json()


def teams_table(payload: dict) -> pd.DataFrame:
    """Return one row per club."""
    rows = [
        {
            "team_id": t["id"],
            "team_name": t["name"],
            "team_abbr": t["abbreviation"],
            "league": t.get("league", {}).get("name"),
            "division": t.get("division", {}).get("name"),
            "venue_id": t.get("venue", {}).get("id"),
        }
        for t in payload["teams"]
    ]
    table = pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)
    assert table["team_id"].is_unique
    return table


def main() -> None:
    payload = fetch_teams()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w") as fh:
        json.dump(payload, fh, indent=2)
    table = teams_table(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "part.parquet", index=False)
    table.to_csv(OUT_DIR / "part.csv", index=False)
    print(table.to_string(index=False))
    print("teams:", len(table))


if __name__ == "__main__":
    main()
