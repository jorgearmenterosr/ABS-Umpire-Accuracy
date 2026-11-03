"""Fetch one game's live feed (every pitch and every challenge).

Usage:  python src/fetch_feed.py 822970
"""

import json
import sys
from pathlib import Path

import requests

BASE_URL = "https://statsapi.mlb.com"
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "feed_live"


def fetch_feed(game_pk: int) -> dict:
    """Return the live-feed JSON for one game."""
    url = f"{BASE_URL}/api/v1.1/game/{game_pk}/feed/live"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 822970
    feed = fetch_feed(game_pk)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{game_pk}.json"
    with path.open("w") as fh:
        json.dump(feed, fh)
    print("saved", path)
    print("top-level keys:", sorted(feed.keys()))


if __name__ == "__main__":
    main()