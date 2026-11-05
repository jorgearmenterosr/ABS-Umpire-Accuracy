"""Download every Final game's live feed for one season.

Reads game_pk values from data/tables/games/season=<season>/part.parquet,
saves each feed to data/raw/feed_live/<game_pk>.json, skips feeds that
already exist, so the run can be stopped and restarted.

Usage:  python src/fetch_feeds.py 2026
"""

import json
import time
import sys
from pathlib import Path

import pandas as pd

from fetch_feed import fetch_feed

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "feed_live"
TABLES_DIR = PROJECT_DIR / "data" / "tables"
KEEP_STATUSES = ("Final", "Completed Early")
PAUSE_SECONDS = 0.5
MAX_ATTEMPTS = 3
PROGRESS_EVERY = 50


def game_pks_for(season: int) -> list[int]:
    """Return the game_pk of every finished game in a season, in date order."""
    path = TABLES_DIR / "games" / f"season={season}" / "part.parquet"
    games = pd.read_parquet(path)
    finished = games[games["status"].isin(KEEP_STATUSES)]
    finished = finished.sort_values(["game_date", "game_pk"])
    return finished["game_pk"].astype(int).tolist()


def download(game_pk: int) -> str:
    """Save one feed; return 'saved', 'skipped' or 'failed'."""
    path = RAW_DIR / f"{game_pk}.json"
    if path.exists():
        return "skipped"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            feed = fetch_feed(game_pk)
        except Exception as error:  # network or HTTP error
            print(f"  {game_pk}: attempt {attempt} failed ({error})")
            time.sleep(PAUSE_SECONDS * 2**attempt)
            continue
        with path.open("w") as fh:
            json.dump(feed, fh)
        time.sleep(PAUSE_SECONDS)
        return "saved"
    return "failed"


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    game_pks = game_pks_for(season)[:limit]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"season {season}: {len(game_pks)} finished games to check")

    counts = {"saved": 0, "skipped": 0, "failed": 0}
    failed = []
    start = time.time()
    for i, game_pk in enumerate(game_pks, start=1):
        result = download(game_pk)
        counts[result] += 1
        if result == "failed":
            failed.append(game_pk)
        if i % PROGRESS_EVERY == 0 or i == len(game_pks):
            minutes = (time.time() - start) / 60
            print(f"{i}/{len(game_pks)}  {counts}  {minutes:.1f} min")

    print("done:", counts)
    if failed:
        print("failed game_pks:", failed)

if __name__ == "__main__":
    main()