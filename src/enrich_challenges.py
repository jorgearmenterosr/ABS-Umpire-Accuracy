"""Add game context and team names to every challenge.

Writes data/tables/challenges_enriched/all_seasons.parquet (+ CSV).

Usage:  python src/enrich_challenges.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"


def main() -> None:
    challenges = pd.read_parquet(
        TABLES_DIR / "challenges" / "all_seasons.parquet"
    )
    games = pd.read_parquet(TABLES_DIR / "games" / "all_seasons.parquet")
    teams = pd.read_parquet(TABLES_DIR / "teams" / "part.parquet")

    game_cols = [
        "game_pk",
        "game_date",
        "home_team_id",
        "away_team_id",
        "plate_umpire_id",
        "plate_umpire_name",
    ]
    merged = challenges.merge(games[game_cols], on="game_pk", how="left")

    is_home = merged["challenge_team_id"].eq(merged["home_team_id"])
    merged["challenger_side"] = is_home.map({True: "home", False: "away"})
    merged["opponent_team_id"] = merged["away_team_id"].where(
        is_home, merged["home_team_id"]
    )

    names = teams.set_index("team_id")["team_abbr"]
    merged["challenge_team_abbr"] = merged["challenge_team_id"].map(names)
    merged["opponent_team_abbr"] = merged["opponent_team_id"].map(names)

    assert merged["challenge_id"].is_unique
    assert merged["game_date"].notna().all(), "challenge with no matching game"

    out_dir = TABLES_DIR / "challenges_enriched"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "all_seasons.parquet", index=False)
    merged.to_csv(out_dir / "all_seasons.csv", index=False)

    print("rows:", len(merged))
    print(
        merged.groupby("challenger_side")["is_overturned"]
        .agg(["count", "mean"])
        .round(3)
    )
    top = (
        merged.groupby("challenge_team_abbr")["is_overturned"]
        .agg(["count", "mean"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(top.head(10))


if __name__ == "__main__":
    main()
