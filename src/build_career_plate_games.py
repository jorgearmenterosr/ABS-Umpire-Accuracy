"""Build umpire-season game counts and career totals before each season."""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"


def main() -> None:
    archive_files = sorted(
        (TABLES_DIR / "umpire_assignments_archive").glob(
            "season=*/part.parquet"
        )
    )
    current_files = sorted(
        (TABLES_DIR / "umpire_assignments").glob("season=*/part.parquet")
    )
    files = archive_files + current_files

    if not files:
        raise SystemExit("No umpire assignment Parquet files were found.")

    assignments = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )

    assignments = assignments.dropna(
        subset=["season", "game_pk", "official_id"]
    ).drop_duplicates(["season", "game_pk", "official_id"])

    assignments["is_plate"] = assignments["official_type"].eq("Home Plate")

    career = (
        assignments.groupby(["official_id", "season"], as_index=False)
        .agg(
            official_name=("official_name", "last"),
            games_worked=("game_pk", "nunique"),
            plate_games=("is_plate", "sum"),
        )
        .sort_values(["official_id", "season"])
        .reset_index(drop=True)
    )

    career["career_games_worked_before_season"] = (
        career.groupby("official_id")["games_worked"].transform(
            lambda values: values.cumsum() - values
        )
    )
    career["career_plate_games_before_season"] = (
        career.groupby("official_id")["plate_games"].transform(
            lambda values: values.cumsum() - values
        )
    )

    output_path = TABLES_DIR / "career_plate_games.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    career.to_csv(output_path, index=False)

    print(f"Assignment files read: {len(files)}")
    print(f"Umpire-season rows: {len(career):,}")
    print(f"Seasons included: {career['season'].min()}–{career['season'].max()}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()