"""Combine per-season tables into one file per table.

Reads data/tables/<table>/season=*/part.parquet and writes
data/tables/<table>/all_seasons.parquet (plus a CSV for the small tables).

Usage:  python src/combine_seasons.py
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
TABLES = [
    "games",
    "umpire_assignments",
    "crews",
    "pitches",
    "challenges",
    "challenges_remaining",
]
CSV_ROW_LIMIT = 2_000_000  # above this, CSV is too large to be useful


def season_files(table: str) -> list[Path]:
    """Return the per-season Parquet files of a table, oldest first."""
    return sorted((TABLES_DIR / table).glob("season=*/part.parquet"))


def combine(table: str) -> pd.DataFrame | None:
    """Concatenate every season of a table; None if the table is absent."""
    files = season_files(table)
    if not files:
        return None
    pairs = [(f, pd.read_parquet(f)) for f in files]
    pairs = [(p, fr) for p, fr in pairs if len(fr)]
    if not pairs:
        return None
    files = [p for p, _ in pairs]
    frames = [fr for _, fr in pairs]
    for path, frame in zip(files, frames):
        print(
            f"    {path.parent.name}: {len(frame):>8,} rows, "
            f"{frame.shape[1]} cols, game_pk={frame['game_pk'].dtype}"
        )
    combined = pd.concat(frames, ignore_index=True)
    per_season = sum(len(f) for f in frames)
    assert len(combined) == per_season, f"{table}: row count changed"
    return combined


def main() -> None:
    for table in TABLES:
        combined = combine(table)
        if combined is None:
            print(f"{table:22s} (no seasons on disk)")
            continue
        out = TABLES_DIR / table / "all_seasons.parquet"
        combined.to_parquet(out, index=False)
        if len(combined) <= CSV_ROW_LIMIT:
            combined.to_csv(out.with_suffix(".csv"), index=False)
        seasons = sorted(combined["season"].dropna().unique().tolist())
        print(f"{table:22s} rows={len(combined):>9,}  seasons={seasons}")


if __name__ == "__main__":
    main()
