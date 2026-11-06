"""Build season crew assignments and umpire rotation summaries."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
REPORTS_DIR = PROJECT_DIR / "reports"

WORKED_STATUSES = {"Final", "Completed Early"}
EXPECTED_POSITIONS = {
    "Home Plate",
    "First Base",
    "Second Base",
    "Third Base",
}
HISTOGRAM_BIN_WIDTH = 5

GAME_COLUMNS = {
    "game_pk",
    "season",
    "game_date",
    "game_number",
    "status",
    "source_file",
}
ASSIGNMENT_COLUMNS = {
    "game_pk",
    "season",
    "official_id",
    "official_name",
    "official_type",
    "source_file",
}
CREW_COLUMNS = [
    "game_pk",
    "season",
    "crew_id",
    "crew_size",
    "is_complete_crew",
    "game_date",
    "game_number",
    "official_id",
    "official_name",
    "official_type",
    "previous_game_pk",
    "previous_official_type",
    "is_position_change",
    "source_file",
]


def load_season_tables(
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one season's games and umpire assignment tables."""
    games_path = TABLES_DIR / "games" / f"season={season}" / "part.parquet"
    assignments_path = (
        TABLES_DIR / "umpire_assignments" / f"season={season}" / "part.parquet"
    )

    if not games_path.exists():
        raise FileNotFoundError(f"Games table not found: {games_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Umpire assignments table not found: {assignments_path}"
        )

    games = pd.read_parquet(games_path)
    assignments = pd.read_parquet(assignments_path)

    missing_game_columns = GAME_COLUMNS - set(games.columns)
    missing_assignment_columns = ASSIGNMENT_COLUMNS - set(assignments.columns)

    if missing_game_columns:
        raise ValueError(
            f"Games table is missing: {sorted(missing_game_columns)}"
        )
    if missing_assignment_columns:
        raise ValueError(
            "Umpire assignments table is missing: "
            f"{sorted(missing_assignment_columns)}"
        )

    assert games["game_pk"].is_unique, "game_pk is not unique in games"
    assert not assignments.duplicated(["game_pk", "official_id"]).any(), (
        "game_pk and official_id are not unique"
    )

    if not (games["season"] == season).all():
        raise ValueError(f"Games table contains seasons other than {season}")
    if not (assignments["season"] == season).all():
        raise ValueError(
            f"Assignments table contains seasons other than {season}"
        )

    return games, assignments


def build_crew_table(
    games: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per worked game and assigned umpire."""
    worked_games = games[games["status"].isin(WORKED_STATUSES)].copy()

    if worked_games.empty:
        raise ValueError("There are no worked games in this season.")

    worked_game_ids = set(worked_games["game_pk"])
    assignment_game_ids = set(assignments["game_pk"])
    missing_game_ids = worked_game_ids - assignment_game_ids

    if missing_game_ids:
        raise ValueError(
            "Worked games have no umpire assignments: "
            f"{sorted(missing_game_ids)}"
        )

    game_details = worked_games[
        ["game_pk", "season", "game_date", "game_number"]
    ]
    crew_rows = assignments.merge(
        game_details,
        on=["game_pk", "season"],
        how="inner",
        validate="many_to_one",
    )

    if crew_rows["official_id"].isna().any():
        raise ValueError(
            "A worked game has an assignment with no official_id."
        )
    if crew_rows["official_type"].isna().any():
        raise ValueError(
            "A worked game has an assignment with no official_type."
        )
    if crew_rows["game_date"].isna().any():
        raise ValueError("A worked game is missing its game_date.")

    members_by_game = {}
    crew_size_by_game = {}
    completeness_by_game = {}

    for game_pk, group in crew_rows.groupby("game_pk", sort=True):
        official_ids = group["official_id"].astype(int).tolist()
        positions = group["official_type"].astype(str).tolist()

        if len(official_ids) != len(set(official_ids)):
            raise ValueError(f"Game {game_pk} has a repeated official_id.")
        if len(positions) != len(set(positions)):
            raise ValueError(f"Game {game_pk} has a repeated umpire position.")

        unexpected_positions = set(positions) - EXPECTED_POSITIONS
        if unexpected_positions:
            raise ValueError(
                f"Game {game_pk} has unexpected positions: "
                f"{sorted(unexpected_positions)}"
            )
        if "Home Plate" not in positions:
            raise ValueError(
                f"Worked game {game_pk} has no Home Plate umpire."
            )

        members_by_game[game_pk] = tuple(sorted(official_ids))
        crew_size_by_game[game_pk] = len(official_ids)
        completeness_by_game[game_pk] = set(positions) == EXPECTED_POSITIONS

    unique_crews = sorted(set(members_by_game.values()))
    crew_id_by_members = {
        members: crew_number
        for crew_number, members in enumerate(unique_crews, start=1)
    }
    crew_id_by_game = {
        game_pk: crew_id_by_members[members]
        for game_pk, members in members_by_game.items()
    }

    crew_rows["crew_id"] = crew_rows["game_pk"].map(crew_id_by_game)
    crew_rows["crew_size"] = crew_rows["game_pk"].map(crew_size_by_game)
    crew_rows["is_complete_crew"] = crew_rows["game_pk"].map(
        completeness_by_game
    )

    crew_rows = crew_rows.sort_values(
        ["official_id", "game_date", "game_number", "game_pk"],
        na_position="last",
    ).reset_index(drop=True)

    official_groups = crew_rows.groupby(
        ["season", "official_id"],
        sort=False,
    )
    crew_rows["previous_game_pk"] = official_groups["game_pk"].shift(1)
    crew_rows["previous_official_type"] = official_groups[
        "official_type"
    ].shift(1)

    has_previous_game = crew_rows["previous_official_type"].notna()
    crew_rows["is_position_change"] = pd.Series(
        pd.NA,
        index=crew_rows.index,
        dtype="boolean",
    )
    crew_rows.loc[has_previous_game, "is_position_change"] = (
        crew_rows.loc[has_previous_game, "official_type"]
        != crew_rows.loc[has_previous_game, "previous_official_type"]
    )

    integer_columns = [
        "game_pk",
        "season",
        "crew_id",
        "crew_size",
        "game_number",
        "official_id",
        "previous_game_pk",
    ]
    text_columns = [
        "game_date",
        "official_name",
        "official_type",
        "previous_official_type",
        "source_file",
    ]

    for column in integer_columns:
        crew_rows[column] = pd.to_numeric(
            crew_rows[column],
            errors="coerce",
        ).astype("Int64")

    for column in text_columns:
        crew_rows[column] = crew_rows[column].astype("string")

    crew_rows["is_complete_crew"] = crew_rows["is_complete_crew"].astype(bool)
    crew_rows["is_position_change"] = crew_rows["is_position_change"].astype(
        "boolean"
    )

    crew_rows = crew_rows[CREW_COLUMNS]

    assert not crew_rows.duplicated(["game_pk", "official_id"]).any(), (
        "game_pk and official_id are not unique in crews"
    )

    return crew_rows


def build_umpire_summary(crews: pd.DataFrame) -> pd.DataFrame:
    """Count games and Home Plate assignments for each umpire."""
    crews = crews.copy()
    crews["is_plate_game"] = crews["official_type"] == "Home Plate"

    summary = (
        crews.groupby(
            ["season", "official_id"],
            as_index=False,
        )
        .agg(
            official_name=("official_name", "first"),
            games_worked=("game_pk", "nunique"),
            plate_games=("is_plate_game", "sum"),
        )
        .sort_values(["season", "official_id"])
        .reset_index(drop=True)
    )

    summary["plate_game_share"] = (
        summary["plate_games"] / summary["games_worked"]
    ).round(4)
    summary["source_file"] = (
        "data/tables/umpire_assignments/"
        f"season={int(crews['season'].iloc[0])}/part.parquet"
    )

    for column in ["season", "official_id", "games_worked", "plate_games"]:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        ).astype("Int64")

    summary["official_name"] = summary["official_name"].astype("string")
    summary["source_file"] = summary["source_file"].astype("string")

    assert not summary.duplicated(["season", "official_id"]).any(), (
        "season and official_id are not unique in the summary"
    )

    return summary


def histogram_text(summary: pd.DataFrame, season: int) -> str:
    """Create a text histogram of plate games per umpire."""
    maximum = int(summary["plate_games"].max())
    lines = [
        f"# Plate games per umpire: {season}",
        "",
        "Each row groups umpires by their number of Home Plate games.",
        "",
        "| Plate games | Umpires | Histogram |",
        "| --- | ---: | --- |",
    ]

    for lower in range(0, maximum + 1, HISTOGRAM_BIN_WIDTH):
        upper = lower + HISTOGRAM_BIN_WIDTH - 1
        count = int(summary["plate_games"].between(lower, upper).sum())
        bar = "#" * count
        lines.append(f"| {lower}–{upper} | {count} | {bar} |")

    return "\n".join(lines) + "\n"


def save_table(table: pd.DataFrame, season: int) -> Path:
    """Save the crews table as Parquet and CSV."""
    output_dir = TABLES_DIR / "crews" / f"season={season}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "part.parquet"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(output_dir / "part.csv", index=False)

    return parquet_path


def save_reports(
    summary: pd.DataFrame,
    crews: pd.DataFrame,
    season: int,
) -> tuple[Path, Path]:
    """Save the umpire summary and plate-game histogram reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / f"umpire_season_summary_{season}.csv"
    summary.to_csv(summary_path, index=False)

    histogram_path = REPORTS_DIR / f"plate_games_histogram_{season}.md"
    histogram_path.write_text(
        histogram_text(summary, season),
        encoding="utf-8",
    )

    incomplete = crews.loc[
        ~crews["is_complete_crew"],
        ["game_pk", "game_date", "crew_id", "crew_size"],
    ].drop_duplicates()

    print(f"Games with incomplete crews: {len(incomplete)}")
    for row in incomplete.itertuples(index=False):
        assigned_positions = set(
            crews.loc[
                crews["game_pk"] == row.game_pk,
                "official_type",
            ]
        )
        missing_positions = sorted(EXPECTED_POSITIONS - assigned_positions)
        print(
            f"  game {row.game_pk}: {row.crew_size} umpires; "
            f"missing {', '.join(missing_positions)}"
        )

    return summary_path, histogram_path


def main() -> None:
    """Build crew assignments and reports for one season."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/build_crews.py <season>")

    season = int(sys.argv[1])
    games, assignments = load_season_tables(season)
    crews = build_crew_table(games, assignments)
    summary = build_umpire_summary(crews)

    print(f"Season: {season}")
    print(f"Worked games: {crews['game_pk'].nunique()}")
    print(f"Unique crews: {crews['crew_id'].nunique()}")
    print(f"Umpires in summary: {len(summary)}")
    print(
        "Average assignments per worked game: "
        f"{len(crews) / crews['game_pk'].nunique():.2f}"
    )

    print("Saved crews table:", save_table(crews, season))
    summary_path, histogram_path = save_reports(
        summary,
        crews,
        season,
    )
    print("Saved umpire summary:", summary_path)
    print("Saved plate-games histogram:", histogram_path)


if __name__ == "__main__":
    main()
