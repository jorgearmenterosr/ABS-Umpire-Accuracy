"""Build a global umpire table from assignments and Retrosheet bio data."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
BIOFILE_PATH = PROJECT_DIR / "data" / "raw" / "retrosheet" / "biofile0.csv"
BIOFILE_RELATIVE_PATH = "data/raw/retrosheet/biofile0.csv"
NO_UMPIRE_NAME = "NO UMPIRE"
WORKED_STATUSES = {"Final", "Completed Early"}
VALID_SEASONS = set(range(2021, 2027))

OUTPUT_COLUMNS = [
    "official_id",
    "official_name",
    "seasons_active",
    "first_game_date",
    "last_game_date",
    "retrosheet_id",
    "birth_date",
    "mlb_debut_date",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    """Parse the seasons to include."""
    parser = argparse.ArgumentParser(
        description="Build the global umpire table."
    )
    parser.add_argument(
        "seasons",
        nargs="+",
        type=int,
        help="Seasons to include, such as 2021 2022 2023 2024 2025 2026.",
    )
    args = parser.parse_args()

    if len(args.seasons) != len(set(args.seasons)):
        parser.error("Do not list a season more than once.")

    invalid_seasons = set(args.seasons) - VALID_SEASONS
    if invalid_seasons:
        parser.error(
            f"Seasons must be from 2021 through 2026: "
            f"{sorted(invalid_seasons)}"
        )

    return args


def clean_name(value: object) -> str | None:
    """Normalize capitalization and spacing for exact name matching."""
    if pd.isna(value):
        return None
    return " ".join(str(value).split()).casefold()


def make_used_name(
    first_name: object,
    last_name: object,
) -> str | None:
    """Build a Retrosheet career-use name."""
    if pd.isna(first_name) or pd.isna(last_name):
        return None
    return clean_name(f"{first_name} {last_name}")


def parse_retrosheet_date(value: object) -> str | None:
    """Convert a Retrosheet date to YYYY-MM-DD."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    for date_format in ("%Y%m%d", "%Y-%m-%d"):
        parsed = pd.to_datetime(
            text,
            format=date_format,
            errors="coerce",
        )
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")

    return None


def load_worked_assignments(seasons: list[int]) -> pd.DataFrame:
    """Load assignments for games that were played."""
    season_tables = []

    for season in seasons:
        assignment_path = (
            PROJECT_DIR
            / "data"
            / "tables"
            / "umpire_assignments"
            / f"season={season}"
            / "part.parquet"
        )
        games_path = (
            PROJECT_DIR
            / "data"
            / "tables"
            / "games"
            / f"season={season}"
            / "part.parquet"
        )

        if not assignment_path.exists():
            raise FileNotFoundError(assignment_path)
        if not games_path.exists():
            raise FileNotFoundError(games_path)

        assignments = pd.read_parquet(assignment_path)
        games = pd.read_parquet(games_path)

        worked_games = games.loc[
            games["status"].isin(WORKED_STATUSES),
            ["game_pk", "season", "game_date"],
        ]

        season_assignments = assignments.merge(
            worked_games,
            on=["game_pk", "season"],
            how="inner",
            validate="many_to_one",
        )
        season_tables.append(season_assignments)

    worked = pd.concat(season_tables, ignore_index=True)

    # The schedule API uses this literal name for an unfilled umpire role.
    placeholder = (
        worked["official_name"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq(NO_UMPIRE_NAME.casefold())
    )
    worked = worked.loc[~placeholder].copy()

    duplicate_key = worked.duplicated(subset=["game_pk", "official_id"])
    if duplicate_key.any():
        raise ValueError("Duplicate (game_pk, official_id) assignments found.")

    worked["game_date"] = pd.to_datetime(
        worked["game_date"],
        errors="coerce",
    )
    return worked


def load_retrosheet_umpires() -> dict[str, dict[str, dict[str, object]]]:
    """Load Retrosheet umpires and index their exact name forms."""
    if not BIOFILE_PATH.exists():
        raise FileNotFoundError(BIOFILE_PATH)

    biofile = pd.read_csv(BIOFILE_PATH, dtype="string")
    required_columns = {
        "id",
        "fullname",
        "usename",
        "lastname",
        "birthdate",
        "debut_u",
    }
    missing_columns = required_columns - set(biofile.columns)
    if missing_columns:
        raise ValueError(
            f"Retrosheet biofile is missing columns: {sorted(missing_columns)}"
        )

    biofile = biofile.loc[biofile["debut_u"].notna()].copy()

    if biofile["id"].duplicated().any():
        raise ValueError("Retrosheet umpire IDs are not unique.")

    records_by_name: dict[str, dict[str, dict[str, object]]] = {}

    for row in biofile.itertuples(index=False):
        retrosheet_id = str(row.id)
        record = {
            "retrosheet_id": retrosheet_id,
            "birth_date": parse_retrosheet_date(row.birthdate),
            "mlb_debut_date": parse_retrosheet_date(row.debut_u),
        }
        name_keys = {
            clean_name(row.fullname),
            make_used_name(row.usename, row.lastname),
        }

        for name_key in name_keys:
            if name_key is None:
                continue
            records_by_name.setdefault(name_key, {})[retrosheet_id] = record

    return records_by_name


def most_common_name(names: pd.Series) -> str:
    """Choose the most frequently listed MLB official name."""
    counts = names.dropna().astype("string").value_counts()
    if counts.empty:
        return ""

    highest_count = counts.max()
    top_names = sorted(counts[counts.eq(highest_count)].index)
    return str(top_names[0])


def build_umpire_table(
    worked: pd.DataFrame,
    records_by_name: dict[str, dict[str, dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Build umpire rows and report names without a unique match."""
    rows = []
    match_issues = []
    matched_ids = set()

    for official_id, group in worked.groupby("official_id", sort=True):
        official_id = int(official_id)
        active_seasons = sorted(
            {int(season) for season in group["season"].dropna()}
        )
        official_names = sorted(
            set(group["official_name"].dropna().astype(str))
        )
        official_name = most_common_name(group["official_name"])

        candidate_records = {}
        for name in official_names:
            name_key = clean_name(name)
            if name_key is not None:
                candidate_records.update(records_by_name.get(name_key, {}))

        retrosheet_record = None
        match_status = "unmatched"

        if len(candidate_records) == 1:
            retrosheet_record = next(iter(candidate_records.values()))
            matched_ids.add(official_id)
        elif len(candidate_records) > 1:
            match_status = "ambiguous"

        assignment_sources = [
            (f"data/tables/umpire_assignments/season={season}/part.parquet")
            for season in active_seasons
        ]
        source_files = assignment_sources + [BIOFILE_RELATIVE_PATH]

        rows.append(
            {
                "official_id": official_id,
                "official_name": official_name,
                "seasons_active": ", ".join(
                    str(season) for season in active_seasons
                ),
                "first_game_date": group["game_date"].min(),
                "last_game_date": group["game_date"].max(),
                "retrosheet_id": (
                    retrosheet_record["retrosheet_id"]
                    if retrosheet_record
                    else pd.NA
                ),
                "birth_date": (
                    retrosheet_record["birth_date"]
                    if retrosheet_record
                    else pd.NA
                ),
                "mlb_debut_date": (
                    retrosheet_record["mlb_debut_date"]
                    if retrosheet_record
                    else pd.NA
                ),
                "source_file": "; ".join(source_files),
            }
        )

        if match_status == "ambiguous" or not candidate_records:
            match_issues.append(
                {
                    "official_id": official_id,
                    "official_name": official_name,
                    "match_status": match_status,
                }
            )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result["official_id"] = pd.to_numeric(
        result["official_id"],
        errors="raise",
    ).astype("Int64")

    text_columns = [
        "official_name",
        "seasons_active",
        "retrosheet_id",
        "birth_date",
        "mlb_debut_date",
        "source_file",
    ]
    for column in text_columns:
        result[column] = result[column].astype("string")

    for column in ["first_game_date", "last_game_date"]:
        result[column] = (
            pd.to_datetime(result[column], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .astype("string")
        )

    issues = pd.DataFrame(
        match_issues,
        columns=["official_id", "official_name", "match_status"],
    )
    if not issues.empty:
        issues["official_id"] = issues["official_id"].astype("Int64")
        issues["official_name"] = issues["official_name"].astype("string")
        issues["match_status"] = issues["match_status"].astype("string")

    if result["official_id"].isna().any():
        raise ValueError("The umpire table has a missing official_id.")
    if result["official_id"].duplicated().any():
        raise ValueError("The umpire table has duplicate official_id.")

    plate_ids = {
        int(value)
        for value in worked.loc[
            worked["official_type"].eq("Home Plate"),
            "official_id",
        ].dropna()
    }

    return result, issues, plate_ids & matched_ids


def save_outputs(
    table: pd.DataFrame,
    issues: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Save the global umpire table and name-match issues."""
    table_directory = PROJECT_DIR / "data" / "tables" / "umpires"
    reports_directory = PROJECT_DIR / "reports"
    table_directory.mkdir(parents=True, exist_ok=True)
    reports_directory.mkdir(parents=True, exist_ok=True)

    parquet_path = table_directory / "part.parquet"
    csv_path = table_directory / "part.csv"
    issues_path = reports_directory / "umpire_match_issues.csv"

    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)
    issues.to_csv(issues_path, index=False)

    return parquet_path, csv_path, issues_path


def main() -> None:
    """Build and save the global umpire table."""
    args = parse_args()
    seasons = sorted(args.seasons)

    worked = load_worked_assignments(seasons)
    records_by_name = load_retrosheet_umpires()
    table, issues, matched_plate_ids = build_umpire_table(
        worked,
        records_by_name,
    )
    parquet_path, csv_path, issues_path = save_outputs(table, issues)

    plate_ids = (
        worked.loc[
            worked["official_type"].eq("Home Plate"),
            "official_id",
        ]
        .dropna()
        .astype(int)
        .nunique()
    )
    match_rate = len(matched_plate_ids) / plate_ids if plate_ids else 0.0

    print(f"Seasons included: {', '.join(map(str, seasons))}")
    print(f"Umpires in table: {len(table)}")
    print(
        f"Plate umpires matched: {len(matched_plate_ids)}/{plate_ids} "
        f"({match_rate:.1%})"
    )
    print(f"Names without a unique match: {len(issues)}")
    for row in issues.itertuples(index=False):
        print(f"  {row.official_id}: {row.official_name} ({row.match_status})")
    print(f"Saved Parquet table: {parquet_path}")
    print(f"Saved CSV table: {csv_path}")
    print(f"Saved name-match report: {issues_path}")


if __name__ == "__main__":
    main()
