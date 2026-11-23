"""Build umpire career-stage data for the 2021–2026 analysis."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
CAREER_PATH = PROJECT_DIR / "data" / "tables" / "career_plate_games.csv"
RETROSHEET_BIO_PATH = PROJECT_DIR / "data" / "raw" / "retrosheet" / "biofile0.csv"
GAMES_PATH = PROJECT_DIR / "data" / "tables" / "games" / "all_seasons.parquet"
GAMES_DIR = PROJECT_DIR / "data" / "tables" / "games"
OUTPUT_PATH = PROJECT_DIR / "reports" / "tables" / "umpire_career_stage.csv"

ANALYSIS_SEASONS = range(2021, 2027)

# Operational proxy for identifying an umpire's first full-time season.
MIN_GAMES_WORKED = 100
MIN_PLATE_GAMES = 25


def normalize_name(value: object) -> str:
    """Normalize a name for matching across MLB and Retrosheet files."""
    if pd.isna(value):
        return ""

    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def retrosheet_date(values: pd.Series) -> pd.Series:
    """Parse Retrosheet dates, which are commonly stored as YYYYMMDD."""
    cleaned = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def load_season_start_dates() -> pd.Series:
    """Return the earliest recorded regular-season game date for each season."""
    if GAMES_PATH.exists():
        games = pd.read_parquet(GAMES_PATH, columns=["season", "game_date"])
    else:
        paths = sorted(GAMES_DIR.glob("season=*/part.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No games data found at {GAMES_PATH} or in {GAMES_DIR}"
            )
        games = pd.concat(
            [
                pd.read_parquet(path, columns=["season", "game_date"])
                for path in paths
            ],
            ignore_index=True,
        )

    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games.dropna(subset=["season", "game_date"]).copy()
    games["season"] = games["season"].astype(int)

    return games.groupby("season")["game_date"].min()


def load_retrosheet_bios(official_names: pd.Series) -> pd.DataFrame:
    """Match project umpire names to Retrosheet umpire biographies."""
    if not RETROSHEET_BIO_PATH.exists():
        raise FileNotFoundError(f"Retrosheet biography file not found: {RETROSHEET_BIO_PATH}")

    bios = pd.read_csv(
        RETROSHEET_BIO_PATH,
        dtype={"birthdate": "string", "debut_u": "string"},
    )

    required = {"lastname", "usename", "birthdate", "debut_u"}
    missing_columns = required - set(bios.columns)
    if missing_columns:
        raise ValueError(
            "Retrosheet biography file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    bios["name_key"] = (
        bios["usename"].fillna("").astype(str)
        + " "
        + bios["lastname"].fillna("").astype(str)
    ).map(normalize_name)

    project_keys = set(official_names.dropna().map(normalize_name))
    bios = bios[bios["name_key"].isin(project_keys)].copy()

    # Keep only one biography row per matching key. If Retrosheet has more than
    # one distinct person for a project name, leave that name unmatched.
    bio_rows: list[dict[str, object]] = []
    for name_key, group in bios.groupby("name_key", sort=False):
        distinct_people = group.drop_duplicates(
            subset=["birthdate", "debut_u"]
        )

        if len(distinct_people) != 1:
            continue

        row = distinct_people.iloc[0]
        bio_rows.append(
            {
                "name_key": name_key,
                "birthdate": row["birthdate"],
                "debut_u": row["debut_u"],
            }
        )

    result = pd.DataFrame(
        bio_rows,
        columns=["name_key", "birthdate", "debut_u"],
    )
    result["birth_date"] = retrosheet_date(result["birthdate"])
    result["debut_date"] = retrosheet_date(result["debut_u"])
    return result[["name_key", "birth_date", "debut_date"]]


def assign_experience_stage(full_time_season_number: object) -> str:
    """Assign a career stage without using age."""
    if pd.isna(full_time_season_number) or full_time_season_number < 1:
        return "Call-up"
    if full_time_season_number <= 3:
        return "Early (1-3)"
    if full_time_season_number <= 10:
        return "Experienced (4-10)"
    return "Veteran (11+)"


def assign_age_group(age: object) -> str | pd._libs.missing.NAType:
    """Group age separately from career experience."""
    if pd.isna(age):
        return pd.NA
    if age < 45:
        return "Under 45"
    if age < 55:
        return "45-54"
    return "55+"


def build_table() -> pd.DataFrame:
    """Build and save the umpire-season career-stage table."""
    if not CAREER_PATH.exists():
        raise FileNotFoundError(f"Career archive not found: {CAREER_PATH}")

    career = pd.read_csv(CAREER_PATH)
    required = {
        "official_id",
        "season",
        "official_name",
        "games_worked",
        "plate_games",
    }
    missing_columns = required - set(career.columns)
    if missing_columns:
        raise ValueError(
            "Career archive is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in ["official_id", "season", "games_worked", "plate_games"]:
        career[column] = pd.to_numeric(career[column], errors="coerce")

    career = career.dropna(
        subset=["official_id", "season", "games_worked", "plate_games"]
    ).copy()
    career["official_id"] = career["official_id"].astype("int64")
    career["season"] = career["season"].astype(int)
    career["games_worked"] = career["games_worked"].astype(int)
    career["plate_games"] = career["plate_games"].astype(int)

    # Find the first season meeting the stated full-time proxy.
    full_time_rows = career[
        career["games_worked"].ge(MIN_GAMES_WORKED)
        & career["plate_games"].ge(MIN_PLATE_GAMES)
    ]
    first_full_season = (
        full_time_rows.groupby("official_id")["season"]
        .min()
        .rename("first_full_season")
    )

    analysis = career[career["season"].isin(ANALYSIS_SEASONS)].copy()
    analysis = analysis.merge(
        first_full_season,
        on="official_id",
        how="left",
        validate="many_to_one",
    )

    analysis["full_time_season_number"] = (
        analysis["season"] - analysis["first_full_season"] + 1
    ).astype("Int64")

    analysis["career_stage"] = analysis["full_time_season_number"].map(
        assign_experience_stage
    )

    # Mark the first full season in the archive as left-censored only when
    # Retrosheet shows the umpire debuted before the archive starts.
    official_names = career[["official_id", "official_name"]].drop_duplicates(
        "official_id"
    )
    official_names["name_key"] = official_names["official_name"].map(
        normalize_name
    )

    bios = load_retrosheet_bios(official_names["official_name"])
    official_names = official_names.merge(
        bios,
        on="name_key",
        how="left",
        validate="many_to_one",
    )

    analysis = analysis.merge(
        official_names[
            ["official_id", "birth_date", "debut_date"]
        ],
        on="official_id",
        how="left",
        validate="many_to_one",
    )

    season_starts = load_season_start_dates()
    analysis["season_start_date"] = analysis["season"].map(season_starts)

    analysis["age_at_season_start"] = pd.Series(
        pd.NA, index=analysis.index, dtype="Int64"
    )
    has_age_dates = (
        analysis["birth_date"].notna()
        & analysis["season_start_date"].notna()
    )
    birthday_not_reached = (
        analysis.loc[has_age_dates, "season_start_date"].dt.month
        .lt(analysis.loc[has_age_dates, "birth_date"].dt.month)
        | (
            analysis.loc[has_age_dates, "season_start_date"].dt.month.eq(
                analysis.loc[has_age_dates, "birth_date"].dt.month
            )
            & analysis.loc[has_age_dates, "season_start_date"].dt.day.lt(
                analysis.loc[has_age_dates, "birth_date"].dt.day
            )
        )
    )
    age = (
        analysis.loc[has_age_dates, "season_start_date"].dt.year
        - analysis.loc[has_age_dates, "birth_date"].dt.year
        - birthday_not_reached.astype(int)
    )
    analysis.loc[has_age_dates, "age_at_season_start"] = age.astype("Int64")

    analysis["age_group"] = analysis["age_at_season_start"].map(assign_age_group)

    analysis["years_since_mlb_debut"] = (
        analysis["season"] - analysis["debut_date"].dt.year
    ).astype("Int64")

    analysis["rule_note"] = (
        f"Full-time status uses proxy: at least {MIN_GAMES_WORKED} games "
        f"worked and {MIN_PLATE_GAMES} plate games in a season. "
        "Age group is reported separately from career stage."
    )

    # Nullable first_full_season values must be checked with pd.notna before
    # comparing them; otherwise pandas raises “boolean value of NA is ambiguous.”
    left_censored = (
        analysis["first_full_season"].notna()
        & analysis["first_full_season"].eq(2000)
        & analysis["debut_date"].notna()
        & analysis["debut_date"].dt.year.lt(2000)
    )
    analysis.loc[left_censored, "rule_note"] += (
        " First full-time season may be left-censored because the career "
        "archive begins in 2000."
    )

    missing_bio = analysis["birth_date"].isna() | analysis["debut_date"].isna()
    analysis.loc[missing_bio, "rule_note"] += (
        " Retrosheet birth or umpire debut date is unavailable; related "
        "age or debut fields are left missing."
    )

    columns = [
        "season",
        "official_id",
        "official_name",
        "age_at_season_start",
        "age_group",
        "debut_date",
        "years_since_mlb_debut",
        "games_worked",
        "plate_games",
        "first_full_season",
        "full_time_season_number",
        "career_stage",
        "rule_note",
    ]
    result = analysis[columns].sort_values(
        ["season", "official_name"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(result):,}")
    print(f"Umpires represented: {result['official_id'].nunique():,}")
    print(
        "Rows missing age:",
        int(result["age_at_season_start"].isna().sum()),
    )
    print(
        "Rows missing debut date:",
        int(result["debut_date"].isna().sum()),
    )
    print("\nCareer-stage counts by season:")
    print(
        result.groupby(["season", "career_stage"], observed=False)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    return result


if __name__ == "__main__":
    build_table()