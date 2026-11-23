"""Summarize umpire accuracy by career stage and age."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "reports" / "tables"

RANKINGS_PATH = TABLES_DIR / "umpire_rankings.csv"
CAREER_STAGE_PATH = TABLES_DIR / "umpire_career_stage.csv"


def summarize(
    frame: pd.DataFrame,
    groups: list[str],
    output_name: str,
) -> pd.DataFrame:
    """Summarize umpire accuracy and save a grouped table."""
    summary = (
        frame.groupby(groups, dropna=True)
        .agg(
            umpires=("official_id", "nunique"),
            plate_games=("plate_games", "sum"),
            called_pitches=("called_pitches", "sum"),
            correct_calls=("correct", "sum"),
            mean_umpire_accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )
    summary["pooled_accuracy"] = (
        summary["correct_calls"] / summary["called_pitches"]
    )

    output_path = TABLES_DIR / output_name
    summary.to_csv(output_path, index=False)

    print(f"\nSaved {output_path}")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    """Create accuracy summaries by career stage and age."""
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(f"Rankings table not found: {RANKINGS_PATH}")
    if not CAREER_STAGE_PATH.exists():
        raise FileNotFoundError(
            f"Career-stage table not found: {CAREER_STAGE_PATH}"
        )

    rankings = pd.read_csv(RANKINGS_PATH)
    career = pd.read_csv(CAREER_STAGE_PATH)

    required_rankings = {
        "season",
        "official_id",
        "plate_games",
        "called_pitches",
        "correct",
        "accuracy",
    }
    required_career = {
        "season",
        "official_id",
        "career_stage",
        "age_group",
    }

    missing_rankings = required_rankings - set(rankings.columns)
    missing_career = required_career - set(career.columns)

    if missing_rankings:
        raise ValueError(
            "Rankings table is missing columns: "
            + ", ".join(sorted(missing_rankings))
        )
    if missing_career:
        raise ValueError(
            "Career-stage table is missing columns: "
            + ", ".join(sorted(missing_career))
        )

    career_columns = [
        "season",
        "official_id",
        "career_stage",
        "age_group",
    ]
    joined = rankings.merge(
        career[career_columns],
        on=["season", "official_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(joined["_merge"].ne("both").sum())
    missing_stage = int(joined["career_stage"].isna().sum())
    missing_age = int(joined["age_group"].isna().sum())

    print(f"Ranked umpire-seasons: {len(joined):,}")
    print(f"Ranked rows without a career-stage match: {unmatched:,}")
    print(f"Rows missing career stage: {missing_stage:,}")
    print(f"Rows missing age group: {missing_age:,}")

    if unmatched or missing_stage:
        raise ValueError(
            "Some ranked umpire-seasons did not match the career-stage table."
        )

    joined = joined.drop(columns="_merge")

    summarize(
        joined.dropna(subset=["career_stage"]),
        ["season", "career_stage"],
        "accuracy_by_career_stage.csv",
    )

    summarize(
        joined.dropna(subset=["age_group"]),
        ["season", "age_group"],
        "accuracy_by_age_group.csv",
    )

    # Age and experience overlap poorly in this dataset. This focused table
    # shows career stages among Under 45 umpires and age groups among veterans.
    focused = joined[
        joined["age_group"].eq("Under 45")
        | joined["career_stage"].eq("Veteran (11+)")
    ].dropna(subset=["age_group", "career_stage"])

    summarize(
        focused,
        ["season", "age_group", "career_stage"],
        "accuracy_by_age_and_stage.csv",
    )


if __name__ == "__main__":
    main()