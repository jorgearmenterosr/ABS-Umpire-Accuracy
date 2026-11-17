"""Build per-umpire-season data and correlation matrices for J5."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

RANKINGS_PATH = PROJECT_DIR / "reports" / "tables" / "umpire_rankings.csv"
CHALLENGES_PATH = (
    PROJECT_DIR / "reports" / "tables" / "umpire_challenges_2026.csv"
)
UMPIRES_PATH = PROJECT_DIR / "data" / "tables" / "umpires" / "part.parquet"
ASSIGNMENTS_PATH = (
    PROJECT_DIR
    / "data"
    / "tables"
    / "umpire_assignments"
    / "all_seasons.parquet"
)
CREWS_PATH = PROJECT_DIR / "data" / "tables" / "crews" / "all_seasons.parquet"

PANEL_PATH = PROJECT_DIR / "reports" / "tables" / "j5_umpire_season.csv"
CORRELATION_PATH = (
    PROJECT_DIR / "reports" / "tables" / "j5_correlation_matrix.csv"
)
SAMPLE_SIZE_PATH = (
    PROJECT_DIR / "reports" / "tables" / "j5_correlation_sample_sizes.csv"
)
NOTES_PATH = PROJECT_DIR / "reports" / "j5_correlation_notes.md"

CORRELATION_VARIABLES = [
    "accuracy",
    "called_pitches",
    "plate_games",
    "games_worked",
    "distinct_crew_count",
    "age_at_season_start",
    "years_since_mlb_debut",
    "challenge_rate_2026",
    "overturn_rate_2026",
]

Z_95 = 1.959963984540054


def read_inputs() -> tuple[pd.DataFrame, ...]:
    """Read the O2/O5 files and required project tables."""
    rankings = pd.read_csv(RANKINGS_PATH)
    challenges = pd.read_csv(CHALLENGES_PATH)
    umpires = pd.read_parquet(UMPIRES_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    crews = pd.read_parquet(CREWS_PATH)

    return rankings, challenges, umpires, assignments, crews


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert season and official ID to consistent integer keys."""
    result = frame.copy()
    result["season"] = pd.to_numeric(result["season"], errors="coerce")
    result["official_id"] = pd.to_numeric(
        result["official_id"], errors="coerce"
    )
    result = result.dropna(subset=["season", "official_id"]).copy()
    result["season"] = result["season"].astype(int)
    result["official_id"] = result["official_id"].astype(int)
    return result


def completed_years_at_season_start(
    season: int,
    event_date: pd.Timestamp,
    *,
    require_event_to_have_occurred: bool,
) -> float:
    """Return completed years as of January 1 of the season."""
    if pd.isna(event_date):
        return np.nan

    start_date = pd.Timestamp(int(season), 1, 1)
    if require_event_to_have_occurred and event_date > start_date:
        return np.nan

    years = start_date.year - event_date.year
    if (event_date.month, event_date.day) > (start_date.month, start_date.day):
        years -= 1
    return float(years)


def summarize_games_worked(assignments: pd.DataFrame) -> pd.DataFrame:
    """Count distinct regular-season games per umpire-season."""
    assignments = normalize_keys(assignments)
    assignments = assignments.drop_duplicates(
        subset=["season", "official_id", "game_pk"]
    )

    return assignments.groupby(
        ["season", "official_id"], as_index=False
    ).agg(games_worked=("game_pk", "nunique"))


def summarize_crews(crews: pd.DataFrame) -> pd.DataFrame:
    """Summarize crew IDs and distinct crews per umpire-season."""
    crews = normalize_keys(crews)
    rows = []

    for (season, official_id), group in crews.groupby(
        ["season", "official_id"], sort=True
    ):
        crew_ids = sorted(
            group["crew_id"].dropna().astype(str).unique().tolist()
        )
        rows.append(
            {
                "season": int(season),
                "official_id": int(official_id),
                "crew_ids": "; ".join(crew_ids),
                "distinct_crew_count": len(crew_ids),
            }
        )

    return pd.DataFrame(rows)


def summarize_challenges(challenges: pd.DataFrame) -> pd.DataFrame:
    """Prepare 2026 challenge and overturn measures from O5."""
    required = {
        "official_id",
        "challenges",
        "overturned",
        "overturn_rate",
        "ci_low",
        "ci_high",
    }
    missing = required - set(challenges.columns)
    if missing:
        raise ValueError(
            f"O5 file is missing required columns: {sorted(missing)}"
        )

    challenges = challenges.copy()
    challenges["official_id"] = pd.to_numeric(
        challenges["official_id"], errors="coerce"
    )
    challenges = challenges.dropna(subset=["official_id"]).copy()
    challenges["official_id"] = challenges["official_id"].astype(int)

    for column in [
        "challenges",
        "overturned",
        "overturn_rate",
        "ci_low",
        "ci_high",
    ]:
        challenges[column] = pd.to_numeric(
            challenges[column], errors="coerce"
        )

    if challenges["official_id"].duplicated().any():
        raise ValueError("O5 file has more than one row per official_id.")

    challenges = challenges.rename(
        columns={
            "challenges": "challenges_2026",
            "overturned": "overturned_2026",
            "overturn_rate": "overturn_rate_2026",
            "ci_low": "overturn_ci_low_2026",
            "ci_high": "overturn_ci_high_2026",
        }
    )

    return challenges[
        [
            "official_id",
            "challenges_2026",
            "overturned_2026",
            "overturn_rate_2026",
            "overturn_ci_low_2026",
            "overturn_ci_high_2026",
        ]
    ]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a proportion."""
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2 * total)) / denominator
    margin = (
        Z_95
        * np.sqrt(
            proportion * (1 - proportion) / total
            + Z_95**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def add_challenge_measures(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate 2026 challenge rate and its Wilson interval."""
    panel = panel.copy()
    panel["challenge_rate_denominator_2026"] = np.where(
        panel["season"].eq(2026),
        panel["called_pitches"],
        np.nan,
    )
    panel["challenge_rate_2026"] = (
        panel["challenges_2026"] / panel["challenge_rate_denominator_2026"]
    )

    intervals = [
        wilson_interval(int(successes), int(total))
        if pd.notna(successes) and pd.notna(total) and total > 0
        else (np.nan, np.nan)
        for successes, total in zip(
            panel["challenges_2026"],
            panel["challenge_rate_denominator_2026"],
        )
    ]
    panel["challenge_ci_low_2026"] = [interval[0] for interval in intervals]
    panel["challenge_ci_high_2026"] = [interval[1] for interval in intervals]
    panel["overturn_rate_denominator_2026"] = panel["challenges_2026"]
    return panel


def build_panel(
    rankings: pd.DataFrame,
    challenges: pd.DataFrame,
    umpires: pd.DataFrame,
    assignments: pd.DataFrame,
    crews: pd.DataFrame,
) -> pd.DataFrame:
    """Join inputs into one row per ranked umpire-season."""
    rankings = normalize_keys(rankings)
    required_rankings = {
        "official_name",
        "called_pitches",
        "correct",
        "accuracy",
        "ci_low",
        "ci_high",
        "plate_games",
    }
    missing = required_rankings - set(rankings.columns)
    if missing:
        raise ValueError(
            f"O2 file is missing required columns: {sorted(missing)}"
        )

    if rankings.duplicated(["season", "official_id"]).any():
        raise ValueError("O2 file has duplicate season and official_id rows.")

    numeric_columns = [
        "called_pitches",
        "correct",
        "accuracy",
        "ci_low",
        "ci_high",
        "plate_games",
    ]
    for column in numeric_columns:
        rankings[column] = pd.to_numeric(
            rankings[column], errors="coerce"
        )

    umpires = umpires.copy()
    umpires["official_id"] = pd.to_numeric(
        umpires["official_id"], errors="coerce"
    )
    umpires = umpires.dropna(subset=["official_id"]).copy()
    umpires["official_id"] = umpires["official_id"].astype(int)
    umpires["birth_date"] = pd.to_datetime(
        umpires["birth_date"], errors="coerce"
    )
    umpires["mlb_debut_date"] = pd.to_datetime(
        umpires["mlb_debut_date"], errors="coerce"
    )
    umpires = umpires.drop_duplicates("official_id")

    panel = rankings.merge(
        summarize_games_worked(assignments),
        on=["season", "official_id"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        summarize_crews(crews),
        on=["season", "official_id"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        umpires[["official_id", "birth_date", "mlb_debut_date"]],
        on="official_id",
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        summarize_challenges(challenges),
        on="official_id",
        how="left",
        validate="many_to_one",
    )

    panel["age_at_season_start"] = [
        completed_years_at_season_start(
            season,
            birth_date,
            require_event_to_have_occurred=False,
        )
        for season, birth_date in zip(panel["season"], panel["birth_date"])
    ]
    panel["years_since_mlb_debut"] = [
        completed_years_at_season_start(
            season,
            debut_date,
            require_event_to_have_occurred=True,
        )
        for season, debut_date in zip(panel["season"], panel["mlb_debut_date"])
    ]

    non_2026 = panel["season"].ne(2026)
    challenge_columns = [
        "challenges_2026",
        "overturned_2026",
        "overturn_rate_2026",
        "overturn_ci_low_2026",
        "overturn_ci_high_2026",
    ]
    panel.loc[non_2026, challenge_columns] = np.nan
    return add_challenge_measures(panel)


def make_correlation_matrices(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise Pearson correlations and sample sizes."""
    correlations = pd.DataFrame(
        index=CORRELATION_VARIABLES,
        columns=CORRELATION_VARIABLES,
        dtype=float,
    )
    sample_sizes = pd.DataFrame(
        index=CORRELATION_VARIABLES,
        columns=CORRELATION_VARIABLES,
        dtype="Int64",
    )

    for variable_x in CORRELATION_VARIABLES:
        for variable_y in CORRELATION_VARIABLES:
            paired = pd.concat(
                [
                    panel[variable_x].rename("x"),
                    panel[variable_y].rename("y"),
                ],
                axis=1,
            ).dropna()

            n = len(paired)
            sample_sizes.loc[variable_x, variable_y] = n

            if (
                n >= 2
                and paired["x"].nunique() > 1
                and paired["y"].nunique() > 1
            ):
                correlations.loc[variable_x, variable_y] = paired["x"].corr(
                    paired["y"], method="pearson"
                )

    correlations.index.name = "variable"
    correlations.columns.name = "variable"
    sample_sizes.index.name = "variable"
    sample_sizes.columns.name = "variable"
    return correlations, sample_sizes


def write_notes(panel: pd.DataFrame) -> None:
    """Write interpretation and method notes for J5."""
    notes = [
        "# J5: Umpire-season correlation analysis",
        "",
        f"Rows in the analysis: {len(panel):,} umpire-seasons.",
        (
            "The input O2 rankings apply its minimum of 10 plate games per "
            "umpire-season."
        ),
        (
            "The matrix contains pairwise Pearson correlations. Each cell's "
            "sample size is saved in j5_correlation_sample_sizes.csv."
        ),
        (
            "Age and MLB experience are completed years as of January 1 of "
            "each season. Experience is missing when the umpire's MLB debut "
            "occurred after that season began."
        ),
        (
            "Crew IDs are descriptive labels. The correlation matrix uses "
            "the number of distinct crews worked during the season."
        ),
        (
            "Challenge rate is challenges divided by called pitches; its "
            "95% interval is Wilson. Overturn rate is overturns divided by "
            "challenges; its 95% interval comes from O5."
        ),
        (
            "Challenge and overturn measures are included only for 2026. "
            "Missing O5 matches remain missing."
        ),
        (
            "These are descriptive associations, not causal effects. The same "
            "umpire can appear in multiple seasons, so rows are not independent."
        ),
        (
            "Age, experience, and calendar time move together. The 2025-to-2026 "
            "accuracy change also mixes umpire behavior with MLB's change in "
            "how it measures the strike zone."
        ),
    ]

    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    rankings, challenges, umpires, assignments, crews = read_inputs()
    panel = build_panel(
        rankings,
        challenges,
        umpires,
        assignments,
        crews,
    )
    correlations, sample_sizes = make_correlation_matrices(panel)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH)
    sample_sizes.to_csv(SAMPLE_SIZE_PATH)
    write_notes(panel)

    print(f"Umpire-seasons: {len(panel):,}")
    print(f"Saved panel: {PANEL_PATH}")
    print(f"Saved correlation matrix: {CORRELATION_PATH}")
    print(f"Saved sample sizes: {SAMPLE_SIZE_PATH}")
    print(f"Saved notes: {NOTES_PATH}")
    print()
    print("Pearson correlation matrix:")
    print(correlations.to_string())
    print()
    print("Pairwise sample sizes:")
    print(sample_sizes.to_string())


if __name__ == "__main__":
    main()