"""Build the umpire master table and career-stage/tier summaries."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "reports" / "tables"

TIERS_PATH = TABLES_DIR / "umpire_tiers.csv"
CAREER_STAGE_PATH = TABLES_DIR / "umpire_career_stage.csv"

Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Calculate a 95% Wilson interval for a pooled accuracy proportion."""
    if trials <= 0:
        return float("nan"), float("nan")

    proportion = successes / trials
    z_squared = Z_95**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        Z_95
        * (
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        ** 0.5
        / denominator
    )
    return center - margin, center + margin


def save_table(table: pd.DataFrame, filename: str) -> None:
    """Save a report table as CSV."""
    path = TABLES_DIR / filename
    table.to_csv(path, index=False)
    print(f"Saved {path.relative_to(PROJECT_DIR)} ({len(table):,} rows)")


def main() -> None:
    """Build the combined master table and O11 summary tables."""
    for path in [TIERS_PATH, CAREER_STAGE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required table not found: {path}")

    tiers = pd.read_csv(TIERS_PATH)
    career = pd.read_csv(CAREER_STAGE_PATH)

    keys = ["season", "official_id"]
    if tiers.duplicated(keys).any():
        raise ValueError("Tier table has duplicate season/official_id rows.")
    if career.duplicated(keys).any():
        raise ValueError("Career-stage table has duplicate season/official_id rows.")

    master = tiers.merge(
        career,
        on=keys,
        how="left",
        suffixes=("", "_career"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int(master["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} tier rows did not match the career-stage table."
        )
    master = master.drop(columns="_merge")

    missing_tier_for_qualified = (
        master["qualifies"].fillna(False) & master["tier_name"].isna()
    )
    if missing_tier_for_qualified.any():
        raise ValueError("Some qualified umpire-seasons have no tier name.")

    master["performance_tier"] = master["tier_name"].fillna("Not qualified")
    save_table(master, "umpire_master.csv")

    # Accuracy comparisons use the tier script's qualifying umpire-seasons.
    qualifying = master[master["qualifies"].fillna(False)].copy()

    rows = []
    for (season, stage), group in qualifying.groupby(
        ["season", "career_stage"], sort=True
    ):
        called = int(group["called_pitches"].sum())
        correct = int(group["correct"].sum())
        low, high = wilson_interval(correct, called)

        rows.append(
            {
                "season": int(season),
                "career_stage": stage,
                "umpires": int(group["official_id"].nunique()),
                "called_pitches": called,
                "correct_calls": correct,
                "pooled_accuracy": correct / called if called else float("nan"),
                "ci_low": low,
                "ci_high": high,
            }
        )

    accuracy_by_stage = pd.DataFrame(rows)
    save_table(accuracy_by_stage, "accuracy_by_stage.csv")

    stage_by_tier = (
        master.groupby(
            ["season", "career_stage", "performance_tier"],
            dropna=False,
            sort=True,
        )
        .agg(
            umpire_seasons=("official_id", "size"),
            umpires=("official_id", "nunique"),
            called_pitches=("called_pitches", "sum"),
        )
        .reset_index()
    )
    save_table(stage_by_tier, "stage_by_tier.csv")

    print(
        "\nAccuracy intervals are pooled pitch-level Wilson intervals. "
        "They do not adjust for clustering pitches within umpires."
    )


if __name__ == "__main__":
    main()