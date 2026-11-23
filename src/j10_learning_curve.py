"""Analyze monthly accuracy and the 2026 ABS feedback learning curve."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_helpers import wilson_interval


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_TABLES = PROJECT_DIR / "data" / "tables"
REPORT_TABLES = PROJECT_DIR / "reports" / "tables"
REPORT_FIGURES = PROJECT_DIR / "reports" / "figures"

PITCHES_PATH = DATA_TABLES / "pitches_graded" / "all_seasons.parquet"
GAMES_PATH = DATA_TABLES / "games" / "all_seasons.parquet"
GAME_ACCURACY_PATH = REPORT_TABLES / "umpire_game_accuracy.csv"

MONTHS = range(4, 10)  # April through September
SEASONS = range(2021, 2027)


def make_monthly_accuracy() -> pd.DataFrame:
    """Calculate monthly accuracy and Wilson intervals by season."""
    pitches = pd.read_parquet(
        PITCHES_PATH,
        columns=[
            "season",
            "game_date",
            "original_call_code",
            "is_call_correct",
        ],
    )

    pitches["game_date"] = pd.to_datetime(pitches["game_date"], errors="coerce")
    pitches["month"] = pitches["game_date"].dt.month
    pitches = pitches[
        pitches["season"].isin(SEASONS)
        & pitches["month"].isin(MONTHS)
        & pitches["original_call_code"].isin(["B", "C"])
        & pitches["is_call_correct"].notna()
    ].copy()

    pitches["is_call_correct"] = pitches["is_call_correct"].astype(bool)
    monthly = (
        pitches.groupby(["season", "month"])
        .agg(
            called_pitches=("is_call_correct", "size"),
            correct_calls=("is_call_correct", "sum"),
        )
        .reset_index()
    )

    monthly["accuracy"] = (
        monthly["correct_calls"] / monthly["called_pitches"]
    )
    intervals = monthly.apply(
        lambda row: wilson_interval(
            int(row["correct_calls"]),
            int(row["called_pitches"]),
        ),
        axis=1,
    )
    monthly["ci_low"] = [interval[0] for interval in intervals]
    monthly["ci_high"] = [interval[1] for interval in intervals]
    monthly["zone"] = np.where(monthly["season"].eq(2026), "new", "old")

    return monthly.sort_values(["season", "month"]).reset_index(drop=True)


def save_monthly_table(monthly: pd.DataFrame) -> None:
    """Save monthly accuracy and create the learning-curve chart."""
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_TABLES / "accuracy_by_month.csv"
    monthly.to_csv(output_path, index=False)
    print(f"Saved {output_path.relative_to(PROJECT_DIR)} ({len(monthly)} rows)")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    historical = monthly[monthly["season"].between(2021, 2025)]
    for season, group in historical.groupby("season"):
        ax.plot(
            group["month"],
            group["accuracy"] * 100,
            marker="o",
            linewidth=1,
            alpha=0.35,
            label=str(season),
        )

    historical_mean = (
        historical.groupby("month")["accuracy"].mean().sort_index()
    )
    ax.plot(
        historical_mean.index,
        historical_mean.values * 100,
        color="black",
        linestyle="--",
        marker="o",
        linewidth=2,
        label="2021–2025 average",
    )

    season_2026 = monthly[monthly["season"].eq(2026)].sort_values("month")
    if not season_2026.empty:
        y = season_2026["accuracy"].to_numpy() * 100
        yerr = np.vstack(
            [
                (season_2026["accuracy"] - season_2026["ci_low"]).to_numpy()
                * 100,
                (season_2026["ci_high"] - season_2026["accuracy"]).to_numpy()
                * 100,
            ]
        )
        ax.errorbar(
            season_2026["month"],
            y,
            yerr=yerr,
            color="#c62828",
            marker="o",
            linewidth=2.5,
            capsize=3,
            label="2026 (new zone)",
        )

    month_labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    ax.set_xticks(list(MONTHS), month_labels)
    ax.set_ylabel("Called-pitch accuracy (%)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Umpire Accuracy: 2021–2026")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()

    figure_path = REPORT_FIGURES / "learning_curve_2026.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"Saved {figure_path.relative_to(PROJECT_DIR)}")

    slopes = []
    for season, group in monthly.groupby("season"):
        if len(group) >= 2:
            slope = np.polyfit(
                group["month"].to_numpy(),
                group["accuracy"].to_numpy() * 100,
                1,
            )[0]
            slopes.append({"season": int(season), "slope_pp_per_month": slope})

    slope_table = pd.DataFrame(slopes)
    old_slope = slope_table.loc[
        slope_table["season"].between(2021, 2025), "slope_pp_per_month"
    ].mean()
    new_slope = slope_table.loc[
        slope_table["season"].eq(2026), "slope_pp_per_month"
    ]

    print("\nMonthly accuracy slopes (percentage points per month):")
    print(slope_table.to_string(index=False))
    if not new_slope.empty:
        print(f"2021–2025 average slope: {old_slope:.3f}")
        print(f"2026 slope: {new_slope.iloc[0]:.3f}")


def build_feedback_observations() -> pd.DataFrame:
    """Build first-15 versus next-15 game observations for 2025 and 2026."""
    required_paths = [GAMES_PATH, GAME_ACCURACY_PATH, PITCHES_PATH]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    games = pd.read_parquet(
        GAMES_PATH,
        columns=["game_pk", "game_date"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    game_accuracy = pd.read_csv(GAME_ACCURACY_PATH)
    game_accuracy = game_accuracy[
        game_accuracy["season"].isin([2025, 2026])
    ].copy()
    game_accuracy = game_accuracy.merge(
        games,
        on="game_pk",
        how="left",
        validate="many_to_one",
    )
    game_accuracy = game_accuracy.dropna(subset=["game_date"]).copy()
    game_accuracy["called"] = pd.to_numeric(
        game_accuracy["called"], errors="coerce"
    )
    game_accuracy["correct"] = pd.to_numeric(
        game_accuracy["correct"], errors="coerce"
    )
    game_accuracy = game_accuracy.dropna(
        subset=["called", "correct"]
    ).copy()
    game_accuracy["misses"] = (
        game_accuracy["called"] - game_accuracy["correct"]
    )

    overturned = pd.read_parquet(
        PITCHES_PATH,
        columns=["season", "game_pk", "is_overturned"],
    )
    overturned = overturned[overturned["season"].eq(2026)].copy()
    overturned["is_overturned"] = (
        overturned["is_overturned"].fillna(False).astype(bool)
    )
    overturns_by_game = (
        overturned.groupby("game_pk")["is_overturned"]
        .sum()
        .rename("overturns")
    )
    game_accuracy = game_accuracy.merge(
        overturns_by_game,
        on="game_pk",
        how="left",
        validate="many_to_one",
    )
    game_accuracy["overturns"] = game_accuracy["overturns"].fillna(0)

    observations = []
    for season in [2025, 2026]:
        season_games = game_accuracy[
            game_accuracy["season"].eq(season)
        ].copy()

        for (official_id, official_name), group in season_games.groupby(
            ["official_id", "official_name"], dropna=False
        ):
            group = group.sort_values(["game_date", "game_pk"])
            if len(group) < 30:
                continue

            first = group.iloc[:15]
            second = group.iloc[15:30]

            first_called = int(first["called"].sum())
            second_called = int(second["called"].sum())
            if first_called == 0 or second_called == 0:
                continue

            first_accuracy = first["correct"].sum() / first_called
            second_accuracy = second["correct"].sum() / second_called

            if season == 2026:
                dose_name = "overturns_first_15"
                dose = int(first["overturns"].sum())
            else:
                dose_name = "placebo_misses_first_15"
                dose = int(first["misses"].sum())

            observations.append(
                {
                    "season": season,
                    "official_id": official_id,
                    "official_name": official_name,
                    "first_15_games": len(first),
                    "second_15_games": len(second),
                    "first_half_called": first_called,
                    "second_half_called": second_called,
                    "first_half_accuracy": first_accuracy,
                    "second_half_accuracy": second_accuracy,
                    "accuracy_change_pp": (
                        second_accuracy - first_accuracy
                    )
                    * 100,
                    "dose_definition": dose_name,
                    "dose_first_15": dose,
                }
            )

    return pd.DataFrame(observations)


def fit_hc3_model(frame: pd.DataFrame, label: str) -> dict[str, object]:
    """Regress accuracy change on early dose, controlling for early accuracy."""
    if len(frame) < 4:
        raise ValueError(f"Not enough umpire observations for {label}.")

    dose = frame["dose_first_15"].to_numpy(dtype=float)
    first_accuracy_pp = (
        frame["first_half_accuracy"].to_numpy(dtype=float) * 100
    )
    outcome = frame["accuracy_change_pp"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(frame)), dose, first_accuracy_pp]
    )

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(f"Regression predictors are collinear for {label}.")

    beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ beta

    bread = np.linalg.inv(design.T @ design)
    leverage = np.sum((design @ bread) * design, axis=1)
    adjusted = residuals / (1 - leverage)
    weighted_design = design * adjusted[:, None]
    meat = weighted_design.T @ weighted_design
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))

    dose_se = standard_errors[1]
    dose_coefficient = beta[1]
    if dose_se > 0:
        z_score = dose_coefficient / dose_se
        p_value = math.erfc(abs(z_score) / math.sqrt(2))
    else:
        p_value = float("nan")

    return {
        "analysis": label,
        "umpires": len(frame),
        "dose_definition": frame["dose_definition"].iloc[0],
        "dose_coefficient_pp": dose_coefficient,
        "hc3_standard_error": dose_se,
        "ci_low": dose_coefficient - 1.96 * dose_se,
        "ci_high": dose_coefficient + 1.96 * dose_se,
        "normal_approx_p_value": p_value,
        "first_half_accuracy_coefficient": beta[2],
    }


def save_feedback_tables(observations: pd.DataFrame) -> None:
    """Save umpire-level observations and feedback-dose model results."""
    if observations.empty:
        raise ValueError("No umpire has 30 qualifying games in 2025 or 2026.")

    REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    observations_path = REPORT_TABLES / "feedback_dose_response.csv"
    observations.to_csv(observations_path, index=False)
    print(
        f"Saved {observations_path.relative_to(PROJECT_DIR)} "
        f"({len(observations):,} rows)"
    )

    models = []
    for season, label in [
        (2026, "2026 ABS feedback"),
        (2025, "2025 placebo"),
    ]:
        sample = observations[observations["season"].eq(season)]
        if sample.empty:
            print(f"No eligible observations for {label}.")
            continue
        models.append(fit_hc3_model(sample, label))

    model_table = pd.DataFrame(models)
    model_path = REPORT_TABLES / "feedback_dose_response_model.csv"
    model_table.to_csv(model_path, index=False)
    print(f"Saved {model_path.relative_to(PROJECT_DIR)}")
    print(model_table.to_string(index=False))


def main() -> None:
    """Run the monthly learning-curve and feedback-dose analyses."""
    monthly = make_monthly_accuracy()
    save_monthly_table(monthly)

    observations = build_feedback_observations()
    print("\nEligible umpire observations:")
    print(observations.groupby("season")["official_id"].nunique().to_string())
    save_feedback_tables(observations)


if __name__ == "__main__":
    main()