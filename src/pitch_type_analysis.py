"""Summarize called-pitch accuracy by pitch type and pitch characteristics."""

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_DIR / "data" / "tables" / "pitches_graded" / "all_seasons.parquet"
)
TABLES_DIR = PROJECT_DIR / "reports" / "tables"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

MIN_CALLED_PITCHES = 500
MOVEMENT_BIN_WIDTH_IN = 5
SPEED_BIN_WIDTH_MPH = 5

REQUIRED_COLUMNS = {
    "season",
    "pitch_type_raw",
    "original_call_code",
    "is_call_correct",
    "is_challenged",
    "is_overturned",
    "start_speed_mph",
    "pfx_x_in",
    "pfx_z_in",
    "zone_region",
}


def wilson_interval(successes, trials):
    """Return a 95% Wilson confidence interval for a proportion."""
    if trials == 0:
        return float("nan"), float("nan")

    z = 1.96
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    margin = math.sqrt(
        proportion * (1 - proportion) / trials
        + z**2 / (4 * trials**2)
    ) * z / denominator

    return max(0, centre - margin), min(1, centre + margin)


def add_numeric_bands(frame, source_column, band_column, width):
    """Add left-closed numeric bins with the requested width."""
    values = pd.to_numeric(frame[source_column], errors="coerce")
    observed = values.dropna()

    if observed.empty:
        frame[band_column] = pd.NA
        return

    lower = math.floor(float(observed.min()) / width) * width
    upper = math.floor(float(observed.max()) / width) * width + width
    edges = np.arange(lower, upper + width, width)

    labels = [
        f"{left:g} to <{right:g}"
        for left, right in itertools.pairwise(edges)
    ]

    frame[band_column] = pd.cut(
        values,
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )


def summarize_groups(frame, group_columns):
    """Calculate call accuracy and, for 2026, challenge rates."""
    rows = []

    for key, group in frame.groupby(
        group_columns,
        dropna=True,
        observed=True,
    ):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values))

        called_count = len(group)
        if called_count < MIN_CALLED_PITCHES:
            continue

        correct = group["is_call_correct"].astype(bool)
        incorrect = ~correct
        calls = group["original_call_code"]

        error_count = int(incorrect.sum())
        error_low, error_high = wilson_interval(
            error_count,
            called_count,
        )

        called_strikes = calls.eq("C")
        called_balls = calls.eq("B")
        false_strikes = int((called_strikes & incorrect).sum())
        false_balls = int((called_balls & incorrect).sum())
        called_strike_count = int(called_strikes.sum())
        called_ball_count = int(called_balls.sum())

        false_strike_low, false_strike_high = wilson_interval(
            false_strikes,
            called_strike_count,
        )
        false_ball_low, false_ball_high = wilson_interval(
            false_balls,
            called_ball_count,
        )

        row.update(
            {
                "called_pitches": called_count,
                "incorrect_calls": error_count,
                "error_rate": error_count / called_count,
                "error_ci_low": error_low,
                "error_ci_high": error_high,
                "called_strikes": called_strike_count,
                "false_strikes": false_strikes,
                "false_strike_rate": (
                    false_strikes / called_strike_count
                    if called_strike_count
                    else float("nan")
                ),
                "false_strike_ci_low": false_strike_low,
                "false_strike_ci_high": false_strike_high,
                "called_balls": called_ball_count,
                "false_balls": false_balls,
                "false_ball_rate": (
                    false_balls / called_ball_count
                    if called_ball_count
                    else float("nan")
                ),
                "false_ball_ci_low": false_ball_low,
                "false_ball_ci_high": false_ball_high,
            }
        )

        if int(row["season"]) == 2026:
            challenged = group["is_challenged"].fillna(False).astype(bool)
            challenged_count = int(challenged.sum())
            overturned = (
                challenged
                & group["is_overturned"].fillna(False).astype(bool)
            )
            overturned_count = int(overturned.sum())

            challenge_low, challenge_high = wilson_interval(
                challenged_count,
                called_count,
            )
            overturn_low, overturn_high = wilson_interval(
                overturned_count,
                challenged_count,
            )

            row.update(
                {
                    "challenged_pitches": challenged_count,
                    "challenge_rate": challenged_count / called_count,
                    "challenge_ci_low": challenge_low,
                    "challenge_ci_high": challenge_high,
                    "overturned_challenges": overturned_count,
                    "overturn_rate": (
                        overturned_count / challenged_count
                        if challenged_count
                        else float("nan")
                    ),
                    "overturn_ci_low": overturn_low,
                    "overturn_ci_high": overturn_high,
                }
            )
        else:
            row.update(
                {
                    "challenged_pitches": float("nan"),
                    "challenge_rate": float("nan"),
                    "challenge_ci_low": float("nan"),
                    "challenge_ci_high": float("nan"),
                    "overturned_challenges": float("nan"),
                    "overturn_rate": float("nan"),
                    "overturn_ci_low": float("nan"),
                    "overturn_ci_high": float("nan"),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_error_rate_chart(pitch_type_table):
    """Plot 2026 error rates by pitch type with 95% intervals."""
    chart_data = pitch_type_table.loc[
        pitch_type_table["season"].eq(2026)
    ].copy()

    if chart_data.empty:
        print("No 2026 pitch types meet the 500-pitch chart threshold.")
        return

    chart_data = chart_data.sort_values("error_rate")
    y_positions = np.arange(len(chart_data))
    lower_errors = (
        chart_data["error_rate"] - chart_data["error_ci_low"]
    )
    upper_errors = (
        chart_data["error_ci_high"] - chart_data["error_rate"]
    )

    figure_height = max(5, 0.35 * len(chart_data))
    figure, axis = plt.subplots(figsize=(9, figure_height))

    axis.errorbar(
        chart_data["error_rate"],
        y_positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="o",
        capsize=3,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(chart_data["pitch_type_raw"])
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.set_xlabel("Incorrect calls / called pitches")
    axis.set_title("2026 called-pitch error rate by pitch type")
    axis.grid(axis="x", alpha=0.25)

    figure.text(
        0.01,
        0.01,
        "Bars show 95% Wilson intervals. ST/SV labels were added in 2023; "
        "earlier sliders may include those pitch types.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = FIGURES_DIR / "pitch_types_error_rate_2026.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved chart: {output_path}")


def main():
    """Build J1 tables and the 2026 pitch-type chart."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Graded pitch file not found: {INPUT_PATH}\n"
            "Run src/grade_pitches.py first."
        )

    frame = pd.read_parquet(INPUT_PATH)

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "The graded pitch file is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = frame.loc[
        frame["original_call_code"].isin(["B", "C"])
        & frame["is_call_correct"].notna()
    ].copy()

    add_numeric_bands(
        frame,
        "start_speed_mph",
        "speed_band_mph",
        SPEED_BIN_WIDTH_MPH,
    )
    add_numeric_bands(
        frame,
        "pfx_x_in",
        "horizontal_movement_band_in",
        MOVEMENT_BIN_WIDTH_IN,
    )
    add_numeric_bands(
        frame,
        "pfx_z_in",
        "vertical_movement_band_in",
        MOVEMENT_BIN_WIDTH_IN,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    analyses = {
        "pitch_types_by_type.csv": ["season", "pitch_type_raw"],
        "pitch_types_by_speed.csv": ["season", "speed_band_mph"],
        "pitch_types_by_horizontal_movement.csv": [
            "season",
            "horizontal_movement_band_in",
        ],
        "pitch_types_by_vertical_movement.csv": [
            "season",
            "vertical_movement_band_in",
        ],
        "pitch_types_by_zone_region.csv": ["season", "zone_region"],
    }

    pitch_type_table = None

    for filename, group_columns in analyses.items():
        result = summarize_groups(frame, group_columns)
        output_path = TABLES_DIR / filename
        result.to_csv(output_path, index=False)
        print(f"Saved {len(result):,} rows: {output_path}")

        if filename == "pitch_types_by_type.csv":
            pitch_type_table = result

    save_error_rate_chart(pitch_type_table)

    print(f"Called pitches analyzed: {len(frame):,}")
    print(
        "Note: Statcast added the ST and SV pitch labels in 2023; "
        "pre-2023 sliders may include those pitch types."
    )


if __name__ == "__main__":
    main()