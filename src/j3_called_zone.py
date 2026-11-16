"""Analyze within-umpire accuracy changes and called-zone contours."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PITCHES_PATH = (
    PROJECT_DIR / "data" / "tables" / "pitches_graded" / "all_seasons.parquet"
)
GAMES_PATH = PROJECT_DIR / "data" / "tables" / "games" / "all_seasons.parquet"
TABLES_DIR = PROJECT_DIR / "reports" / "tables"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
REPORTS_DIR = PROJECT_DIR / "reports"

MIN_UMPIRE_CALLS = 1_500
GRID_WIDTH_IN = 1
MIN_GRID_CELL_CALLS = 100
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 703

PITCH_COLUMNS = {
    "season",
    "game_pk",
    "original_call_code",
    "is_call_correct",
    "plate_x_ft",
    "plate_z_ft",
    "sz_top_ft",
    "sz_bot_ft",
}
GAME_COLUMNS = {
    "game_pk",
    "plate_umpire_id",
    "plate_umpire_name",
}


def wilson_interval(successes, trials):
    """Return a 95% Wilson interval for a single proportion."""
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


def load_analysis_data():
    """Load graded calls and attach each game's plate umpire."""
    if not PITCHES_PATH.exists():
        raise FileNotFoundError(
            f"Graded pitch file not found: {PITCHES_PATH}\n"
            "Run src/grade_pitches.py first."
        )
    if not GAMES_PATH.exists():
        raise FileNotFoundError(f"Games table not found: {GAMES_PATH}")

    pitches = pd.read_parquet(
        PITCHES_PATH,
        columns=sorted(PITCH_COLUMNS),
    )
    games = pd.read_parquet(
        GAMES_PATH,
        columns=sorted(GAME_COLUMNS),
    )

    missing_pitch_columns = PITCH_COLUMNS - set(pitches.columns)
    missing_game_columns = GAME_COLUMNS - set(games.columns)
    if missing_pitch_columns or missing_game_columns:
        missing = sorted(missing_pitch_columns | missing_game_columns)
        raise ValueError("Missing required columns: " + ", ".join(missing))

    if games["game_pk"].duplicated().any():
        raise ValueError("The games table has duplicate game_pk values.")

    pitches = pitches.loc[
        pitches["original_call_code"].isin(["B", "C"])
        & pitches["is_call_correct"].notna()
    ].copy()
    pitches["is_call_correct"] = pitches["is_call_correct"].astype(bool)

    return pitches.merge(
        games,
        on="game_pk",
        how="left",
        validate="many_to_one",
    )


def summarize_games(frame):
    """Return correct calls and total calls for each umpire-game."""
    return (
        frame.groupby("game_pk", as_index=False)
        .agg(
            called_pitches=("is_call_correct", "size"),
            correct_calls=("is_call_correct", "sum"),
        )
    )


def bootstrap_difference_interval(prior_games, season_2026_games, rng):
    """Bootstrap the 2026-minus-prior accuracy difference by game."""
    prior_calls = prior_games["called_pitches"].to_numpy(dtype=float)
    prior_correct = prior_games["correct_calls"].to_numpy(dtype=float)
    current_calls = season_2026_games["called_pitches"].to_numpy(
        dtype=float
    )
    current_correct = season_2026_games["correct_calls"].to_numpy(
        dtype=float
    )

    prior_draws = rng.integers(
        0,
        len(prior_calls),
        size=(BOOTSTRAP_REPS, len(prior_calls)),
    )
    current_draws = rng.integers(
        0,
        len(current_calls),
        size=(BOOTSTRAP_REPS, len(current_calls)),
    )

    prior_rates = (
        prior_correct[prior_draws].sum(axis=1)
        / prior_calls[prior_draws].sum(axis=1)
    )
    current_rates = (
        current_correct[current_draws].sum(axis=1)
        / current_calls[current_draws].sum(axis=1)
    )

    differences = current_rates - prior_rates
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(lower), float(upper)


def build_umpire_change_table(frame):
    """Compare eligible umpires' 2021–2025 and 2026 accuracy."""
    prior = frame.loc[frame["season"].between(2021, 2025)].copy()
    current = frame.loc[frame["season"].eq(2026)].copy()

    prior_ids = set(prior["plate_umpire_id"].dropna().unique())
    current_ids = set(current["plate_umpire_id"].dropna().unique())
    common_ids = sorted(prior_ids & current_ids)

    rows = []
    eligible_number = 0

    for umpire_id in common_ids:
        prior_calls = prior.loc[prior["plate_umpire_id"].eq(umpire_id)]
        current_calls = current.loc[
            current["plate_umpire_id"].eq(umpire_id)
        ]

        if (
            len(prior_calls) < MIN_UMPIRE_CALLS
            or len(current_calls) < MIN_UMPIRE_CALLS
        ):
            continue

        eligible_number += 1
        prior_correct = int(prior_calls["is_call_correct"].sum())
        current_correct = int(current_calls["is_call_correct"].sum())
        prior_accuracy = prior_correct / len(prior_calls)
        current_accuracy = current_correct / len(current_calls)
        accuracy_change = current_accuracy - prior_accuracy

        prior_low, prior_high = wilson_interval(
            prior_correct,
            len(prior_calls),
        )
        current_low, current_high = wilson_interval(
            current_correct,
            len(current_calls),
        )

        prior_games = summarize_games(prior_calls)
        current_games = summarize_games(current_calls)

        rng = np.random.default_rng(BOOTSTRAP_SEED + eligible_number)
        change_low, change_high = bootstrap_difference_interval(
            prior_games,
            current_games,
            rng,
        )

        umpire_names = pd.concat(
            [
                prior_calls["plate_umpire_name"],
                current_calls["plate_umpire_name"],
            ]
        ).dropna()
        umpire_name = (
            umpire_names.iloc[0] if not umpire_names.empty else "Unknown"
        )

        rows.append(
            {
                "official_id": umpire_id,
                "official_name": umpire_name,
                "prior_period": "2021-2025",
                "prior_called_pitches": len(prior_calls),
                "prior_correct_calls": prior_correct,
                "prior_accuracy": prior_accuracy,
                "prior_ci_low": prior_low,
                "prior_ci_high": prior_high,
                "prior_plate_games": len(prior_games),
                "season_2026_called_pitches": len(current_calls),
                "season_2026_correct_calls": current_correct,
                "season_2026_accuracy": current_accuracy,
                "season_2026_ci_low": current_low,
                "season_2026_ci_high": current_high,
                "season_2026_plate_games": len(current_games),
                "accuracy_change": accuracy_change,
                "change_ci_low": change_low,
                "change_ci_high": change_high,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "accuracy_change",
            ascending=False,
        ).reset_index(drop=True)

    print(
        "Umpires meeting the 1,500-call minimum in both periods: "
        f"{len(result)}"
    )
    return result


def plot_umpire_change(table):
    """Plot 2026 accuracy against 2021–2025 accuracy."""
    if table.empty:
        print("Skipped umpire scatter plot: no eligible umpires.")
        return

    minimum = min(
        table["prior_accuracy"].min(),
        table["season_2026_accuracy"].min(),
    )
    lower_limit = max(0, min(0.80, minimum - 0.02))

    maximum = max(
        table["prior_accuracy"].max(),
        table["season_2026_accuracy"].max(),
    )
    upper_limit = min(1, max(0.99, maximum + 0.02))

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(
        table["prior_accuracy"],
        table["season_2026_accuracy"],
        alpha=0.8,
    )
    axis.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        color="black",
        linewidth=1,
    )

    axis.set_xlim(lower_limit, upper_limit)
    axis.set_ylim(lower_limit, upper_limit)
    axis.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )
    axis.set_xlabel("Umpire accuracy, 2021–2025")
    axis.set_ylabel("Umpire accuracy, 2026")
    axis.set_title("Within-umpire called-pitch accuracy")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output_path = FIGURES_DIR / "j3_umpire_accuracy_change.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved chart: {output_path}")


def build_called_strike_grid(season_calls):
    """Calculate called-strike shares on a 1-inch absolute-location grid."""
    frame = season_calls.copy()
    frame["x_bin_in"] = np.floor(
        frame["plate_x_ft"] * 12
    ).astype(int)
    frame["z_bin_in"] = np.floor(
        frame["plate_z_ft"] * 12
    ).astype(int)
    frame["is_called_strike"] = frame["original_call_code"].eq("C")

    cells = (
        frame.groupby(
            ["x_bin_in", "z_bin_in"],
            as_index=False,
            observed=True,
        )
        .agg(
            called_pitches=("original_call_code", "size"),
            called_strikes=("is_called_strike", "sum"),
        )
    )

    supported = cells["called_pitches"].ge(MIN_GRID_CELL_CALLS)
    cells["called_strike_rate"] = np.where(
        supported,
        cells["called_strikes"] / cells["called_pitches"],
        np.nan,
    )

    rate = cells["called_strike_rate"]
    count = cells["called_pitches"].astype(float)
    denominator = 1 + 1.96**2 / count
    centre = (rate + 1.96**2 / (2 * count)) / denominator
    margin = 1.96 * np.sqrt(
        rate * (1 - rate) / count
        + 1.96**2 / (4 * count**2)
    ) / denominator

    cells["called_strike_ci_low"] = np.where(
        supported,
        np.maximum(0, centre - margin),
        np.nan,
    )
    cells["called_strike_ci_high"] = np.where(
        supported,
        np.minimum(1, centre + margin),
        np.nan,
    )

    return cells


def estimate_called_zone(season_calls, season):
    """Trace the largest 50% called-strike contour for one season."""
    cells = build_called_strike_grid(season_calls)
    supported = cells.loc[cells["called_strike_rate"].notna()].copy()

    edges = {
        "season": int(season),
        "called_pitches": len(season_calls),
        "grid_cells_used": len(supported),
        "minimum_calls_per_grid_cell": MIN_GRID_CELL_CALLS,
        "called_zone_left_edge_in": float("nan"),
        "called_zone_right_edge_in": float("nan"),
        "called_zone_bottom_edge_in": float("nan"),
        "called_zone_top_edge_in": float("nan"),
        "contour_segments_found": 0,
        "physical_plate_left_edge_in": -8.5,
        "physical_plate_right_edge_in": 8.5,
        "median_recorded_zone_bottom_in": (
            season_calls["sz_bot_ft"].median() * 12
        ),
        "median_recorded_zone_top_in": (
            season_calls["sz_top_ft"].median() * 12
        ),
    }

    if supported.empty:
        cells["season"] = int(season)
        return cells, edges

    x_bins = np.arange(
        supported["x_bin_in"].min(),
        supported["x_bin_in"].max() + GRID_WIDTH_IN,
        GRID_WIDTH_IN,
    )
    z_bins = np.arange(
        supported["z_bin_in"].min(),
        supported["z_bin_in"].max() + GRID_WIDTH_IN,
        GRID_WIDTH_IN,
    )

    matrix = supported.pivot(
        index="z_bin_in",
        columns="x_bin_in",
        values="called_strike_rate",
    ).reindex(index=z_bins, columns=x_bins)

    x_centres = x_bins + GRID_WIDTH_IN / 2
    z_centres = z_bins + GRID_WIDTH_IN / 2
    values = matrix.to_numpy(dtype=float)

    segments = []
    if (
        len(x_centres) >= 2
        and len(z_centres) >= 2
        and np.nanmin(values) <= 0.5 <= np.nanmax(values)
    ):
        figure, axis = plt.subplots()
        contour = axis.contour(
            x_centres,
            z_centres,
            np.ma.masked_invalid(values),
            levels=[0.5],
        )
        segments = [
            np.asarray(segment)
            for segment in contour.allsegs[0]
            if len(segment) >= 2
        ]
        plt.close(figure)

    edges["contour_segments_found"] = len(segments)

    if segments:
        main_contour = max(segments, key=len)
        edges.update(
            {
                "called_zone_left_edge_in": float(
                    main_contour[:, 0].min()
                ),
                "called_zone_right_edge_in": float(
                    main_contour[:, 0].max()
                ),
                "called_zone_bottom_edge_in": float(
                    main_contour[:, 1].min()
                ),
                "called_zone_top_edge_in": float(
                    main_contour[:, 1].max()
                ),
            }
        )

    cells["season"] = int(season)
    return cells, edges


def write_called_zone_notes(edges_table, output_path):
    """Write a short methods note with the 2026 ABS comparison."""
    lines = [
        "# J3 called-zone estimates",
        "",
        (
            "For each season, called pitches were grouped into 1-inch "
            "cells using absolute plate location. The called-strike share "
            "in each cell is called strikes divided by all called pitches "
            "in that cell. Cells with fewer than 100 calls were excluded "
            "from the 50% contour. The reported edges are the horizontal "
            "and vertical extents of the largest 50% contour."
        ),
        "",
        (
            "The 2026 reference rectangle is 17 inches wide "
            "(plate x from -8.5 to 8.5 inches). Its vertical reference "
            "uses the median recorded 2026 zone bottom and top in the "
            "edge table, because ABS vertical limits vary by batter."
        ),
        "",
        (
            "The 2026 called-zone estimates are descriptive. MLB location "
            "coordinates are measured at the front of the plate through "
            "2025 and at the plate midpoint in 2026. The 2026 ABS zone is "
            "also defined differently from earlier recorded zones."
        ),
        "",
        (
            "The 50% called-strike contour describes the pattern of "
            "umpire calls; it is not itself a measure of call correctness."
        ),
        "",
    ]

    row_2026 = edges_table.loc[edges_table["season"].eq(2026)]
    if not row_2026.empty:
        row = row_2026.iloc[0]
        if pd.notna(row["called_zone_left_edge_in"]):
            lines.extend(
                [
                    (
                        "2026 estimated contour: "
                        f"x = {row['called_zone_left_edge_in']:.1f} to "
                        f"{row['called_zone_right_edge_in']:.1f} inches; "
                        f"z = {row['called_zone_bottom_edge_in']:.1f} to "
                        f"{row['called_zone_top_edge_in']:.1f} inches."
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "A 2026 50% contour was not found with the selected "
                        "1-inch grid and 100-call cell minimum."
                    ),
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved notes: {output_path}")


def main():
    """Build J3 umpire comparisons and called-zone estimates."""
    joined = load_analysis_data()

    called = joined.loc[
        joined["plate_umpire_id"].notna()
        & joined["plate_x_ft"].notna()
        & joined["plate_z_ft"].notna()
        & joined["sz_top_ft"].notna()
        & joined["sz_bot_ft"].notna()
    ].copy()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    umpire_table = build_umpire_change_table(called)
    umpire_table_path = TABLES_DIR / "j3_within_umpire_accuracy.csv"
    umpire_table.to_csv(umpire_table_path, index=False)
    print(f"Saved table: {umpire_table_path}")
    plot_umpire_change(umpire_table)

    grid_tables = []
    edge_rows = []

    for season in sorted(called["season"].dropna().unique()):
        season_calls = called.loc[called["season"].eq(season)]
        cells, edges = estimate_called_zone(season_calls, season)
        grid_tables.append(cells)
        edge_rows.append(edges)

    grid_table = pd.concat(grid_tables, ignore_index=True)
    edges_table = pd.DataFrame(edge_rows)

    grid_path = TABLES_DIR / "j3_called_strike_grid_1in.csv"
    edges_path = TABLES_DIR / "j3_called_zone_edges.csv"
    notes_path = REPORTS_DIR / "j3_called_zone_notes.md"

    grid_table.to_csv(grid_path, index=False)
    edges_table.to_csv(edges_path, index=False)

    print(f"Saved grid table: {grid_path}")
    print(f"Saved edge table: {edges_path}")
    print("\nCalled-zone edges by season:")
    print(edges_table.to_string(index=False))

    write_called_zone_notes(edges_table, notes_path)


if __name__ == "__main__":
    main()