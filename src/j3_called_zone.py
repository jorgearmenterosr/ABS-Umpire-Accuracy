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
BALL_RADIUS_IN = 1.45

PITCH_COLUMNS = {
    "season",
    "game_pk",
    "game_date",
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


def trace_called_zone_contour(
    cells,
    season,
    called_pitches,
    median_zone_bottom_in,
    median_zone_top_in,
):
    """Trace the largest 50% contour from a called-strike grid."""
    supported = cells.loc[cells["called_strike_rate"].notna()].copy()

    edges = {
        "season": int(season),
        "called_pitches": int(called_pitches),
        "grid_cells_used": len(supported),
        "minimum_calls_per_grid_cell": MIN_GRID_CELL_CALLS,
        "called_zone_left_edge_in": float("nan"),
        "called_zone_right_edge_in": float("nan"),
        "called_zone_bottom_edge_in": float("nan"),
        "called_zone_top_edge_in": float("nan"),
        "contour_segments_found": 0,
        "physical_plate_left_edge_in": -8.5,
        "physical_plate_right_edge_in": 8.5,
        "median_recorded_zone_bottom_in": median_zone_bottom_in,
        "median_recorded_zone_top_in": median_zone_top_in,
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


def estimate_called_zone(season_calls, season):
    """Trace the largest 50% called-strike contour for one season."""
    cells = build_called_strike_grid(season_calls)
    return trace_called_zone_contour(
        cells,
        season,
        len(season_calls),
        season_calls["sz_bot_ft"].median() * 12,
        season_calls["sz_top_ft"].median() * 12,
    )


def official_limits_in(season_calls):
    """Return legal ball-centre limits in inches for a season's calls."""
    return {
        "official_left_limit_in": -8.5 - BALL_RADIUS_IN,
        "official_right_limit_in": 8.5 + BALL_RADIUS_IN,
        "official_bottom_limit_in": (
            season_calls["sz_bot_ft"].median() * 12 - BALL_RADIUS_IN
        ),
        "official_top_limit_in": (
            season_calls["sz_top_ft"].median() * 12 + BALL_RADIUS_IN
        ),
    }


def edge_comparison(edges, season_calls, limits=None):
    """Compare contour edges with the official limits, including ball radius."""
    if limits is None:
        limits = official_limits_in(season_calls)
    called_edges = {
        "called_zone_left_edge_in": "left",
        "called_zone_right_edge_in": "right",
        "called_zone_bottom_edge_in": "bottom",
        "called_zone_top_edge_in": "top",
    }
    comparisons = {
        "left": limits["official_left_limit_in"]
        - edges["called_zone_left_edge_in"],
        "right": edges["called_zone_right_edge_in"]
        - limits["official_right_limit_in"],
        "bottom": limits["official_bottom_limit_in"]
        - edges["called_zone_bottom_edge_in"],
        "top": edges["called_zone_top_edge_in"]
        - limits["official_top_limit_in"],
    }
    row = {
        "season": int(edges["season"]),
        "called_pitches": int(edges["called_pitches"]),
        "games": int(season_calls["game_pk"].nunique()),
        **limits,
    }
    for edge_name, side in called_edges.items():
        row[edge_name] = edges[edge_name]
        row[f"called_{side}_past_official_limit_in"] = comparisons[side]
    return row


def bootstrap_zone_edge_intervals(season_calls, season):
    """Get percentile CIs for called-zone edges by resampling whole games."""
    frame = season_calls[
        [
            "game_pk",
            "plate_x_ft",
            "plate_z_ft",
            "original_call_code",
        ]
    ].copy()
    frame["x_bin_in"] = np.floor(frame["plate_x_ft"] * 12).astype(int)
    frame["z_bin_in"] = np.floor(frame["plate_z_ft"] * 12).astype(int)
    frame["is_called_strike"] = frame["original_call_code"].eq("C")
    game_cells = (
        frame.groupby(
            ["game_pk", "x_bin_in", "z_bin_in"],
            as_index=False,
            observed=True,
        )
        .agg(
            called_pitches=("original_call_code", "size"),
            called_strikes=("is_called_strike", "sum"),
        )
    )
    game_ids = np.unique(game_cells["game_pk"].to_numpy())
    cell_keys = game_cells[["x_bin_in", "z_bin_in"]].drop_duplicates()
    cell_keys = cell_keys.reset_index(drop=True)
    game_cells = game_cells.merge(
        cell_keys.assign(_cell_code=np.arange(len(cell_keys))),
        on=["x_bin_in", "z_bin_in"],
        how="left",
        validate="many_to_one",
    )
    cell_codes = game_cells["_cell_code"].to_numpy(dtype=int)
    call_counts = game_cells["called_pitches"].to_numpy(dtype=float)
    strike_counts = game_cells["called_strikes"].to_numpy(dtype=float)
    game_codes = game_cells["game_pk"].map(
        {game_pk: index for index, game_pk in enumerate(game_ids)}
    ).to_numpy(dtype=int)

    if len(game_ids) < 2:
        return {}, 0

    rng = np.random.default_rng(BOOTSTRAP_SEED + int(season))
    edge_names = [
        "called_zone_left_edge_in",
        "called_zone_right_edge_in",
        "called_zone_bottom_edge_in",
        "called_zone_top_edge_in",
    ]
    overreach_names = [
        "called_left_past_official_limit_in",
        "called_right_past_official_limit_in",
        "called_bottom_past_official_limit_in",
        "called_top_past_official_limit_in",
    ]
    samples = {name: [] for name in edge_names + overreach_names}

    limits = official_limits_in(season_calls)
    for replicate in range(BOOTSTRAP_REPS):
        draws = rng.integers(0, len(game_ids), size=len(game_ids))
        game_multipliers = np.bincount(draws, minlength=len(game_ids))
        weights = game_multipliers[game_codes]
        sampled_calls = np.bincount(
            cell_codes,
            weights=weights * call_counts,
            minlength=len(cell_keys),
        )
        sampled_strikes = np.bincount(
            cell_codes,
            weights=weights * strike_counts,
            minlength=len(cell_keys),
        )
        sampled_rates = np.divide(
            sampled_strikes,
            sampled_calls,
            out=np.full(len(cell_keys), np.nan),
            where=sampled_calls >= MIN_GRID_CELL_CALLS,
        )
        sampled_grid = cell_keys.copy()
        sampled_grid["called_pitches"] = sampled_calls.astype(int)
        sampled_grid["called_strikes"] = sampled_strikes.astype(int)
        sampled_grid["called_strike_rate"] = sampled_rates
        _, sampled_edges = trace_called_zone_contour(
            sampled_grid,
            season,
            int(sampled_calls.sum()),
            season_calls["sz_bot_ft"].median() * 12,
            season_calls["sz_top_ft"].median() * 12,
        )
        if pd.isna(sampled_edges["called_zone_left_edge_in"]):
            continue

        comparison = edge_comparison(sampled_edges, season_calls, limits)
        for name in samples:
            samples[name].append(comparison[name])

        if (replicate + 1) % 250 == 0:
            print(
                f"  season {season} zone bootstrap: "
                f"{replicate + 1:,}/{BOOTSTRAP_REPS:,}"
            )

    intervals = {}
    for name, values in samples.items():
        if values:
            lower, upper = np.quantile(values, [0.025, 0.975])
        else:
            lower, upper = float("nan"), float("nan")
        intervals[f"{name}_ci_low"] = float(lower)
        intervals[f"{name}_ci_high"] = float(upper)

    return intervals, len(samples[edge_names[0]])


def build_called_vs_official_zone(called):
    """Build seasonal contour edges, official limits, and game-bootstrap CIs."""
    rows = []
    for season in sorted(called["season"].dropna().unique()):
        season_calls = called.loc[called["season"].eq(season)].copy()
        _, edges = estimate_called_zone(season_calls, season)
        row = edge_comparison(edges, season_calls)
        intervals, valid_reps = bootstrap_zone_edge_intervals(
            season_calls,
            season,
        )
        row.update(intervals)
        row["bootstrap_valid_reps"] = valid_reps
        rows.append(row)

    return pd.DataFrame(rows)


def build_2026_monthly_called_top(called):
    """Estimate the monthly 2026 top from a central plate cross-section."""
    center_half_width_in = 3.0
    minimum_bin_calls = 100

    season_calls = called.loc[called["season"].eq(2026)].copy()
    season_calls["game_date"] = pd.to_datetime(
        season_calls["game_date"],
        errors="coerce",
    )
    season_calls = season_calls.dropna(subset=["game_date"])
    season_calls["month"] = season_calls["game_date"].dt.to_period("M").astype(
        str
    )
    season_calls["plate_x_in"] = season_calls["plate_x_ft"] * 12
    season_calls["z_bin_in"] = np.floor(
        season_calls["plate_z_ft"] * 12
    ).astype(int)
    season_calls["is_called_strike"] = season_calls[
        "original_call_code"
    ].eq("C")

    rows = []
    for month, month_calls in season_calls.groupby("month", sort=True):
        center_calls = month_calls.loc[
            month_calls["plate_x_in"].abs().le(center_half_width_in)
        ].copy()

        height = (
            center_calls.groupby("z_bin_in")
            .agg(
                called_pitches=("original_call_code", "size"),
                called_strikes=("is_called_strike", "sum"),
            )
            .sort_index()
        )

        if height.empty:
            called_top = float("nan")
        else:
            complete_bins = range(
                int(height.index.min()),
                int(height.index.max()) + 1,
            )
            height = height.reindex(complete_bins, fill_value=0)

            # Pool adjacent height bins to reduce month-to-month noise.
            rolling_calls = height["called_pitches"].rolling(
                3,
                center=True,
                min_periods=2,
            ).sum()
            rolling_strikes = height["called_strikes"].rolling(
                3,
                center=True,
                min_periods=2,
            ).sum()
            strike_rate = rolling_strikes / rolling_calls
            supported = rolling_calls.ge(minimum_bin_calls)

            official_top = official_limits_in(month_calls)[
                "official_top_limit_in"
            ]
            crossings = []

            rates = strike_rate.to_numpy()
            support = supported.to_numpy()
            bins = np.asarray(height.index, dtype=float)

            for index in range(len(height) - 1):
                if (
                    support[index]
                    and support[index + 1]
                    and rates[index] >= 0.5
                    and rates[index + 1] < 0.5
                ):
                    rate_drop = rates[index] - rates[index + 1]
                    fraction = (
                        (rates[index] - 0.5) / rate_drop
                        if rate_drop > 0
                        else 0.5
                    )
                    crossing_in = (
                        bins[index]
                        + 0.5
                        + fraction * (bins[index + 1] - bins[index])
                    )
                    crossings.append(crossing_in)

            called_top = (
                min(crossings, key=lambda value: abs(value - official_top))
                if crossings
                else float("nan")
            )

        limits = official_limits_in(month_calls)
        rows.append(
            {
                "season": 2026,
                "month": month,
                "called_pitches": len(month_calls),
                "center_band_pitches": len(center_calls),
                "games": int(month_calls["game_pk"].nunique()),
                "center_band_half_width_in": center_half_width_in,
                "minimum_bin_calls": minimum_bin_calls,
                "called_zone_top_edge_in": called_top,
                "official_top_limit_in": limits["official_top_limit_in"],
                "called_top_past_official_limit_in": (
                    called_top - limits["official_top_limit_in"]
                ),
            }
        )

    return pd.DataFrame(rows)


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
            "(plate x from -8.5 to 8.5 inches). Official limits for the "
            "pitch centre extend by the 1.45-inch ball radius: horizontally "
            "to 9.95 inches on either side, and vertically to the recorded "
            "zone limits minus/plus 1.45 inches. The vertical summary uses "
            "each season's median recorded zone bottom and top because the "
            "limits vary by batter."
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
        (
            "called_vs_official_zone.csv reports each contour edge, the "
            "official ball-centre limit, the distance past that limit, and "
            "95% percentile intervals from 2,000 whole-game resamples, "
            "conditional on the season's observed official-zone limits. "
            "j3_2026_called_top_by_month.csv reports the 50% called-strike "
            "crossing in the central 6-inch plate band for each month, plus "
            "the monthly median official top. March has no supported "
            "downward crossing and is left missing."
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
    comparison_path = TABLES_DIR / "called_vs_official_zone.csv"
    monthly_top_path = TABLES_DIR / "j3_2026_called_top_by_month.csv"
    notes_path = REPORTS_DIR / "j3_called_zone_notes.md"

    comparison_table = build_called_vs_official_zone(called)
    monthly_top_table = build_2026_monthly_called_top(called)

    grid_table.to_csv(grid_path, index=False)
    edges_table.to_csv(edges_path, index=False)
    comparison_table.to_csv(comparison_path, index=False)
    monthly_top_table.to_csv(monthly_top_path, index=False)

    print(f"Saved grid table: {grid_path}")
    print(f"Saved edge table: {edges_path}")
    print(f"Saved called-vs-official table: {comparison_path}")
    print(f"Saved 2026 monthly called-top table: {monthly_top_path}")
    print("\nCalled-zone edges by season:")
    print(edges_table.to_string(index=False))
    print("\nCalled zone compared with official limits:")
    print(comparison_table.to_string(index=False))
    print("\n2026 called-zone top by month:")
    print(monthly_top_table.to_string(index=False))

    write_called_zone_notes(edges_table, notes_path)


if __name__ == "__main__":
    main()