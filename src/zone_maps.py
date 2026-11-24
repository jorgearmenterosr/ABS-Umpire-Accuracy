"""Create season-by-season and per-umpire pitch maps."""

import math
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    PROJECT_DIR / "data" / "tables" / "pitches_graded" / "all_seasons.parquet"
)
GAMES_PATH = PROJECT_DIR / "data" / "tables" / "games" / "all_seasons.parquet"
TABLES_DIR = PROJECT_DIR / "reports" / "tables"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
UMPIRE_FIGURES_DIR = FIGURES_DIR / "umpire_heatmaps"

GRID_WIDTH_IN = 2
UMPIRE_GRID_WIDTH_IN = 4
MIN_CELL_PITCHES = 100
MIN_UMPIRE_GAMES = 10
MIN_UMPIRE_CELL_PITCHES = 50

REQUIRED_COLUMNS = {
    "game_pk",
    "season",
    "original_call_code",
    "is_call_correct",
    "is_strike_zone",
    "is_challenged",
    "is_overturned",
    "plate_x_ft",
    "plate_z_ft",
    "sz_bot_ft",
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


def add_location_bins(frame, grid_width_in=GRID_WIDTH_IN):
    """Add horizontal and zone-relative vertical bins."""
    frame = frame.copy()
    frame["plate_x_in"] = frame["plate_x_ft"] * 12
    frame["height_above_zone_bottom_in"] = (
        frame["plate_z_ft"] - frame["sz_bot_ft"]
    ) * 12
    frame["x_bin_lower_in"] = (
        np.floor(frame["plate_x_in"] / grid_width_in) * grid_width_in
    ).astype(int)
    frame["z_bin_lower_in"] = (
        np.floor(
            frame["height_above_zone_bottom_in"] / grid_width_in
        )
        * grid_width_in
    ).astype(int)
    return frame


def build_grid_table(frame):
    """Aggregate called pitches into 2-inch league-wide cells."""
    frame = add_location_bins(frame, GRID_WIDTH_IN)

    rows = []
    group_columns = [
        "season",
        "x_bin_lower_in",
        "z_bin_lower_in",
    ]

    for key, group in frame.groupby(
        group_columns,
        dropna=True,
        observed=True,
    ):
        season, x_bin, z_bin = key
        called_pitches = len(group)
        correct = group["is_call_correct"].astype(bool)
        incorrect_calls = int((~correct).sum())

        error_rate = float("nan")
        error_ci_low = float("nan")
        error_ci_high = float("nan")

        if called_pitches >= MIN_CELL_PITCHES:
            error_rate = incorrect_calls / called_pitches
            error_ci_low, error_ci_high = wilson_interval(
                incorrect_calls,
                called_pitches,
            )

        row = {
            "season": int(season),
            "x_bin_lower_in": x_bin,
            "height_above_zone_bottom_bin_lower_in": z_bin,
            "called_pitches": called_pitches,
            "incorrect_calls": incorrect_calls,
            "error_rate": error_rate,
            "error_ci_low": error_ci_low,
            "error_ci_high": error_ci_high,
            "challenged_pitches": float("nan"),
            "challenge_rate": float("nan"),
            "challenge_ci_low": float("nan"),
            "challenge_ci_high": float("nan"),
            "overturned_challenges": float("nan"),
            "overturn_rate": float("nan"),
            "overturn_ci_low": float("nan"),
            "overturn_ci_high": float("nan"),
        }

        if int(season) == 2026:
            challenged = group["is_challenged"].fillna(False).astype(bool)
            challenged_pitches = int(challenged.sum())
            overturned = (
                challenged
                & group["is_overturned"].fillna(False).astype(bool)
            )
            overturned_challenges = int(overturned.sum())

            row["challenged_pitches"] = challenged_pitches
            row["overturned_challenges"] = overturned_challenges

            if called_pitches >= MIN_CELL_PITCHES:
                challenge_rate = challenged_pitches / called_pitches
                challenge_low, challenge_high = wilson_interval(
                    challenged_pitches,
                    called_pitches,
                )
                row.update(
                    {
                        "challenge_rate": challenge_rate,
                        "challenge_ci_low": challenge_low,
                        "challenge_ci_high": challenge_high,
                    }
                )

                if challenged_pitches > 0:
                    overturn_rate = (
                        overturned_challenges / challenged_pitches
                    )
                    overturn_low, overturn_high = wilson_interval(
                        overturned_challenges,
                        challenged_pitches,
                    )
                    row.update(
                        {
                            "overturn_rate": overturn_rate,
                            "overturn_ci_low": overturn_low,
                            "overturn_ci_high": overturn_high,
                        }
                    )

        rows.append(row)

    return pd.DataFrame(rows)


def map_bin_edges(grid, grid_width_in=GRID_WIDTH_IN):
    """Return common bin edges so maps share the same axes."""
    x_bins = np.arange(
        int(grid["x_bin_lower_in"].min()),
        int(grid["x_bin_lower_in"].max()) + grid_width_in,
        grid_width_in,
    )
    z_column = "height_above_zone_bottom_bin_lower_in"
    z_bins = np.arange(
        int(grid[z_column].min()),
        int(grid[z_column].max()) + grid_width_in,
        grid_width_in,
    )
    return x_bins, z_bins


def plot_grid_map(grid, season, metric, title, output_name):
    """Save a season map, leaving low-count cells blank."""
    season_grid = grid.loc[grid["season"].eq(season)].copy()
    season_grid = season_grid.loc[
        season_grid["called_pitches"].ge(MIN_CELL_PITCHES)
        & season_grid[metric].notna()
    ]

    if season_grid.empty:
        print(f"Skipped empty map: {output_name}")
        return

    x_bins, z_bins = map_bin_edges(grid, GRID_WIDTH_IN)
    z_column = "height_above_zone_bottom_bin_lower_in"

    matrix = season_grid.pivot(
        index=z_column,
        columns="x_bin_lower_in",
        values=metric,
    ).reindex(index=z_bins, columns=x_bins)

    x_edges = np.append(x_bins, x_bins[-1] + GRID_WIDTH_IN)
    z_edges = np.append(z_bins, z_bins[-1] + GRID_WIDTH_IN)

    figure, axis = plt.subplots(figsize=(12, 9))

    if metric == "called_pitches":
        maximum = max(
            float(
                grid.loc[
                    grid["called_pitches"].ge(MIN_CELL_PITCHES),
                    "called_pitches",
                ].max()
            ),
            MIN_CELL_PITCHES + 1,
        )
        mesh = axis.pcolormesh(
            x_edges,
            z_edges,
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            shading="flat",
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(
                vmin=MIN_CELL_PITCHES,
                vmax=maximum,
            ),
        )
        colorbar = figure.colorbar(mesh, ax=axis)
        colorbar.set_label("Called pitches per cell (log scale)")
    else:
        mesh = axis.pcolormesh(
            x_edges,
            z_edges,
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            shading="flat",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
        )
        colorbar = figure.colorbar(mesh, ax=axis)
        colorbar.set_label("Rate")
        colorbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1)
        )

    for row in season_grid.itertuples(index=False):
        if metric == "called_pitches":
            count_label = f"{int(row.called_pitches)}"
        elif metric == "overturn_rate":
            count_label = f"n={int(row.challenged_pitches)}"
        else:
            count_label = f"n={int(row.called_pitches)}"

        axis.text(
            row.x_bin_lower_in + GRID_WIDTH_IN / 2,
            getattr(row, z_column) + GRID_WIDTH_IN / 2,
            count_label,
            ha="center",
            va="center",
            fontsize=6,
        )

    axis.set_title(f"{title} - {season}")
    axis.set_xlabel("Horizontal location, plate x (inches; catcher view)")
    axis.set_ylabel("Pitch height above that pitch's zone bottom (inches)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)

    figure.text(
        0.01,
        0.01,
        "Each square is 2 x 2 inches. Cells with fewer than 100 called "
        "pitches are blank. Coordinates are front-of-plate through 2025 "
        "and plate-midpoint in 2026.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = FIGURES_DIR / output_name
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved map: {output_path}")


def count_plate_games(games):
    """Count regular-season plate assignments by umpire and season."""
    assignments = games.dropna(subset=["plate_umpire_id"]).copy()
    assignments["plate_umpire_id"] = assignments[
        "plate_umpire_id"
    ].astype("int64")
    assignments = assignments.drop_duplicates(["season", "game_pk"])

    return (
        assignments.groupby(["season", "plate_umpire_id"])
        .agg(
            plate_games=("game_pk", "nunique"),
            official_name=("plate_umpire_name", "first"),
        )
        .reset_index()
    )


def build_umpire_grid_table(frame, games):
    """Aggregate umpire-season calls into 4-inch cells with signed bias."""
    plate_games = count_plate_games(games)
    eligible = plate_games[
        plate_games["plate_games"].ge(MIN_UMPIRE_GAMES)
    ].copy()

    game_assignments = games[
        ["season", "game_pk", "plate_umpire_id", "plate_umpire_name"]
    ].drop_duplicates(["season", "game_pk"])

    umpire_frame = frame.merge(
        game_assignments,
        on=["season", "game_pk"],
        how="left",
        validate="many_to_one",
    )
    umpire_frame = umpire_frame.dropna(
        subset=["plate_umpire_id", "plate_umpire_name", "is_strike_zone"]
    ).copy()
    umpire_frame["plate_umpire_id"] = umpire_frame[
        "plate_umpire_id"
    ].astype("int64")

    umpire_frame = umpire_frame.merge(
        eligible[["season", "plate_umpire_id", "plate_games"]],
        on=["season", "plate_umpire_id"],
        how="inner",
        validate="many_to_one",
    )

    umpire_frame = add_location_bins(
        umpire_frame,
        UMPIRE_GRID_WIDTH_IN,
    )
    umpire_frame["is_call_correct"] = umpire_frame[
        "is_call_correct"
    ].astype(bool)
    umpire_frame["is_strike_zone"] = umpire_frame[
        "is_strike_zone"
    ].astype(bool)
    umpire_frame["incorrect_call"] = ~umpire_frame["is_call_correct"]

    # Positive signed bias means more false strikes than false balls.
    umpire_frame["false_strike"] = (
        umpire_frame["original_call_code"].eq("C")
        & ~umpire_frame["is_strike_zone"]
    )
    umpire_frame["false_ball"] = (
        umpire_frame["original_call_code"].eq("B")
        & umpire_frame["is_strike_zone"]
    )

    group_columns = [
        "season",
        "plate_umpire_id",
        "plate_umpire_name",
        "plate_games",
        "x_bin_lower_in",
        "z_bin_lower_in",
    ]
    cells = (
        umpire_frame.groupby(group_columns, observed=True)
        .agg(
            called_pitches=("is_call_correct", "size"),
            incorrect_calls=("incorrect_call", "sum"),
            false_strikes=("false_strike", "sum"),
            false_balls=("false_ball", "sum"),
        )
        .reset_index()
    )

    cells["miss_rate"] = (
        cells["incorrect_calls"] / cells["called_pitches"]
    )
    cells["signed_bias_pp"] = (
        (cells["false_strikes"] - cells["false_balls"])
        / cells["called_pitches"]
        * 100
    )
    return cells


def umpire_bias_limit(cells):
    """Choose one common color limit from adequately sampled cells."""
    supported = cells.loc[
        cells["called_pitches"].ge(MIN_UMPIRE_CELL_PITCHES),
        "signed_bias_pp",
    ].abs()

    if supported.empty:
        return 2.0

    percentile = float(np.nanpercentile(supported, 98))
    return max(2.0, math.ceil(percentile / 2) * 2)


def plot_umpire_heatmap(cells, x_bins, z_bins, bias_limit):
    """Save miss-rate and signed-bias panels for one umpire-season."""
    official_id = int(cells["plate_umpire_id"].iloc[0])
    official_name = str(cells["plate_umpire_name"].iloc[0])
    season = int(cells["season"].iloc[0])
    plate_games = int(cells["plate_games"].iloc[0])

    count_matrix = cells.pivot(
        index="z_bin_lower_in",
        columns="x_bin_lower_in",
        values="called_pitches",
    ).reindex(index=z_bins, columns=x_bins, fill_value=0)

    supported = count_matrix.ge(MIN_UMPIRE_CELL_PITCHES)

    miss_matrix = (
        cells.pivot(
            index="z_bin_lower_in",
            columns="x_bin_lower_in",
            values="miss_rate",
        )
        .reindex(index=z_bins, columns=x_bins)
        .where(supported)
    )
    bias_matrix = (
        cells.pivot(
            index="z_bin_lower_in",
            columns="x_bin_lower_in",
            values="signed_bias_pp",
        )
        .reindex(index=z_bins, columns=x_bins)
        .where(supported)
    )

    x_edges = np.append(x_bins, x_bins[-1] + UMPIRE_GRID_WIDTH_IN)
    z_edges = np.append(z_bins, z_bins[-1] + UMPIRE_GRID_WIDTH_IN)
    miss_cmap = plt.get_cmap("RdYlBu_r").copy()
    bias_cmap = plt.get_cmap("RdBu_r").copy()
    miss_cmap.set_bad("#bdbdbd")
    bias_cmap.set_bad("#bdbdbd")

    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    miss_mesh = axes[0].pcolormesh(
        x_edges,
        z_edges,
        np.ma.masked_invalid(miss_matrix.to_numpy(dtype=float)),
        shading="flat",
        cmap=miss_cmap,
        vmin=0,
        vmax=0.25,
    )
    miss_bar = figure.colorbar(miss_mesh, ax=axes[0], pad=0.02)
    miss_bar.set_label("Incorrect-call rate")
    miss_bar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )

    bias_mesh = axes[1].pcolormesh(
        x_edges,
        z_edges,
        np.ma.masked_invalid(bias_matrix.to_numpy(dtype=float)),
        shading="flat",
        cmap=bias_cmap,
        norm=matplotlib.colors.TwoSlopeNorm(
            vmin=-bias_limit,
            vcenter=0,
            vmax=bias_limit,
        ),
    )
    bias_bar = figure.colorbar(bias_mesh, ax=axes[1], pad=0.02)
    bias_bar.set_label("Signed bias (percentage points)")

    for row in cells.itertuples(index=False):
        if row.called_pitches < MIN_UMPIRE_CELL_PITCHES:
            continue

        x_center = row.x_bin_lower_in + UMPIRE_GRID_WIDTH_IN / 2
        z_center = row.z_bin_lower_in + UMPIRE_GRID_WIDTH_IN / 2
        label = f"n={int(row.called_pitches)}"

        for axis in axes:
            axis.text(
                x_center,
                z_center,
                label,
                ha="center",
                va="center",
                fontsize=7,
            )

    axes[0].set_title("Miss rate")
    axes[1].set_title("Signed bias: false strikes minus false balls")
    for axis in axes:
        axis.set_xlabel("Plate x (inches; catcher view)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.12)

    axes[0].set_ylabel("Height above that pitch's zone bottom (inches)")
    figure.suptitle(
        f"{official_name} (ID {official_id}) - {season} "
        f"({plate_games} plate games)"
    )
    figure.text(
        0.01,
        0.01,
        "Each cell is 4 x 4 inches. Cells with fewer than 50 called "
        "pitches are gray. Positive bias means more false strikes.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))

    UMPIRE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UMPIRE_FIGURES_DIR / f"{official_id}_{season}.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_all_umpire_heatmaps(cells):
    """Save maps for all umpire-seasons with enough plate games."""
    if cells.empty:
        print("No eligible umpire-season cells for individual maps.")
        return

    bin_source = cells.rename(
        columns={
            "z_bin_lower_in": "height_above_zone_bottom_bin_lower_in"
        }
    )
    x_bins, z_bins = map_bin_edges(
        bin_source,
        UMPIRE_GRID_WIDTH_IN,
    )
    bias_limit = umpire_bias_limit(cells)

    map_groups = cells.groupby(
        ["season", "plate_umpire_id", "plate_umpire_name"],
        sort=True,
        observed=True,
    )

    saved = 0
    all_gray = 0
    for _, group in map_groups:
        if not group["called_pitches"].ge(
            MIN_UMPIRE_CELL_PITCHES
        ).any():
            all_gray += 1

        plot_umpire_heatmap(group, x_bins, z_bins, bias_limit)
        saved += 1

    print(f"Saved {saved:,} umpire-season heatmaps: {UMPIRE_FIGURES_DIR}")
    print(f"Maps with all cells gray: {all_gray:,}")
    print(
        f"Minimums: {MIN_UMPIRE_GAMES} plate games and "
        f"{MIN_UMPIRE_CELL_PITCHES} calls per colored cell."
    )
    print(
        "The signed-bias color scale is shared across maps and capped "
        f"at +/-{bias_limit:g} percentage points."
    )


def main():
    """Build league-wide maps and per-umpire heatmaps."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Graded pitch file not found: {INPUT_PATH}\n"
            "Run src/grade_pitches.py first."
        )
    if not GAMES_PATH.exists():
        raise FileNotFoundError(
            f"Games file not found: {GAMES_PATH}\n"
            "Run src/combine_seasons.py first."
        )

    frame = pd.read_parquet(INPUT_PATH, columns=sorted(REQUIRED_COLUMNS))
    frame = frame.loc[
        frame["original_call_code"].isin(["B", "C"])
        & frame["is_call_correct"].notna()
        & frame["is_strike_zone"].notna()
        & frame["plate_x_ft"].notna()
        & frame["plate_z_ft"].notna()
        & frame["sz_bot_ft"].notna()
    ].copy()

    games = pd.read_parquet(
        GAMES_PATH,
        columns=[
            "season",
            "game_pk",
            "plate_umpire_id",
            "plate_umpire_name",
        ],
    )

    grid = build_grid_table(frame)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table_path = TABLES_DIR / "zone_maps_grid.csv"
    grid.to_csv(table_path, index=False)
    print(f"Saved {len(grid):,} grid cells: {table_path}")

    metrics = [
        ("called_pitches", "Called pitches", "called_pitches"),
        ("error_rate", "Umpire error rate", "error_rate"),
    ]

    for season in sorted(grid["season"].unique()):
        for metric, title, suffix in metrics:
            plot_grid_map(
                grid,
                int(season),
                metric,
                title,
                f"zone_map_{suffix}_{int(season)}.png",
            )

    for metric, title, suffix in [
        ("challenge_rate", "Challenge rate", "challenge_rate"),
        ("overturn_rate", "Overturn rate", "overturn_rate"),
    ]:
        plot_grid_map(
            grid,
            2026,
            metric,
            title,
            f"zone_map_{suffix}_2026.png",
        )

    umpire_cells = build_umpire_grid_table(frame, games)
    plot_all_umpire_heatmaps(umpire_cells)

    print(f"Called pitches mapped: {len(frame):,}")
    print(
        "Challenge and overturn maps are 2026 only. "
        "Seasons 2021-2025 have no ABS challenge data."
    )


if __name__ == "__main__":
    main()