"""Create season-by-season maps of called pitches and umpire accuracy."""

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
TABLES_DIR = PROJECT_DIR / "reports" / "tables"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

GRID_WIDTH_IN = 2
MIN_CELL_PITCHES = 100

REQUIRED_COLUMNS = {
    "season",
    "original_call_code",
    "is_call_correct",
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


def build_grid_table(frame):
    """Aggregate called pitches into 2-inch cells and calculate rates."""
    frame = frame.copy()

    frame["plate_x_in"] = frame["plate_x_ft"] * 12
    frame["height_above_zone_bottom_in"] = (
        frame["plate_z_ft"] - frame["sz_bot_ft"]
    ) * 12

    frame["x_bin_lower_in"] = (
        np.floor(frame["plate_x_in"] / GRID_WIDTH_IN) * GRID_WIDTH_IN
    ).astype(int)
    frame["z_bin_lower_in"] = (
        np.floor(
            frame["height_above_zone_bottom_in"] / GRID_WIDTH_IN
        ) * GRID_WIDTH_IN
    ).astype(int)

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


def plot_grid_map(grid, season, metric, title, output_name):
    """Save one heatmap, leaving cells with fewer than 100 pitches blank."""
    season_grid = grid.loc[grid["season"].eq(season)].copy()
    season_grid = season_grid.loc[
        season_grid["called_pitches"].ge(MIN_CELL_PITCHES)
        & season_grid[metric].notna()
    ]

    if season_grid.empty:
        print(f"Skipped empty map: {output_name}")
        return

    x_bins = np.arange(
        int(grid["x_bin_lower_in"].min()),
        int(grid["x_bin_lower_in"].max()) + GRID_WIDTH_IN,
        GRID_WIDTH_IN,
    )
    z_bins = np.arange(
        int(grid["height_above_zone_bottom_bin_lower_in"].min()),
        int(grid["height_above_zone_bottom_bin_lower_in"].max())
        + GRID_WIDTH_IN,
        GRID_WIDTH_IN,
    )

    matrix = season_grid.pivot(
        index="height_above_zone_bottom_bin_lower_in",
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
            row.height_above_zone_bottom_bin_lower_in + GRID_WIDTH_IN / 2,
            count_label,
            ha="center",
            va="center",
            fontsize=6,
        )

    axis.set_title(f"{title} — {season}")
    axis.set_xlabel("Horizontal location, plate x (inches; catcher view)")
    axis.set_ylabel("Pitch height above that pitch's zone bottom (inches)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)

    figure.text(
        0.01,
        0.01,
        "Each square is 2 × 2 inches. Cells with fewer than 100 called "
        "pitches are blank. Coordinates are front-of-plate through 2025 "
        "and plate-midpoint in 2026.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = FIGURES_DIR / output_name
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved map: {output_path}")


def main():
    """Build the grid table and season-specific maps."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Graded pitch file not found: {INPUT_PATH}\n"
            "Run src/grade_pitches.py first."
        )

    frame = pd.read_parquet(INPUT_PATH, columns=sorted(REQUIRED_COLUMNS))
    frame = frame.loc[
        frame["original_call_code"].isin(["B", "C"])
        & frame["is_call_correct"].notna()
        & frame["plate_x_ft"].notna()
        & frame["plate_z_ft"].notna()
        & frame["sz_bot_ft"].notna()
    ].copy()

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

    print(f"Called pitches mapped: {len(frame):,}")
    print(
        "Challenge and overturn maps are 2026 only. "
        "2021–2025 have no ABS challenge data."
    )


if __name__ == "__main__":
    main()