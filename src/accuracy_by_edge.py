"""Old zone beside new zone: accuracy by the edge that decides the call.

The side edges of the strike zone are defined the same way under the old
zone (2021-2025) and the new ABS zone (2026): the 17-inch plate plus the
ball. The top and the bottom are where the definition changed. Splitting
called pitches by the edge that decides them shows where the 2026 change
sits, while every season is still graded against its own zone. Nothing is
re-graded or re-weighted.

Regions come from grade_pitches.py: thirds of each season's own zone by
height (low, mid, high) and thirds of the plate by width (inside, middle,
outside); "corner" is a pitch off the plate and outside the zone's height.

Tables (reports/tables)
- accuracy_by_edge_season.csv: accuracy with a 95% Wilson interval and the
  share of called pitches, by season, at three levels of detail: the ten
  regions, seven edge groups and three edge families.
- accuracy_by_edge_change.csv: the yearly change 2021-2025 beside the
  change from 2025 to 2026, with an interval from resampling whole games.
- region_shares_by_season.csv: share of called pitches in each region.

Figure (reports/figures)
- accuracy_by_edge.png

Run from the project folder:  python src/accuracy_by_edge.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plot_style
from stats_helpers import wilson_interval

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
REPORTS_DIR = PROJECT_DIR / "reports" / "tables"

NEW_ZONE_FIRST_SEASON = 2026
FIRST_SEASON = 2021
LAST_OLD_SEASON = 2025
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 2026
LEVELS = ("zone_region", "edge_group", "edge_family")
REGIONS = pd.DataFrame(
    [
        ("mid-inside", "side edge only", "sides", "same rule"),
        ("mid-outside", "side edge only", "sides", "same rule"),
        ("high-middle", "top edge only", "top", "rule changed"),
        ("high-inside", "top corner", "top", "rule changed"),
        ("high-outside", "top corner", "top", "rule changed"),
        ("low-middle", "bottom edge only", "bottom", "rule changed"),
        ("low-inside", "bottom corner", "bottom", "rule changed"),
        ("low-outside", "bottom corner", "bottom", "rule changed"),
        ("mid-middle", "heart of the zone", "no edge", "no edge"),
        ("corner", "outside on both axes", "no edge", "no edge"),
    ],
    columns=["zone_region", "edge_group", "edge_family", "rule"],
)
FAMILY_LABELS = {
    "sides": "Side edges (same rule)",
    "top": "Top of the zone (rule changed)",
    "bottom": "Bottom (rule changed)",
}


def load_game_counts() -> pd.DataFrame:
    """Return called pitches and correct calls per game per region."""
    pitches = pd.read_parquet(
        TABLES_DIR / "pitches_graded" / "all_seasons.parquet",
        columns=["season", "game_pk", "zone_region", "is_call_correct"],
    )
    pitches = pitches[pitches["is_call_correct"].notna()]
    counts = (
        pitches.groupby(["season", "game_pk", "zone_region"])[
            "is_call_correct"
        ]
        .agg(called="size", correct="sum")
        .reset_index()
    )
    return counts.merge(REGIONS, on="zone_region", how="left")


def level_table(counts: pd.DataFrame, level: str) -> pd.DataFrame:
    """Return accuracy by season at one level of detail."""
    table = (
        counts.groupby(["season", level, "rule"])[["called", "correct"]]
        .sum()
        .reset_index()
        .rename(columns={level: "name"})
    )
    table.insert(0, "level", level)

    table["accuracy"] = table["correct"] / table["called"]
    intervals = [
        wilson_interval(int(correct), int(called))
        for correct, called in zip(
            table["correct"], table["called"], strict=True
        )
    ]
    table["ci_low"] = [low for low, _ in intervals]
    table["ci_high"] = [high for _, high in intervals]
    season_total = table.groupby("season")["called"].transform("sum")
    table["share_of_called"] = table["called"] / season_total
    table["zone"] = np.where(
        table["season"] >= NEW_ZONE_FIRST_SEASON, "new", "old"
    )
    return table


def season_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Stack the three levels of detail into one long table."""
    table = pd.concat(
        [level_table(counts, level) for level in LEVELS], ignore_index=True
    )
    return table[
        [
            "level",
            "name",
            "rule",
            "season",
            "zone",
            "called",
            "correct",
            "accuracy",
            "ci_low",
            "ci_high",
            "share_of_called",
        ]
    ]


def resampled_accuracy(
    counts: pd.DataFrame, level: str, season: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Resample whole games of one season; return accuracy by name."""
    in_season = counts[counts["season"] == season]
    wide = in_season.pivot_table(
        index="game_pk",
        columns=level,
        values=["called", "correct"],
        aggfunc="sum",
        fill_value=0,
    )
    picks = rng.integers(0, len(wide), size=(BOOTSTRAP_REPS, len(wide)))
    rates = {}
    for name in wide["called"].columns:
        called = wide["called"][name].to_numpy()[picks].sum(axis=1)
        correct = wide["correct"][name].to_numpy()[picks].sum(axis=1)
        rates[name] = correct / called
    return pd.DataFrame(rates)


def change_table(
    counts: pd.DataFrame, by_season: pd.DataFrame
) -> pd.DataFrame:
    """Set the 2025-2026 change beside the yearly change before it."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    old_years = LAST_OLD_SEASON - FIRST_SEASON
    pieces = []
    for level in LEVELS:
        rows = by_season[by_season["level"] == level]
        accuracy = rows.pivot(
            index="name", columns="season", values="accuracy"
        )
        before = resampled_accuracy(counts, level, LAST_OLD_SEASON, rng)
        after = resampled_accuracy(counts, level, NEW_ZONE_FIRST_SEASON, rng)
        spread = (after - before) * 100
        low, high = np.percentile(spread, [2.5, 97.5], axis=0)

        piece = pd.DataFrame(
            {
                "level": level,
                "name": accuracy.index,
                "accuracy_2021": accuracy[FIRST_SEASON].to_numpy(),
                "accuracy_2025": accuracy[LAST_OLD_SEASON].to_numpy(),
                "accuracy_2026": accuracy[NEW_ZONE_FIRST_SEASON].to_numpy(),
            }
        )
        piece["yearly_change_2021_2025_pp"] = (
            (piece["accuracy_2025"] - piece["accuracy_2021"]) * 100 / old_years
        )
        piece["change_2025_2026_pp"] = (
            piece["accuracy_2026"] - piece["accuracy_2025"]
        ) * 100
        interval = pd.DataFrame(
            {
                "name": spread.columns,
                "change_low_pp": low,
                "change_high_pp": high,
            }
        )
        piece = piece.merge(interval, on="name")
        piece["break_from_trend_pp"] = (
            piece["change_2025_2026_pp"] - piece["yearly_change_2021_2025_pp"]
        )
        pieces.append(piece)
    table = pd.concat(pieces, ignore_index=True)
    rules = by_season.drop_duplicates(["level", "name"])[
        ["level", "name", "rule"]
    ]
    return rules.merge(table, on=["level", "name"])


def region_shares(by_season: pd.DataFrame) -> pd.DataFrame:
    """Return the share of called pitches in each region, by season."""
    regions = by_season[by_season["level"] == "zone_region"]
    return (
        regions.pivot(index="name", columns="season", values="share_of_called")
        .rename_axis("zone_region")
        .reset_index()
    )


def plot_edge_families(by_season: pd.DataFrame) -> Figure:
    """Draw accuracy over the seasons for the three edge families."""
    families = by_season[by_season["level"] == "edge_family"]
    fig, ax = plot_style.new_figure(top=0.68)
    seasons = sorted(families["season"].unique())

    for (name, label), color in zip(
        FAMILY_LABELS.items(), plot_style.SERIES, strict=True
    ):
        rows = families[families["name"] == name].sort_values("season")
        percent = rows["accuracy"] * 100
        ax.plot(
            rows["season"],
            percent,
            color=color,
            linewidth=2,
            marker="o",
            markersize=9,
            markeredgecolor=plot_style.SURFACE,
            markeredgewidth=2,
            solid_capstyle="round",
            label=label,
        )
        ax.text(
            seasons[-1] + 0.14,
            percent.iloc[-1],
            f"{percent.iloc[-1]:.1f}%",
            va="center",
            fontsize=10,
            color=plot_style.INK,
        )

    low, high = ax.get_ylim()
    ax.set_ylim(np.floor(low), np.ceil(high))
    ax.set_xlim(seasons[0] - 0.3, seasons[-1] + 0.75)
    ax.set_xticks(seasons)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    change = NEW_ZONE_FIRST_SEASON - 0.5
    ax.axvline(change, color=plot_style.BASELINE, linewidth=1)
    ax.text(
        change + 0.05,
        ax.get_ylim()[0] + 0.12,
        "new ABS zone",
        va="bottom",
        fontsize=9.5,
        color=plot_style.INK_SECONDARY,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.02),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.4,
        columnspacing=1.4,
        labelcolor=plot_style.INK_SECONDARY,
    )
    plot_style.add_title(
        fig,
        "The 2026 jump is at the top of the zone, where the rule changed",
        "Share of correct ball/strike calls, by the edge that decides the "
        "call.\nEach season is graded against its own zone.",
    )
    return fig


def build() -> dict[str, pd.DataFrame]:
    """Build every table, keyed by the name of the CSV it is saved as."""
    counts = load_game_counts()
    by_season = season_table(counts)
    return {
        "accuracy_by_edge_season": by_season,
        "accuracy_by_edge_change": change_table(counts, by_season),
        "region_shares_by_season": region_shares(by_season),
    }


def save(tables: dict[str, pd.DataFrame]) -> None:
    """Write each table to reports/tables as a CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = REPORTS_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"saved {path.relative_to(PROJECT_DIR)}  ({len(table):,} rows)")


def make_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Figure]:
    """Draw the chart and save it to reports/figures."""
    figures = {
        "accuracy_by_edge": plot_edge_families(
            tables["accuracy_by_edge_season"]
        )
    }
    for name, fig in figures.items():
        path = plot_style.save_figure(fig, name)
        print(f"saved {path.relative_to(PROJECT_DIR)}")
    return figures


def main() -> None:
    """Build the tables and the chart, save them and print a summary."""
    tables = build()
    save(tables)
    for fig in make_figures(tables).values():
        plt.close(fig)

    change = tables["accuracy_by_edge_change"]
    columns = [
        "name",
        "rule",
        "accuracy_2021",
        "accuracy_2025",
        "accuracy_2026",
        "yearly_change_2021_2025_pp",
        "change_2025_2026_pp",
        "change_low_pp",
        "change_high_pp",
    ]
    for level in LEVELS:
        print(f"\n{level}")
        rows = change.loc[change["level"] == level, columns]
        print(rows.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
