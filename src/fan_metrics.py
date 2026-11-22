"""Turn the umpire tiers into numbers a fan can picture.

Every figure is a count of ball/strike calls, shown beside the percentage
it comes from. Each season is measured against its own strike zone.

Tables (reports/tables)
- fan_metrics_by_tier.csv: misses per 1,000 calls, per game and per
  season for each tier, with the gap to Tier 1 and to the league.
- tier_gap_intervals.csv: Tier 1 against Tier 5 each season, with a 95%
  interval from resampling whole games.
- fan_metrics_by_umpire.csv: the same fan numbers for every qualifying
  umpire, plus the calls he would need to reach the league average.
- league_missed_calls_by_season.csv: every graded call, all umpires.
- tier_next_season.csv: how each tier did the following season.

Figures (reports/figures)
- missed_calls_by_season.png
- misses_per_game_by_tier.png

Run from the project folder:  python src/fan_metrics.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plot_style
import umpire_tiers
from stats_helpers import wilson_interval

# A full-time umpire works about 29 games behind the plate in a season.
FULL_SEASON_PLATE_GAMES = 29
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 2026
TOP_TIER = 1
BOTTOM_TIER = 5
TOP_QUARTILE = (1, 2)
BOTTOM_QUARTILE = (4, 5)
UMPIRE_SEASON = ["season", "official_id"]
AXIS_STEP = 10_000
UMPIRE_COLUMNS = [
    "season",
    "zone",
    "official_id",
    "official_name",
    "tier",
    "tier_name",
    "rank_in_season",
    "plate_games",
    "called_pitches",
    "correct",
    "missed",
    "accuracy",
    "league_accuracy",
    "misses_per_1000",
    "misses_per_game",
    "misses_per_season",
    "misses_vs_league",
    "calls_to_reach_league",
]


def add_intervals(table: pd.DataFrame) -> pd.DataFrame:
    """Add a 95% Wilson interval for the accuracy in each row."""
    intervals = [
        wilson_interval(int(correct), int(called))
        for correct, called in zip(
            table["correct"], table["called_pitches"], strict=True
        )
    ]
    return table.assign(
        ci_low=[low for low, _ in intervals],
        ci_high=[high for _, high in intervals],
    )


def tier_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Give each tier its fan numbers and its gap to Tier 1."""
    table = add_intervals(summary)
    table["misses_per_season"] = (
        table["misses_per_game"] * FULL_SEASON_PLATE_GAMES
    )
    table["accuracy_vs_league_pp"] = (
        table["accuracy"] - table["league_accuracy"]
    ) * 100

    top = table[table["tier"] == TOP_TIER].set_index("season")
    table["accuracy_vs_tier_1_pp"] = (
        table["accuracy"] - table["season"].map(top["accuracy"])
    ) * 100
    for column in ("misses_per_1000", "misses_per_game", "misses_per_season"):
        table[f"extra_{column}_vs_tier_1"] = table[column] - table[
            "season"
        ].map(top[column])
    return table


def resampled(
    games: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole games; return accuracy and misses per game."""
    called = games["called"].to_numpy()
    correct = games["correct"].to_numpy()
    picks = rng.integers(0, len(games), size=(BOOTSTRAP_REPS, len(games)))
    total_called = called[picks].sum(axis=1)
    total_correct = correct[picks].sum(axis=1)
    accuracy = total_correct / total_called
    misses_per_game = (total_called - total_correct) / len(games)
    return accuracy, misses_per_game


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the middle 95% of the resampled values."""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def tier_gap_intervals(
    tiers: pd.DataFrame, by_game: pd.DataFrame
) -> pd.DataFrame:
    """Compare Tier 1 with Tier 5 each season.

    The 95% interval comes from resampling whole games within each tier.
    It takes the list of umpires in each tier as given.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    membership = tiers.loc[
        tiers["qualifies"], [*UMPIRE_SEASON, "zone", "tier"]
    ]
    games = by_game.merge(membership, on=UMPIRE_SEASON)
    games = games.assign(missed=games["called"] - games["correct"])

    rows = []
    for season, in_season in games.groupby("season"):
        top = in_season[in_season["tier"] == TOP_TIER]
        bottom = in_season[in_season["tier"] == BOTTOM_TIER]
        top_accuracy, top_misses = resampled(top, rng)
        bottom_accuracy, bottom_misses = resampled(bottom, rng)

        accuracy_gap = 100 * (
            top["correct"].sum() / top["called"].sum()
            - bottom["correct"].sum() / bottom["called"].sum()
        )
        accuracy_low, accuracy_high = interval(
            100 * (top_accuracy - bottom_accuracy)
        )
        misses_gap = bottom["missed"].mean() - top["missed"].mean()
        misses_low, misses_high = interval(bottom_misses - top_misses)
        rows.append(
            {
                "season": season,
                "zone": in_season["zone"].iloc[0],
                "tier_1_games": len(top),
                "tier_5_games": len(bottom),
                "accuracy_gap_pp": accuracy_gap,
                "accuracy_gap_low_pp": accuracy_low,
                "accuracy_gap_high_pp": accuracy_high,
                "fewer_misses_per_1000": accuracy_gap * 10,
                "fewer_misses_per_game": misses_gap,
                "fewer_misses_per_game_low": misses_low,
                "fewer_misses_per_game_high": misses_high,
                "fewer_misses_per_season": misses_gap
                * FULL_SEASON_PLATE_GAMES,
                "fewer_misses_per_season_low": misses_low
                * FULL_SEASON_PLATE_GAMES,
                "fewer_misses_per_season_high": misses_high
                * FULL_SEASON_PLATE_GAMES,
            }
        )
    return pd.DataFrame(rows)


def umpire_metrics(tiers: pd.DataFrame) -> pd.DataFrame:
    """Give every qualifying umpire his fan numbers.

    misses_vs_league is how many more calls he missed than a league-
    average umpire would have missed on the same number of calls
    (negative means fewer). calls_to_reach_league is that number for the
    umpires below the league average, and zero for the rest.
    """
    pool = tiers[tiers["qualifies"]].copy()
    pool["misses_per_season"] = (
        pool["misses_per_game"] * FULL_SEASON_PLATE_GAMES
    )
    pool["misses_vs_league"] = (
        pool["league_accuracy"] - pool["accuracy"]
    ) * pool["called_pitches"]
    pool["calls_to_reach_league"] = np.ceil(
        pool["misses_vs_league"].clip(lower=0)
    ).astype(int)
    return pool[UMPIRE_COLUMNS].reset_index(drop=True)


def league_table(tiers: pd.DataFrame) -> pd.DataFrame:
    """Count every graded call each season, all umpires included."""
    league = (
        tiers.groupby(["season", "zone"])
        .agg(
            umpires=("official_id", "nunique"),
            games=("plate_games", "sum"),
            called_pitches=("called_pitches", "sum"),
            correct=("correct", "sum"),
            missed=("missed", "sum"),
        )
        .reset_index()
    )
    league["accuracy"] = league["correct"] / league["called_pitches"]
    league["misses_per_game"] = league["missed"] / league["games"]
    league["misses_per_1000"] = (
        league["missed"] / league["called_pitches"] * 1000
    )
    return add_intervals(league)


def next_season_by_tier(tiers: pd.DataFrame) -> pd.DataFrame:
    """Show how the umpires of each tier did the following season.

    aal_next_season_pp is measured on new games, so it is the honest test
    of what a tier is worth. It should sit close to predicted_pp, the
    shrunken value the tier was built on.
    """
    pool = tiers.loc[
        tiers["qualifies"],
        [*UMPIRE_SEASON, "tier", "tier_name", "aal_pp", "aal_shrunk_pp"],
    ]
    following = (
        pool[[*UMPIRE_SEASON, "aal_pp", "tier"]]
        .assign(season=pool["season"] - 1)
        .rename(columns={"aal_pp": "aal_next_pp", "tier": "tier_next"})
    )
    pairs = pool.merge(following, on=UMPIRE_SEASON)
    pairs = pairs.assign(
        above_league_next=pairs["aal_next_pp"] > 0,
        top_quartile_next=pairs["tier_next"].isin(TOP_QUARTILE),
        bottom_quartile_next=pairs["tier_next"].isin(BOTTOM_QUARTILE),
    )
    return (
        pairs.groupby(["tier", "tier_name"])
        .agg(
            umpire_seasons=("official_id", "count"),
            aal_this_season_pp=("aal_pp", "mean"),
            predicted_pp=("aal_shrunk_pp", "mean"),
            aal_next_season_pp=("aal_next_pp", "mean"),
            above_league_next_share=("above_league_next", "mean"),
            top_quartile_next_share=("top_quartile_next", "mean"),
            bottom_quartile_next_share=("bottom_quartile_next", "mean"),
        )
        .reset_index()
    )


def plot_missed_calls(league: pd.DataFrame) -> Figure:
    """Draw the league's missed calls per season as thin columns."""
    seasons = league["season"].tolist()
    missed = league["missed"].tolist()
    fewer = missed[0] - missed[-1]

    fig, ax = plot_style.new_figure(top=0.74)
    y_top = max(missed) * 1.15
    ax.set_xlim(min(seasons) - 0.6, max(seasons) + 0.6)
    ax.set_ylim(0, y_top)
    ax.set_xticks(seasons)
    ticks = list(range(0, int(y_top) + 1, AXIS_STEP))
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        ["0" if tick == 0 else f"{tick // 1000}K" for tick in ticks]
    )
    colors = [plot_style.SERIES[0]] * len(seasons)
    plot_style.rounded_columns(ax, seasons, missed, colors)
    plot_style.label_columns(
        ax, seasons, missed, [f"{value:,}" for value in missed]
    )
    plot_style.add_title(
        fig,
        f"Umpires missed {fewer:,} fewer calls in {seasons[-1]} "
        f"than in {seasons[0]}",
        "Missed ball/strike calls per regular season.\n2021-2025 are "
        "graded against the old zone, 2026 against the new ABS zone.",
    )
    return fig


def plot_misses_by_tier(table: pd.DataFrame, season: int) -> Figure:
    """Draw misses per game for the five tiers of one season."""
    rows = table[table["season"] == season].sort_values("tier")
    tiers = rows["tier"].astype(int).tolist()
    misses = rows["misses_per_game"].tolist()
    zone = rows["zone"].iloc[0]
    gap = misses[-1] - misses[0]

    fig, ax = plot_style.new_figure()
    ax.set_xlim(0.4, len(tiers) + 0.6)
    ax.set_ylim(0, 2 * math.ceil(max(misses) * 1.15 / 2))
    ax.set_xticks(tiers)
    ax.set_xticklabels(
        [
            f"Tier {tier}\n{name}"
            for tier, name in zip(tiers, rows["tier_name"], strict=True)
        ]
    )
    colors = [plot_style.TIER_COLORS[tier] for tier in tiers]
    plot_style.rounded_columns(ax, tiers, misses, colors)
    plot_style.label_columns(
        ax, tiers, misses, [f"{value:.1f}" for value in misses]
    )
    plot_style.add_title(
        fig,
        f"A Tier 5 umpire misses {gap:.1f} more calls a game "
        "than a Tier 1 umpire",
        f"Missed ball/strike calls per game behind the plate, {season} "
        f"({zone} zone).",
    )
    fig.subplots_adjust(bottom=0.16)
    return fig


def build() -> dict[str, pd.DataFrame]:
    """Build every table, keyed by the name of the CSV it is saved as."""
    tier_tables = umpire_tiers.build()
    tiers = tier_tables["umpire_tiers"]
    by_game = tier_tables["umpire_game_accuracy"]
    return {
        "fan_metrics_by_tier": tier_metrics(tier_tables["tier_summary"]),
        "tier_gap_intervals": tier_gap_intervals(tiers, by_game),
        "fan_metrics_by_umpire": umpire_metrics(tiers),
        "league_missed_calls_by_season": league_table(tiers),
        "tier_next_season": next_season_by_tier(tiers),
    }


def make_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Figure]:
    """Draw the charts and save them to reports/figures."""
    by_tier = tables["fan_metrics_by_tier"]
    figures = {
        "missed_calls_by_season": plot_missed_calls(
            tables["league_missed_calls_by_season"]
        ),
        "misses_per_game_by_tier": plot_misses_by_tier(
            by_tier, int(by_tier["season"].max())
        ),
    }
    for name, fig in figures.items():
        path = plot_style.save_figure(fig, name)
        print(f"saved {path.relative_to(plot_style.PROJECT_DIR)}")
    return figures


def main() -> None:
    """Build the tables and charts, save them and print a summary."""
    tables = build()
    umpire_tiers.save(tables)
    for fig in make_figures(tables).values():
        plt.close(fig)

    print()
    print(tables["tier_gap_intervals"].round(2).to_string(index=False))
    print()
    print(tables["tier_next_season"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
