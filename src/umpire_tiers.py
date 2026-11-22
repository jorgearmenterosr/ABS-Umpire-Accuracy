"""Build umpire performance tiers, one season at a time.

Each season is graded against its own strike zone: the old zone for
2021-2025 and the new ABS zone for 2026. Nothing is standardised across
seasons, so an umpire is only ever compared with the umpires who called
the same zone in the same year.

How a tier is assigned
1. Accuracy above league (AAL): the umpire's share of correct calls minus
   the share for all qualifying umpires that season, in percentage points.
2. Shrinkage: each AAL is pulled toward zero in proportion to how noisy it
   is, so a short season cannot outrank a full one on luck.
3. Tiers: umpires are ranked on the shrunken AAL within the season. The
   top 10% are Tier 1, the next 15% Tier 2, the middle 50% Tier 3, the
   next 15% Tier 4 and the bottom 10% Tier 5.

Umpires with fewer than 10 plate games stay in the table, flagged as not
qualifying and without a tier.

Run from the project folder:  python src/umpire_tiers.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from stats_helpers import wilson_interval

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
REPORTS_DIR = PROJECT_DIR / "reports" / "tables"

MIN_PLATE_GAMES = 10
MIN_CAREER_SEASONS = 3
NEW_ZONE_FIRST_SEASON = 2026
Z_95 = 1.96
TOP_SHARE = 0.10
QUARTER_SHARE = 0.25
MIDDLE_TIER = 3
TIER_NAMES = {
    1: "Elite",
    2: "High",
    3: "Average",
    4: "Below average",
    5: "Lowest",
}
UMPIRE_SEASON = ["season", "official_id"]
TIER_COLUMNS = [
    "season",
    "zone",
    "official_id",
    "official_name",
    "qualifies",
    "plate_games",
    "called_pitches",
    "correct",
    "missed",
    "accuracy",
    "ci_low",
    "ci_high",
    "league_accuracy",
    "aal_pp",
    "se_game_pp",
    "se_binomial_pp",
    "skill_sd_pp",
    "shrink_factor",
    "aal_shrunk_pp",
    "shrunk_low_pp",
    "shrunk_high_pp",
    "secure",
    "rank_in_season",
    "umpires_in_season",
    "percentile",
    "tier",
    "tier_name",
    "misses_per_game",
    "misses_per_1000",
    "game_sd_pp",
    "best_game",
    "worst_game",
]


def load_game_table() -> pd.DataFrame:
    """Return one row per umpire per game: called pitches and correct."""
    pitches = pd.read_parquet(
        TABLES_DIR / "pitches_graded" / "all_seasons.parquet",
        columns=["season", "game_pk", "is_call_correct"],
    )
    pitches = pitches[pitches["is_call_correct"].notna()]

    games = pd.read_parquet(
        TABLES_DIR / "games" / "all_seasons.parquet",
        columns=["game_pk", "plate_umpire_id", "plate_umpire_name"],
    ).dropna(subset=["plate_umpire_id"])
    games = games.astype({"game_pk": "int64", "plate_umpire_id": "int64"})
    games = games.rename(
        columns={
            "plate_umpire_id": "official_id",
            "plate_umpire_name": "official_name",
        }
    )

    graded = len(pitches)
    pitches = pitches.merge(games, on="game_pk", how="inner")
    if len(pitches) < graded:
        dropped = graded - len(pitches)
        print(f"note: {dropped:,} graded pitches have no plate umpire")

    by_game = (
        pitches.groupby(["season", "official_id", "game_pk"])[
            "is_call_correct"
        ]
        .agg(called="size", correct="sum")
        .reset_index()
    )
    # One spelling of each name, so an umpire is never split in two.
    names = games.drop_duplicates("official_id", keep="last").set_index(
        "official_id"
    )["official_name"]
    by_game.insert(2, "official_name", by_game["official_id"].map(names))
    return by_game


def season_table(by_game: pd.DataFrame) -> pd.DataFrame:
    """Return one row per umpire per season with accuracy and misses."""
    per_game = by_game.assign(
        game_accuracy=by_game["correct"] / by_game["called"]
    )
    ump = (
        per_game.groupby(["season", "official_id", "official_name"])
        .agg(
            plate_games=("game_pk", "nunique"),
            called_pitches=("called", "sum"),
            correct=("correct", "sum"),
            game_sd_pp=("game_accuracy", "std"),
            best_game=("game_accuracy", "max"),
            worst_game=("game_accuracy", "min"),
        )
        .reset_index()
    )
    ump["game_sd_pp"] = ump["game_sd_pp"] * 100
    ump["missed"] = ump["called_pitches"] - ump["correct"]
    ump["accuracy"] = ump["correct"] / ump["called_pitches"]
    ump["misses_per_game"] = ump["missed"] / ump["plate_games"]
    ump["misses_per_1000"] = ump["missed"] / ump["called_pitches"] * 1000
    ump["zone"] = np.where(
        ump["season"] >= NEW_ZONE_FIRST_SEASON, "new", "old"
    )
    ump["qualifies"] = ump["plate_games"] >= MIN_PLATE_GAMES

    intervals = [
        wilson_interval(int(correct), int(called))
        for correct, called in zip(
            ump["correct"], ump["called_pitches"], strict=True
        )
    ]
    ump["ci_low"] = [low for low, _ in intervals]
    ump["ci_high"] = [high for _, high in intervals]
    return ump


def add_league(ump: pd.DataFrame) -> pd.DataFrame:
    """Add the league accuracy of qualifying umpires and each AAL."""
    pool = ump[ump["qualifies"]]
    league = pool.groupby("season")[["called_pitches", "correct"]].sum()
    league_accuracy = league["correct"] / league["called_pitches"]
    ump = ump.assign(league_accuracy=ump["season"].map(league_accuracy))
    ump["aal_pp"] = (ump["accuracy"] - ump["league_accuracy"]) * 100
    return ump


def add_sampling_error(
    ump: pd.DataFrame, by_game: pd.DataFrame
) -> pd.DataFrame:
    """Add each umpire-season's standard error, measured two ways.

    se_binomial_pp treats every pitch as independent.
    se_game_pp uses the umpire's own game-to-game variation, so pitches
    from the same game (same pitchers, catcher and conditions) are not
    counted as independent. The tiers use se_game_pp.
    """
    accuracy = ump["accuracy"]
    ump = ump.assign(
        se_binomial_pp=100
        * np.sqrt(accuracy * (1 - accuracy) / ump["called_pitches"])
    )

    games = by_game.merge(
        ump[[*UMPIRE_SEASON, "accuracy"]], on=UMPIRE_SEASON, how="left"
    )
    games["squared_residual"] = (
        games["correct"] - games["accuracy"] * games["called"]
    ) ** 2
    residuals = (
        games.groupby(UMPIRE_SEASON)["squared_residual"].sum().reset_index()
    )
    ump = ump.merge(residuals, on=UMPIRE_SEASON, how="left")

    plate_games = ump["plate_games"]
    correction = plate_games / (plate_games - 1).where(plate_games > 1)
    ump["se_game_pp"] = (
        100
        * np.sqrt(correction * ump["squared_residual"])
        / ump["called_pitches"]
    )
    return ump.drop(columns="squared_residual")


def add_shrinkage(ump: pd.DataFrame) -> pd.DataFrame:
    """Shrink each umpire's AAL toward zero by its reliability.

    skill_sd_pp is how much true skill differs across the qualifying
    umpires of a season: the observed variance of AAL minus the average
    sampling variance. shrink_factor = skill^2 / (skill^2 + se^2) is the
    share of an umpire's gap to the league that is kept.
    """
    by_season = ump[ump["qualifies"]].groupby("season")
    observed_var = by_season["aal_pp"].var()
    noise_var = by_season["se_game_pp"].agg(lambda se: (se**2).mean())
    skill_sd = np.sqrt((observed_var - noise_var).clip(lower=0))

    ump = ump.assign(skill_sd_pp=ump["season"].map(skill_sd))
    skill = ump["skill_sd_pp"] ** 2
    noise = ump["se_game_pp"] ** 2
    ump["shrink_factor"] = skill / (skill + noise)
    ump["aal_shrunk_pp"] = ump["shrink_factor"] * ump["aal_pp"]
    half_width = Z_95 * np.sqrt(ump["shrink_factor"] * noise)
    ump["shrunk_low_pp"] = ump["aal_shrunk_pp"] - half_width
    ump["shrunk_high_pp"] = ump["aal_shrunk_pp"] + half_width
    ump["secure"] = (ump["shrunk_low_pp"] > 0) | (ump["shrunk_high_pp"] < 0)
    return ump


def tiers_from_scores(scores: pd.Series) -> pd.DataFrame:
    """Rank scores, best first, and cut the ranking into five tiers."""
    count = len(scores)
    rank = scores.rank(ascending=False, method="first").astype(int)
    top = math.floor(TOP_SHARE * count + 0.5)
    quarter = math.floor(QUARTER_SHARE * count + 0.5)

    tier = pd.Series(MIDDLE_TIER, index=scores.index)
    tier[rank <= quarter] = 2
    tier[rank <= top] = 1
    tier[rank > count - quarter] = 4
    tier[rank > count - top] = 5
    return pd.DataFrame(
        {
            "rank": rank,
            "group_size": count,
            "percentile": 100 * (count - rank + 0.5) / count,
            "tier": tier,
        }
    )


def add_tiers(ump: pd.DataFrame) -> pd.DataFrame:
    """Rank qualifying umpires within each season and assign tiers."""
    pool = ump[ump["qualifies"]]
    ranked = pd.concat(
        tiers_from_scores(group["aal_shrunk_pp"])
        for _, group in pool.groupby("season")
    ).rename(
        columns={"rank": "rank_in_season", "group_size": "umpires_in_season"}
    )
    ump = ump.join(ranked)
    for column in ("rank_in_season", "umpires_in_season", "tier"):
        ump[column] = ump[column].astype("Int64")
    ump["tier_name"] = ump["tier"].map(TIER_NAMES)
    return ump.sort_values(
        ["season", "qualifies", "rank_in_season", "accuracy"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)[TIER_COLUMNS]


def tier_summary(ump: pd.DataFrame) -> pd.DataFrame:
    """Describe each tier in each season: size, cut-offs and accuracy."""
    pool = ump[ump["qualifies"]]
    summary = (
        pool.groupby(["season", "zone", "tier", "tier_name"])
        .agg(
            umpires=("official_id", "count"),
            lowest_aal_shrunk_pp=("aal_shrunk_pp", "min"),
            highest_aal_shrunk_pp=("aal_shrunk_pp", "max"),
            lowest_accuracy=("accuracy", "min"),
            highest_accuracy=("accuracy", "max"),
            plate_games=("plate_games", "sum"),
            called_pitches=("called_pitches", "sum"),
            correct=("correct", "sum"),
            missed=("missed", "sum"),
            league_accuracy=("league_accuracy", "first"),
            skill_sd_pp=("skill_sd_pp", "first"),
        )
        .reset_index()
    )
    summary["accuracy"] = summary["correct"] / summary["called_pitches"]
    summary["misses_per_1000"] = (
        summary["missed"] / summary["called_pitches"] * 1000
    )
    summary["misses_per_game"] = summary["missed"] / summary["plate_games"]
    return summary


def career_table(ump: pd.DataFrame) -> pd.DataFrame:
    """Average each umpire's shrunken AAL over his qualifying seasons.

    The old-zone seasons (2021-2025) and the new-zone season (2026) are
    also shown in separate columns, so the two zones are never blended
    out of sight.
    """
    pool = ump[ump["qualifies"]].sort_values("season")
    pool = pool.assign(
        season_tier=pool["season"].astype(str)
        + ":T"
        + pool["tier"].astype(str)
    )
    career = (
        pool.groupby("official_id")
        .agg(
            official_name=("official_name", "last"),
            seasons=("season", "nunique"),
            first_season=("season", "min"),
            last_season=("season", "max"),
            plate_games=("plate_games", "sum"),
            called_pitches=("called_pitches", "sum"),
            correct=("correct", "sum"),
            missed=("missed", "sum"),
            career_score_pp=("aal_shrunk_pp", "mean"),
            season_sd_pp=("aal_shrunk_pp", "std"),
            best_tier=("tier", "min"),
            worst_tier=("tier", "max"),
            tiers_by_season=("season_tier", " ".join),
        )
        .reset_index()
    )
    career["career_se_pp"] = career["season_sd_pp"] / np.sqrt(
        career["seasons"]
    )

    for zone in ("old", "new"):
        in_zone = pool[pool["zone"] == zone].groupby("official_id")
        totals = in_zone[["called_pitches", "correct"]].sum()
        career[f"{zone}_zone_seasons"] = (
            career["official_id"].map(in_zone["season"].nunique()).fillna(0)
        ).astype(int)
        career[f"{zone}_zone_accuracy"] = career["official_id"].map(
            totals["correct"] / totals["called_pitches"]
        )
        career[f"{zone}_zone_score_pp"] = career["official_id"].map(
            in_zone["aal_shrunk_pp"].mean()
        )

    career["eligible"] = career["seasons"] >= MIN_CAREER_SEASONS
    eligible = career[career["eligible"]]
    ranked = tiers_from_scores(eligible["career_score_pp"]).rename(
        columns={
            "rank": "career_rank",
            "group_size": "eligible_umpires",
            "percentile": "career_percentile",
            "tier": "career_tier",
        }
    )
    career = career.join(ranked)
    for column in ("career_rank", "eligible_umpires", "career_tier"):
        career[column] = career[column].astype("Int64")
    career["career_tier_name"] = career["career_tier"].map(TIER_NAMES)
    return career.sort_values(
        ["eligible", "career_score_pp"], ascending=[False, False]
    ).reset_index(drop=True)


def chance_shares(pairs: pd.DataFrame) -> tuple[float, float]:
    """Return the same-tier and within-one-tier shares luck would give."""
    share_from = pairs["tier_from"].value_counts(normalize=True)
    share_to = pairs["tier_to"].value_counts(normalize=True)
    same = 0.0
    within_one = 0.0
    for tier_from, p_from in share_from.items():
        for tier_to, p_to in share_to.items():
            gap = abs(int(tier_from) - int(tier_to))
            if gap == 0:
                same += p_from * p_to
            if gap <= 1:
                within_one += p_from * p_to
    return same, within_one


def stability_row(label: str, pairs: pd.DataFrame) -> dict:
    """Summarise how well tiers in one season predict the next."""
    same_by_chance, within_one_by_chance = chance_shares(pairs)
    crosses = bool(
        (pairs["season_from"] < NEW_ZONE_FIRST_SEASON).any()
        and (pairs["season_to"] >= NEW_ZONE_FIRST_SEASON).any()
    )
    return {
        "seasons": label,
        "includes_old_to_new_zone": crosses,
        "umpires": len(pairs),
        "same_tier_share": float((pairs["step"] == 0).mean()),
        "same_tier_by_chance": same_by_chance,
        "within_one_tier_share": float((pairs["step"] <= 1).mean()),
        "within_one_tier_by_chance": within_one_by_chance,
        "aal_correlation": pairs["aal_from"].corr(pairs["aal_to"]),
    }


def stability_tables(
    ump: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each umpire's tier in one season with his tier the next."""
    pool = ump.loc[ump["qualifies"], [*UMPIRE_SEASON, "tier", "aal_shrunk_pp"]]
    following = pool.assign(season=pool["season"] - 1).rename(
        columns={"tier": "tier_to", "aal_shrunk_pp": "aal_to"}
    )
    pairs = pool.merge(following, on=UMPIRE_SEASON).rename(
        columns={
            "season": "season_from",
            "tier": "tier_from",
            "aal_shrunk_pp": "aal_from",
        }
    )
    pairs["season_to"] = pairs["season_from"] + 1
    pairs["step"] = (pairs["tier_to"] - pairs["tier_from"]).abs()

    transitions = (
        pairs.groupby(["season_from", "season_to", "tier_from", "tier_to"])
        .size()
        .reset_index(name="umpires")
    )
    rows = [
        stability_row(f"{season_from}-{season_to}", group)
        for (season_from, season_to), group in pairs.groupby(
            ["season_from", "season_to"]
        )
    ]
    rows.append(stability_row("all pairs", pairs))
    return transitions, pd.DataFrame(rows)


def build() -> dict[str, pd.DataFrame]:
    """Build every table, keyed by the name of the CSV it is saved as."""
    by_game = load_game_table()
    ump = season_table(by_game)
    ump = add_league(ump)
    ump = add_sampling_error(ump, by_game)
    ump = add_shrinkage(ump)
    ump = add_tiers(ump)
    transitions, stability = stability_tables(ump)
    return {
        "umpire_game_accuracy": by_game,
        "umpire_tiers": ump,
        "tier_summary": tier_summary(ump),
        "umpire_career_tiers": career_table(ump),
        "tier_stability": transitions,
        "tier_stability_summary": stability,
    }


def save(tables: dict[str, pd.DataFrame]) -> None:
    """Write each table to reports/tables as a CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = REPORTS_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"saved {path.relative_to(PROJECT_DIR)}  ({len(table):,} rows)")


def main() -> None:
    """Build the tables, save them and print a short summary."""
    tables = build()
    save(tables)

    tiers = tables["umpire_tiers"]
    pool = tiers[tiers["qualifies"]]
    print(f"\numpire-seasons: {len(tiers)} | qualifying: {len(pool)}")
    overview = pool.groupby("season").agg(
        umpires=("official_id", "count"),
        league_accuracy=("league_accuracy", "first"),
        skill_sd_pp=("skill_sd_pp", "first"),
        typical_se_pp=("se_game_pp", "median"),
        typical_shrink=("shrink_factor", "median"),
    )
    print(overview.round(4).to_string())

    latest = pool[pool["season"] == pool["season"].max()]
    for tier in (1, 5):
        names = latest.loc[latest["tier"] == tier, "official_name"]
        print(f"\nTier {tier} ({TIER_NAMES[tier]}): " + ", ".join(names))
    print()
    print(tables["tier_stability_summary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
