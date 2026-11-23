"""One row per umpire per season, holding every number the study has.

Tiers and career stages are labels added on top of these numbers; nothing
is dropped to make them. Umpires with fewer than 10 plate games stay in
the table, flagged and without a tier. Each season is graded against its
own strike zone and nothing is standardised.

Where the columns come from
- umpire_tiers.py: accuracy, accuracy above the league, tier, career tier.
- fan_metrics.py: misses per season and misses against the league.
- umpire_profiles.py: false strikes and false balls, lean, zone shape and
  accuracy in every split.
- data/tables/umpire_assignments: games worked at any base.
- data/tables/postseason_umpire_assignments: postseason games worked.
- data/tables/challenges: ABS challenges against the umpire (2026 only).
- reports/tables/umpire_career_stage.csv (Jorge's table): age, years since
  debut and career stage. It is added when the file is there. Until then
  the master table is built without it.

Tables (reports/tables)
- umpire_master.csv: the master table.
- umpire_master_columns.csv: what every column means and where it is from.
- accuracy_by_stage.csv and stage_by_tier.csv: only once the career stage
  table exists.

Run from the project folder:  python src/umpire_master.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import fan_metrics
import umpire_profiles
import umpire_tiers
from umpire_tiers import NEW_ZONE_FIRST_SEASON, UMPIRE_SEASON

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
STAGE_PATH = PROJECT_DIR / "reports" / "tables" / "umpire_career_stage.csv"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 2026
TOP_QUARTILE = (1, 2)
BOTTOM_QUARTILE = (4, 5)
STAGE_KEYS = ("season", "official_id", "stage")
FAN_COLUMNS = [
    "misses_per_season",
    "misses_vs_league",
    "calls_to_reach_league",
]
CAREER_COLUMNS = {
    "seasons": "career_qualifying_seasons",
    "eligible": "career_eligible",
    "career_score_pp": "career_score_pp",
    "career_se_pp": "career_se_pp",
    "career_rank": "career_rank",
    "career_percentile": "career_percentile",
    "career_tier": "career_tier",
    "career_tier_name": "career_tier_name",
    "old_zone_score_pp": "career_old_zone_score_pp",
    "new_zone_score_pp": "career_new_zone_score_pp",
}
SOURCES = {
    "tiers": "umpire_tiers.py",
    "fan": "fan_metrics.py",
    "profiles": "umpire_profiles.py",
    "games": "data/tables/umpire_assignments",
    "postseason": "data/tables/postseason_umpire_assignments",
    "challenges": "data/tables/challenges",
    "career": "umpire_tiers.py (career table)",
    "stage": "reports/tables/umpire_career_stage.csv (Jorge)",
}
REGION_PICKS = {
    "worst": "the region where his accuracy is furthest below the league",
    "best": "the region where his accuracy is furthest above the league",
    "widest": "the region where he calls the most extra strikes",
    "tightest": "the region where he calls the most extra balls",
}
REGION_FIELDS = {
    "region": "Name of {pick}",
    "region_calls": "Called pitches in {pick}",
    "region_accuracy": "His accuracy in {pick}",
    "region_gap_pp": "His accuracy minus the league's, in points, in {pick}",
    "region_z": "Size of the gap against the noise in {pick}",
    "region_extra_misses": "Calls missed beyond a league-average umpire in "
    "{pick}",
    "region_per_game": "Extra strike calls per game against the league in "
    "{pick}",
    "region_secure": "True if it still stands out after allowing for the "
    "nine regions looked through: {pick}",
}
MEANINGS = {
    "season": "Season",
    "zone": "Zone in force: old (2021-2025) or new (ABS zone, 2026)",
    "official_id": "MLB id of the umpire",
    "official_name": "Name of the umpire",
    "qualifies": "True with 10 or more plate games that season",
    "plate_games": "Games behind the plate with graded calls",
    "called_pitches": "Ball and strike calls graded",
    "correct": "Correct calls",
    "missed": "Missed calls",
    "accuracy": "Share of calls that were correct",
    "ci_low": "Low end of the 95% Wilson interval of accuracy",
    "ci_high": "High end of the 95% Wilson interval of accuracy",
    "league_accuracy": "Accuracy of all qualifying umpires that season",
    "aal_pp": "Accuracy above league, in points",
    "se_game_pp": "Standard error of accuracy from game-to-game variation",
    "se_binomial_pp": "Standard error if every pitch were independent",
    "skill_sd_pp": "How much true skill differs across umpires that season",
    "shrink_factor": "Share of the gap to the league that is kept (0 to 1)",
    "aal_shrunk_pp": "Accuracy above league after shrinking for noise",
    "shrunk_low_pp": "Low end of the 95% interval of the shrunken value",
    "shrunk_high_pp": "High end of the 95% interval of the shrunken value",
    "secure": "True if that interval does not include the league",
    "rank_in_season": "Rank among qualifying umpires (1 = best)",
    "umpires_in_season": "Qualifying umpires that season",
    "percentile": "Percentile among qualifying umpires (100 = best)",
    "tier": "Performance tier, 1 (top 10%) to 5 (bottom 10%)",
    "tier_name": "Name of the tier",
    "misses_per_game": "Missed calls per plate game",
    "misses_per_1000": "Missed calls per 1,000 calls",
    "game_sd_pp": "Spread of his accuracy from game to game, in points",
    "best_game": "Accuracy in his best game",
    "worst_game": "Accuracy in his worst game",
    "misses_per_season": "Missed calls over a full season of 29 plate games",
    "misses_vs_league": "Calls missed beyond a league-average umpire on the "
    "same number of calls (negative = fewer)",
    "calls_to_reach_league": "More correct calls needed to reach the league "
    "average (0 if already there)",
    "true_balls": "Called pitches that were truly balls",
    "true_strikes": "Called pitches that were truly strikes",
    "false_strikes": "True balls he called strikes",
    "false_balls": "True strikes he called balls",
    "false_strike_rate": "False strikes as a share of true balls",
    "false_strike_low": "Low end of its 95% Wilson interval",
    "false_strike_high": "High end of its 95% Wilson interval",
    "false_ball_rate": "False balls as a share of true strikes",
    "false_ball_low": "Low end of its 95% Wilson interval",
    "false_ball_high": "High end of its 95% Wilson interval",
    "false_strikes_per_game": "False strikes per plate game",
    "false_balls_per_game": "False balls per plate game",
    "net_strikes_per_1000": "False strikes minus false balls per 1,000 calls "
    "(positive = more strikes called than the zone has)",
    "league_false_strike_rate": "False-strike rate of qualifying umpires",
    "league_false_ball_rate": "False-ball rate of qualifying umpires",
    "league_net_strikes_per_1000": "Net strikes of qualifying umpires",
    "false_strike_gap_pp": "His false-strike rate minus the league's",
    "false_ball_gap_pp": "His false-ball rate minus the league's",
    "lean_per_1000": "His net strikes minus the league's, per 1,000 calls "
    "(positive = leans to strikes)",
    "lean_se_per_1000": "Standard error of the lean, from his games",
    "lean_low_per_1000": "Low end of the 95% interval of the lean",
    "lean_high_per_1000": "High end of the 95% interval of the lean",
    "lean_per_game": "The lean in strike calls per game",
    "lean_low_per_game": "Low end of its 95% interval, per game",
    "lean_high_per_game": "High end of its 95% interval, per game",
    "lean_systematic": "True if the interval of the lean excludes zero",
    "lean_side": "strikes, balls, no clear lean, or under 10 plate games",
    "bias_log_ratio": "Log of false-strike rate over false-ball rate, minus "
    "the league's (the lean as a ratio)",
    "bias_se": "Standard error of the log ratio, from his games",
    "bias_low": "Low end of its 95% interval",
    "bias_high": "High end of its 95% interval",
    "bias_systematic": "True if that interval excludes zero",
    "profile_note": "One-sentence profile using only firm results",
    "games_worked": "Regular-season games worked at any base",
    "plate_share_of_games": "Plate games as a share of games worked",
    "postseason_games": "Postseason games worked at any base that season",
    "postseason_plate_games": "Postseason games behind the plate",
    "worked_postseason": "True if he worked any postseason game that season",
    "challenges": "ABS challenges against his calls (2026 only)",
    "overturned": "Challenges that overturned his call",
    "overturn_rate": "Share of challenges that overturned his call",
    "overturn_low": "Low end of its 95% Wilson interval",
    "overturn_high": "High end of its 95% Wilson interval",
    "challenges_per_game": "Challenges per plate game",
    "overturns_per_game": "Overturned calls per plate game",
    "challenges_per_100_calls": "Challenges per 100 called pitches",
    "career_qualifying_seasons": "Seasons with 10 or more plate games",
    "career_eligible": "True with 3 or more qualifying seasons",
    "career_score_pp": "Mean shrunken accuracy above league, all seasons",
    "career_se_pp": "Standard error of the career score",
    "career_rank": "Career rank among eligible umpires (1 = best)",
    "career_percentile": "Career percentile among eligible umpires",
    "career_tier": "Career tier, 1 to 5, same cuts as the season tiers",
    "career_tier_name": "Name of the career tier",
    "career_old_zone_score_pp": "Career score over 2021-2025 only",
    "career_new_zone_score_pp": "Career score in 2026 only",
}


def games_worked() -> pd.DataFrame:
    """Count each umpire's regular-season games at any base, by season."""
    assignments = pd.read_parquet(
        TABLES_DIR / "umpire_assignments" / "all_seasons.parquet",
        columns=["season", "game_pk", "official_id"],
    ).dropna()
    worked = assignments.groupby(UMPIRE_SEASON)["game_pk"].nunique()
    worked = worked.rename("games_worked").reset_index()
    return worked.astype({"season": "int64", "official_id": "int64"})


def postseason_work() -> pd.DataFrame:
    """Count each umpire's postseason games: any base and at the plate."""
    assignments = pd.read_parquet(
        TABLES_DIR / "postseason_umpire_assignments" / "all_seasons.parquet",
        columns=["season", "game_pk", "official_id", "official_type"],
    ).dropna()
    assignments["at_plate"] = assignments["official_type"].eq("Home Plate")
    worked = (
        assignments.groupby(UMPIRE_SEASON)
        .agg(
            postseason_games=("game_pk", "nunique"),
            postseason_plate_games=("at_plate", "sum"),
        )
        .reset_index()
    )
    return worked.astype({"season": "int64", "official_id": "int64"})


def challenge_counts() -> pd.DataFrame:
    """Count the ABS challenges made against each plate umpire's calls."""
    challenges = pd.read_parquet(
        TABLES_DIR / "challenges" / "all_seasons.parquet",
        columns=["season", "game_pk", "is_overturned"],
    )
    umpires = umpire_tiers.load_plate_umpires()[["game_pk", "official_id"]]
    challenges = challenges.merge(umpires, on="game_pk", how="inner")
    challenges["is_overturned"] = challenges["is_overturned"].eq(True)
    counts = (
        challenges.groupby(UMPIRE_SEASON)["is_overturned"]
        .agg(challenges="size", overturned="sum")
        .reset_index()
    )
    counts["overturn_rate"] = counts["overturned"] / counts["challenges"]
    counts["overturn_low"], counts["overturn_high"] = (
        umpire_profiles.wilson_columns(
            counts["overturned"], counts["challenges"]
        )
    )
    return counts


def load_stages(path: Path) -> pd.DataFrame | None:
    """Return Jorge's career stage table, or None if it is not there yet."""
    if not path.exists():
        return None
    stages = pd.read_csv(path)
    missing = set(STAGE_KEYS) - set(stages.columns)
    if missing:
        raise ValueError(f"{path.name} has no column {sorted(missing)}")
    if stages.duplicated(UMPIRE_SEASON).any():
        raise ValueError(f"{path.name} repeats an umpire in one season")
    return stages


def add_columns(
    master: pd.DataFrame, extra: pd.DataFrame, source: str, sources: dict
) -> pd.DataFrame:
    """Join the new columns of another table and note where they came from.

    A column the master already has is not overwritten. The other
    table's version is kept beside it with the source's name at the end,
    so a disagreement between two tables stays visible.
    """
    clashes = [
        name
        for name in extra.columns
        if name in master.columns and name not in UMPIRE_SEASON
    ]
    extra = extra.rename(
        columns={name: f"{name}_{source}" for name in clashes}
    )
    new = [name for name in extra.columns if name not in UMPIRE_SEASON]
    sources.update(dict.fromkeys(new, SOURCES[source]))
    return master.merge(extra, on=UMPIRE_SEASON, how="left")


def add_career(
    master: pd.DataFrame, career: pd.DataFrame, sources: dict
) -> pd.DataFrame:
    """Repeat each umpire's career columns on every one of his seasons."""
    career = career[["official_id", *CAREER_COLUMNS]].rename(
        columns=CAREER_COLUMNS
    )
    sources.update(dict.fromkeys(CAREER_COLUMNS.values(), SOURCES["career"]))
    master = master.merge(career, on="official_id", how="left")
    master["career_eligible"] = master["career_eligible"].eq(True)
    return master


def master_table(
    tier_tables: dict[str, pd.DataFrame],
    profiles: pd.DataFrame,
    stages: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    """Join every source into one row per umpire per season."""
    tiers = tier_tables["umpire_tiers"]
    sources = dict.fromkeys(tiers.columns, SOURCES["tiers"])
    master = tiers.copy()

    fan = fan_metrics.umpire_metrics(tiers)[[*UMPIRE_SEASON, *FAN_COLUMNS]]
    master = add_columns(master, fan, "fan", sources)
    only_profile = [
        name
        for name in profiles.columns
        if name not in master.columns or name in UMPIRE_SEASON
    ]
    master = add_columns(master, profiles[only_profile], "profiles", sources)

    master = add_columns(master, games_worked(), "games", sources)
    master["plate_share_of_games"] = (
        master["plate_games"] / master["games_worked"]
    )
    sources["plate_share_of_games"] = SOURCES["games"]

    master = add_columns(master, postseason_work(), "postseason", sources)
    played = ["postseason_games", "postseason_plate_games"]
    master[played] = master[played].fillna(0).astype(int)
    master["worked_postseason"] = master["postseason_games"] > 0
    sources["worked_postseason"] = SOURCES["postseason"]

    master = add_columns(master, challenge_counts(), "challenges", sources)
    per_game = {
        "challenges_per_game": master["challenges"] / master["plate_games"],
        "overturns_per_game": master["overturned"] / master["plate_games"],
        "challenges_per_100_calls": master["challenges"]
        / master["called_pitches"]
        * 100,
    }
    master = master.assign(**per_game)
    sources.update(dict.fromkeys(per_game, SOURCES["challenges"]))

    master = add_career(master, tier_tables["umpire_career_tiers"], sources)
    if stages is not None:
        master = add_columns(master, stages, "stage", sources)
    return master, sources


def meaning(column: str) -> str:
    """Say in plain words what a column of the master table holds."""
    if column in MEANINGS:
        return MEANINGS[column]
    for pick, words in REGION_PICKS.items():
        field = column.removeprefix(f"{pick}_")
        if column.startswith(f"{pick}_") and field in REGION_FIELDS:
            return REGION_FIELDS[field].format(pick=words)
    cells = {
        name: f"{split} = {level}"
        for (split, level), name in umpire_profiles.PROFILE_LEVELS.items()
    }
    if column.startswith("acc_"):
        cell = cells[column.removeprefix("acc_")]
        return f"His accuracy where {cell} (blank under 100 calls)"
    if column.startswith("gap_"):
        cell = cells[column.removeprefix("gap_").removesuffix("_pp")]
        return f"His accuracy minus the league's, in points, where {cell}"
    if column.startswith("zone_"):
        cell = cells[column.removeprefix("zone_").removesuffix("_per_game")]
        return (
            "Extra strike calls per game against the league where "
            f"{cell} (negative = extra ball calls)"
        )
    return ""


def column_guide(master: pd.DataFrame, sources: dict) -> pd.DataFrame:
    """List every column of the master table: meaning and source."""
    guide = pd.DataFrame(
        {
            "column": master.columns,
            "meaning": [meaning(name) for name in master.columns],
            "source": [sources[name] for name in master.columns],
        }
    )
    from_stage = guide["source"] == SOURCES["stage"]
    guide.loc[from_stage & (guide["meaning"] == ""), "meaning"] = (
        "From the career stage table"
    )
    unexplained = guide.loc[guide["meaning"] == "", "column"].tolist()
    if unexplained:
        raise ValueError(f"columns without a meaning: {unexplained}")
    return guide


def with_periods(master: pd.DataFrame) -> pd.DataFrame:
    """Stack the qualifying rows by season, plus the old zone as a whole."""
    pool = master[master["qualifies"] & master["stage"].notna()]
    old = sorted(
        season
        for season in pool["season"].unique()
        if season < NEW_ZONE_FIRST_SEASON
    )
    pieces = [pool.assign(period=pool["season"].astype(str))]
    if len(old) > 1:
        label = f"{old[0]}-{old[-1]} (old zone)"
        pieces.append(pool[pool["season"].isin(old)].assign(period=label))
    return pd.concat(pieces, ignore_index=True)


def mean_interval(
    rows: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a 95% interval for the mean accuracy above league.

    Umpires are resampled whole, so the seasons of one umpire stay
    together and are not counted as separate evidence.
    """
    by_umpire = rows.groupby("official_id")["aal_pp"].agg(["sum", "count"])
    total = by_umpire["sum"].to_numpy()
    count = by_umpire["count"].to_numpy()
    picks = rng.integers(
        0, len(by_umpire), size=(BOOTSTRAP_REPS, len(by_umpire))
    )
    means = total[picks].sum(axis=1) / count[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def accuracy_by_stage(master: pd.DataFrame) -> pd.DataFrame:
    """Return accuracy and accuracy above league for each career stage.

    One row per stage for every season, and for the old-zone seasons
    added together. Accuracy above league is each umpire against the
    league of his own season, so adding seasons together does not mix
    the two zones.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (period, stage), group in with_periods(master).groupby(
        ["period", "stage"], sort=True
    ):
        called = int(group["called_pitches"].sum())
        correct = int(group["correct"].sum())
        low, high = umpire_profiles.wilson_interval(correct, called)
        aal_low, aal_high = mean_interval(group, rng)
        rows.append(
            {
                "period": period,
                "stage": stage,
                "umpire_seasons": len(group),
                "umpires": group["official_id"].nunique(),
                "plate_games": int(group["plate_games"].sum()),
                "called_pitches": called,
                "accuracy": correct / called,
                "ci_low": low,
                "ci_high": high,
                "mean_aal_pp": group["aal_pp"].mean(),
                "mean_aal_low_pp": aal_low,
                "mean_aal_high_pp": aal_high,
                "misses_per_game": (called - correct)
                / group["plate_games"].sum(),
                "share_top_quartile": group["tier"].isin(TOP_QUARTILE).mean(),
                "share_bottom_quartile": group["tier"]
                .isin(BOTTOM_QUARTILE)
                .mean(),
            }
        )
    return pd.DataFrame(rows)


def stage_by_tier(master: pd.DataFrame) -> pd.DataFrame:
    """Count the umpires of each career stage in each performance tier."""
    counts = (
        with_periods(master)
        .groupby(["period", "stage", "tier", "tier_name"])
        .size()
        .reset_index(name="umpire_seasons")
    )
    in_stage = counts.groupby(["period", "stage"])["umpire_seasons"]
    counts["share_of_stage"] = counts["umpire_seasons"] / in_stage.transform(
        "sum"
    )
    return counts


def build(stage_path: Path = STAGE_PATH) -> dict[str, pd.DataFrame]:
    """Build every table, keyed by the name of the CSV it is saved as."""
    tier_tables = umpire_tiers.build()
    profiles = umpire_profiles.build(tier_tables["umpire_tiers"])
    stages = load_stages(stage_path)
    master, sources = master_table(
        tier_tables, profiles["umpire_profiles"], stages
    )
    tables = {
        "umpire_master": master,
        "umpire_master_columns": column_guide(master, sources),
    }
    if stages is not None:
        tables["accuracy_by_stage"] = accuracy_by_stage(master)
        tables["stage_by_tier"] = stage_by_tier(master)
    return tables


def main() -> None:
    """Build the tables, save them and print a short summary."""
    tables = build()
    umpire_tiers.save(tables)

    master = tables["umpire_master"]
    print(f"\nmaster table: {len(master)} rows, {master.shape[1]} columns")
    by_source = tables["umpire_master_columns"]["source"].value_counts()
    print(by_source.to_string())
    if "accuracy_by_stage" in tables:
        print()
        print(tables["accuracy_by_stage"].round(4).to_string(index=False))
    else:
        name = STAGE_PATH.relative_to(PROJECT_DIR)
        print(f"\nno career stage table yet ({name}): built without it")


if __name__ == "__main__":
    main()
