"""Collect every number the slides use into one file.

Reads the frozen tables in reports/tables (plus the challenges table) and
writes reports/presentation/deck_data.json. The deck is built from that
file, so a number on a slide can always be traced back to a table.

Run the scripts that make the tables first (umpire_tiers, fan_metrics,
accuracy_by_edge, umpire_profiles, umpire_master), then:

    python src/deck_data.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"
REPORTS_DIR = PROJECT_DIR / "reports"
CSV_DIR = REPORTS_DIR / "tables"
OUTPUT_PATH = REPORTS_DIR / "presentation" / "deck_data.json"

NEW_ZONE_FIRST_SEASON = 2026
TOP_TIER = 1
BOTTOM_TIER = 5
LISTED = 9
BALL_RADIUS_IN = 1.45
HALF_PLATE_IN = 8.5
EDGE_FAMILIES = ("sides", "top", "bottom")
CARRY_OVER_TRAITS = (
    "aal_pp",
    "lean_per_1000",
    "zone_mid_outside_per_game",
    "zone_high_middle_per_game",
    "gap_fastball_pp",
    "gap_breaking_pp",
    "gap_two_strikes_pp",
    "gap_three_balls_pp",
)


def table(name: str) -> pd.DataFrame:
    """Read one of the saved tables in reports/tables."""
    return pd.read_csv(CSV_DIR / f"{name}.csv")


def records(frame: pd.DataFrame) -> list[dict]:
    """Turn a table into plain rows that can be written as JSON."""
    return json.loads(frame.to_json(orient="records"))


def seasons_block() -> list[dict]:
    """Return accuracy, missed calls and the two kinds of miss by season."""
    league = table("league_missed_calls_by_season")
    splits = table("league_split_accuracy")
    kinds = splits.loc[
        splits["split"] == "all calls",
        [
            "season",
            "false_strikes",
            "false_balls",
            "false_strike_rate",
            "false_ball_rate",
            "net_strikes_per_1000",
        ],
    ]
    return records(league.merge(kinds, on="season"))


def tiers_block() -> dict:
    """Return the latest season's tiers and the checks that back them."""
    tiers = table("umpire_tiers")
    season = int(tiers["season"].max())
    pool = tiers[(tiers["season"] == season) & tiers["qualifies"]]
    summary = table("tier_summary")
    gaps = table("tier_gap_intervals")
    fan = table("fan_metrics_by_umpire")
    fan = fan[fan["season"] == season]
    bottom_quarter = fan[fan["tier"] >= BOTTOM_TIER - 1]
    members = [
        "rank_in_season",
        "official_name",
        "plate_games",
        "accuracy",
        "aal_pp",
        "shrink_factor",
        "aal_shrunk_pp",
        "shrunk_low_pp",
        "shrunk_high_pp",
        "secure",
    ]
    return {
        "season": season,
        "umpires": len(pool),
        "league_accuracy": float(pool["league_accuracy"].iloc[0]),
        "skill_sd_pp": float(pool["skill_sd_pp"].iloc[0]),
        "typical_se_pp": float(pool["se_game_pp"].median()),
        "typical_shrink": float(pool["shrink_factor"].median()),
        "summary": records(summary[summary["season"] == season]),
        "tier_1": records(pool.loc[pool["tier"] == TOP_TIER, members]),
        "tier_5": records(pool.loc[pool["tier"] == BOTTOM_TIER, members]),
        "next_season": records(table("tier_next_season")),
        "gap_first": records(gaps[gaps["season"] == gaps["season"].min()])[0],
        "gap_latest": records(gaps[gaps["season"] == season])[0],
        "below_league": int((fan["misses_vs_league"] > 0).sum()),
        "bottom_quarter": {
            "umpires": len(bottom_quarter),
            "fewest_calls_needed": int(
                bottom_quarter["calls_to_reach_league"].min()
            ),
            "most_calls_needed": int(
                bottom_quarter["calls_to_reach_league"].max()
            ),
            "mean_calls_needed": float(
                bottom_quarter["calls_to_reach_league"].mean()
            ),
        },
    }


def career_block() -> dict:
    """Return the top and bottom of the career table (3+ seasons)."""
    career = table("umpire_career_tiers")
    career = career[career["eligible"]].sort_values("career_rank")
    columns = [
        "career_rank",
        "official_name",
        "seasons",
        "career_score_pp",
        "career_se_pp",
        "old_zone_score_pp",
        "new_zone_score_pp",
        "best_tier",
        "worst_tier",
        "tiers_by_season",
    ]
    return {
        "eligible": len(career),
        "top": records(career[columns].head(LISTED)),
        "bottom": records(career[columns].tail(LISTED)),
    }


def postseason_block() -> list[dict]:
    """Compare umpires who worked the postseason with those who did not."""
    master = table("umpire_master")
    pool = master[master["qualifies"]]
    sums = (
        pool.groupby(["season", "worked_postseason"])
        .agg(
            umpires=("official_id", "count"),
            called=("called_pitches", "sum"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    sums["accuracy"] = sums["correct"] / sums["called"]
    wide = sums.pivot(
        index="season",
        columns="worked_postseason",
        values=["umpires", "accuracy"],
    )
    rows = pd.DataFrame(
        {
            "season": wide.index,
            "postseason_umpires": wide[("umpires", True)].to_numpy(),
            "other_umpires": wide[("umpires", False)].to_numpy(),
            "postseason_accuracy": wide[("accuracy", True)].to_numpy(),
            "other_accuracy": wide[("accuracy", False)].to_numpy(),
        }
    )
    rows["gap_pp"] = (
        rows["postseason_accuracy"] - rows["other_accuracy"]
    ) * 100
    return records(rows)


def profiles_block() -> dict:
    """Return each umpire's lean and accuracy, and the example umpires."""
    profiles = table("umpire_profiles")
    season = int(profiles["season"].max())
    pool = profiles[(profiles["season"] == season) & profiles["qualifies"]]
    stability = table("profile_stability").set_index("trait")
    points = [
        "official_id",
        "official_name",
        "tier",
        "accuracy",
        "lean_per_game",
        "lean_side",
    ]
    carry = ["all_correlation", "beyond_accuracy_correlation"]
    return {
        "season": season,
        "points": records(pool[points]),
        "showcase": records(table("showcase_umpires")),
        "lean_sides": pool["lean_side"].value_counts().to_dict(),
        "lean_sides_bottom_tiers": pool.loc[
            pool["tier"] >= BOTTOM_TIER - 1, "lean_side"
        ]
        .value_counts()
        .to_dict(),
        "lean_sides_top_tiers": pool.loc[
            pool["tier"] <= TOP_TIER + 1, "lean_side"
        ]
        .value_counts()
        .to_dict(),
        "lean_range_per_game": [
            float(pool["lean_per_game"].min()),
            float(pool["lean_per_game"].max()),
        ],
        "zone_shape_umpires": int(
            (
                pool["widest_region_secure"] | pool["tightest_region_secure"]
            ).sum()
        ),
        "carry_over": json.loads(
            stability.loc[list(CARRY_OVER_TRAITS), carry].to_json(
                orient="index"
            )
        ),
    }


def challenges_block() -> dict:
    """Return the challenge numbers: by umpire, by role and unused."""
    master = table("umpire_master")
    pool = master[master["qualifies"] & master["challenges"].notna()]
    challenges = pd.read_parquet(
        TABLES_DIR / "challenges" / "all_seasons.parquet",
        columns=["challenger_role", "is_overturned"],
    )
    roles = (
        challenges.assign(won=challenges["is_overturned"].eq(True))
        .groupby("challenger_role")["won"]
        .agg(challenges="size", won="sum")
        .reset_index()
    )
    roles["rate"] = roles["won"] / roles["challenges"]
    teams = table("team_challengers_2026").sort_values("rate")
    unused = table("unused_challenges_2026")
    points = ["official_name", "accuracy", "overturn_rate", "challenges"]
    return {
        "challenges": len(challenges),
        "overturned": int(challenges["is_overturned"].eq(True).sum()),
        "umpires": len(pool),
        "correlation": float(pool["accuracy"].corr(pool["overturn_rate"])),
        "points": records(pool[points]),
        "roles": records(roles),
        "best_team": records(teams.tail(1))[0],
        "worst_team": records(teams.head(1))[0],
        "team_games": int(unused["team_games"].sum()),
        "team_games_with_one_left": int(
            unused.loc[unused["left"] > 0, "team_games"].sum()
        ),
    }


def edges_block() -> dict:
    """Return accuracy by the edge that decides the call, and by distance."""
    by_season = table("accuracy_by_edge_season")
    families = by_season[
        (by_season["level"] == "edge_family")
        & by_season["name"].isin(EDGE_FAMILIES)
    ]
    change = table("accuracy_by_edge_change")
    change = change[
        (change["level"] == "edge_family") & change["name"].isin(EDGE_FAMILIES)
    ]
    regions = by_season[by_season["level"] == "zone_region"]
    splits = table("league_split_accuracy")
    line = splits[splits["split"] == "line"]
    keep = ["season", "level", "called", "share_of_called", "accuracy"]
    return {
        "families": records(families[["name", "rule", "season", "accuracy"]]),
        "change": records(change),
        "regions": records(regions[["name", "season", "called", "accuracy"]]),
        "line": records(line[keep]),
    }


def count_block() -> dict:
    """Return how often a pitch on the line is a strike, by the count."""
    splits = table("league_split_accuracy")
    rows = splits[
        (splits["split"] == "count_situation") & (splits["level"] != "unknown")
    ]
    rows = rows.assign(
        zone=rows["season"]
        .ge(NEW_ZONE_FIRST_SEASON)
        .map({True: "new", False: "old"})
    )
    sums = rows.groupby(["zone", "level"])[
        ["near_line", "near_line_strikes"]
    ].sum()
    sums["strike_rate"] = sums["near_line_strikes"] / sums["near_line"]
    return {"near_line": records(sums.reset_index())}


def zone_block() -> list[dict]:
    """Set the called zone beside each season's official limit.

    The limit for the centre of the ball is the recorded edge plus the
    ball's radius. Positive numbers mean the called edge is past the
    limit (a bigger called zone). From Jorge's J3 table.
    """
    edges = table("j3_called_zone_edges")
    rows = pd.DataFrame({"season": edges["season"]})
    rows["official_top_in"] = edges["median_recorded_zone_top_in"]
    rows["official_bottom_in"] = edges["median_recorded_zone_bottom_in"]
    rows["called_top_in"] = edges["called_zone_top_edge_in"]
    rows["called_bottom_in"] = edges["called_zone_bottom_edge_in"]
    rows["top_past_limit_in"] = rows["called_top_in"] - (
        rows["official_top_in"] + BALL_RADIUS_IN
    )
    rows["bottom_past_limit_in"] = (
        rows["official_bottom_in"] - BALL_RADIUS_IN
    ) - rows["called_bottom_in"]
    half_width = (
        edges["called_zone_right_edge_in"] - edges["called_zone_left_edge_in"]
    ) / 2
    rows["side_past_limit_in"] = half_width - (HALF_PLATE_IN + BALL_RADIUS_IN)
    return records(rows)


def validation_block() -> dict:
    """Return the size of the data and the check against ABS rulings."""
    tiers = table("umpire_tiers")
    check = pd.read_csv(REPORTS_DIR / "grading_validation_2026.csv").iloc[0]
    challenged = int(check["challenged_pitches"])
    return {
        "graded_calls": int(
            table("league_missed_calls_by_season")["called_pitches"].sum()
        ),
        "umpire_games": len(table("umpire_game_accuracy")),
        "umpire_seasons": len(tiers),
        "qualifying_umpire_seasons": int(tiers["qualifies"].sum()),
        "abs_challenged_pitches": challenged,
        "abs_agreement": float(check["agreement"]),
        "abs_agreeing_pitches": round(challenged * float(check["agreement"])),
    }


def build() -> dict:
    """Collect every block of numbers the slides use."""
    return {
        "seasons": seasons_block(),
        "tiers": tiers_block(),
        "career": career_block(),
        "postseason": postseason_block(),
        "profiles": profiles_block(),
        "challenges": challenges_block(),
        "edges": edges_block(),
        "count": count_block(),
        "zone": zone_block(),
        "validation": validation_block(),
    }


def main() -> None:
    """Write the numbers to reports/presentation/deck_data.json."""
    data = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=1))
    print(f"saved {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    for name, block in data.items():
        print(f"  {name}: {len(block)} entries")


if __name__ == "__main__":
    main()
