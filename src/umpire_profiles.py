"""What each umpire struggles with: one error profile per umpire-season.

Every season is graded against its own strike zone (old zone 2021-2025,
new ABS zone 2026), and an umpire is only compared with the umpires who
called the same zone in the same season. Nothing is standardised.

Two kinds of miss
- false strike: a true ball called a strike. It helps the pitcher.
- false ball: a true strike called a ball. It helps the hitter.

The lean
Net strikes are false strikes minus false balls. That is the same as the
strikes an umpire called minus the strikes the zone says he should have
called, so a positive number means he calls more strikes than the zone
has. The lean is his net strikes minus the league's, per 1,000 calls and
per game. Its 95% interval is built from how much his lean moves from
game to game, and "systematic" means the interval does not include zero.
bias_log_ratio tells the same story as a ratio of the two error rates.

Zone shape
The same sum inside one region: how many more strikes an umpire called
there than the league's habits would give on the same pitches, per game
(zone_..._per_game). Positive means his zone is bigger there, negative
means it is tighter.

The splits
- zone_region: thirds of the zone by height and by width, plus "corner"
  (off the plate and above or below the zone), from grade_pitches.py.
- count_situation: first pitch (0-0), two strikes (0-2, 1-2, 2-2), three
  balls (3-0, 3-1), full count (3-2) and other counts. The count before
  a pitch is the count after it minus the final call on that pitch.
- pitch_group: fastball (FF, SI, FC), breaking (SL, ST, SV, CU, KC, CS),
  offspeed (CH, FS, FO, SC) and other.
- batter: left-handed or right-handed batter.
- line: "borderline" is a pitch whose centre is within 2 inches of the
  line where the right call flips from strike to ball; "clear" is the
  rest. The league table also has the distance in 1-inch steps.

The league beside each umpire is the umpires with 10 or more plate games
that season, the same pool as the tiers. A cell with fewer than 100 calls
stays in the long table, flagged, and is left blank in the profile row.

Tables (reports/tables)
- umpire_profiles.csv: one row per umpire per season.
- umpire_split_accuracy.csv: one row per umpire, season, split and level.
- league_split_accuracy.csv: all umpires together, every split, by season.
- profile_stability.csv: does a trait show up again the next season,
  and is it more than overall accuracy seen from another angle?
- showcase_umpires.csv: four example umpires of the latest season.

Figures (reports/figures)
- umpire_lean_vs_accuracy.png
- borderline_calls_by_count.png

Run from the project folder:  python src/umpire_profiles.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import plot_style
import umpire_tiers
from stats_helpers import wilson_interval
from umpire_tiers import NEW_ZONE_FIRST_SEASON, UMPIRE_SEASON, Z_95

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_DIR / "data" / "tables"

MIN_CELL_CALLS = 100
# The right call flips from strike to ball where the edge of the ball just
# touches the zone: one ball radius outside the rectangle. Same value as
# BALL_RADIUS_IN in grade_pitches.py.
LINE_IN = 1.45
BORDERLINE_IN = 2.0
NEAR_LINE_IN = 1.0
FARTHEST_STEP_IN = 4
MAX_BALLS = 3
MAX_STRIKES = 2
# Umpires get 99.9% of these right, so they cannot tell umpires apart.
HEART = "mid-middle"
# 95% confidence after looking through the nine other regions for the
# weakest one (0.05 / 9, two-sided).
Z_NINE_REGIONS = 2.77

PITCH_COLUMNS = [
    "season",
    "game_pk",
    "call_code_raw",
    "original_call_code",
    "is_strike_zone",
    "is_call_correct",
    "balls_after",
    "strikes_after",
    "bat_side",
    "pitch_type_raw",
    "distance_from_edge_in",
    "zone_region",
]
PITCH_GROUPS = {
    "FF": "fastball",
    "SI": "fastball",
    "FC": "fastball",
    "SL": "breaking",
    "ST": "breaking",
    "SV": "breaking",
    "CU": "breaking",
    "KC": "breaking",
    "CS": "breaking",
    "CH": "offspeed",
    "FS": "offspeed",
    "FO": "offspeed",
    "SC": "offspeed",
}
BATTER_SIDES = {"L": "left-handed batter", "R": "right-handed batter"}
MEASURES = [
    "called",
    "correct",
    "true_balls",
    "true_strikes",
    "false_strikes",
    "false_balls",
    "called_strikes",
    "near_line",
    "near_line_strikes",
    "near_line_true_strikes",
]
ERROR_COUNTS = ["true_balls", "true_strikes", "false_strikes", "false_balls"]
DISTANCE_STEPS = range(-FARTHEST_STEP_IN - 1, FARTHEST_STEP_IN + 1)


def distance_label(step: int) -> str:
    """Name a 1-inch step from the line, such as '0-1 in, ball side'."""
    side = "strike side" if step < 0 else "ball side"
    near = -step - 1 if step < 0 else step
    if near >= FARTHEST_STEP_IN:
        return f"over {FARTHEST_STEP_IN} in, {side}"
    return f"{near}-{near + 1} in, {side}"


# Every split and its levels, in the order they are shown.
SPLITS = {
    "zone_region": [
        "high-inside",
        "high-middle",
        "high-outside",
        "mid-inside",
        HEART,
        "mid-outside",
        "low-inside",
        "low-middle",
        "low-outside",
        "corner",
    ],
    "count_situation": [
        "three balls",
        "first pitch",
        "other counts",
        "full count",
        "two strikes",
    ],
    "count": [
        f"{balls}-{strikes}"
        for balls in range(MAX_BALLS + 1)
        for strikes in range(MAX_STRIKES + 1)
    ],
    "pitch_group": ["fastball", "breaking", "offspeed", "other"],
    "batter": list(BATTER_SIDES.values()),
    "line": ["borderline", "clear"],
    "distance": [distance_label(step) for step in DISTANCE_STEPS],
}
UMPIRE_SPLITS = (
    "zone_region",
    "count_situation",
    "pitch_group",
    "batter",
    "line",
)
# The cells that get their own columns in the profile row, with the short
# name used in those columns.
PROFILE_LEVELS = {
    **{
        ("zone_region", region): region.replace("-", "_")
        for region in SPLITS["zone_region"]
    },
    ("count_situation", "first pitch"): "first_pitch",
    ("count_situation", "two strikes"): "two_strikes",
    ("count_situation", "three balls"): "three_balls",
    ("count_situation", "full count"): "full_count",
    ("pitch_group", "fastball"): "fastball",
    ("pitch_group", "breaking"): "breaking",
    ("pitch_group", "offspeed"): "offspeed",
    ("batter", "left-handed batter"): "vs_left",
    ("batter", "right-handed batter"): "vs_right",
    ("line", "borderline"): "borderline",
    ("line", "clear"): "clear",
}
COUNT_LABELS = {
    "three balls": "Three balls\n(3-0, 3-1)",
    "first pitch": "First pitch\n(0-0)",
    "other counts": "Other counts",
    "full count": "Full count\n(3-2)",
    "two strikes": "Two strikes\n(0-2, 1-2, 2-2)",
}
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
    "aal_shrunk_pp",
    "secure",
    "rank_in_season",
    "umpires_in_season",
    "percentile",
    "tier",
    "tier_name",
    "misses_per_game",
]
SPLIT_COLUMNS = [
    "season",
    "official_id",
    "official_name",
    "qualifies",
    "split",
    "level",
    "called",
    "enough_calls",
    "correct",
    "accuracy",
    "ci_low",
    "ci_high",
    "league_accuracy",
    "gap_pp",
    "gap_se_pp",
    "gap_z",
    "extra_misses",
    "true_balls",
    "false_strikes",
    "false_strike_rate",
    "true_strikes",
    "false_balls",
    "false_ball_rate",
    "net_strikes_per_1000",
    "league_net_strikes_per_1000",
    "lean_per_1000",
    "extra_strikes",
    "extra_strikes_per_game",
    "lean_z",
]
ACCURACY_NAMES = {
    "level": "region",
    "called": "region_calls",
    "accuracy": "region_accuracy",
    "gap_pp": "region_gap_pp",
    "gap_z": "region_z",
    "extra_misses": "region_extra_misses",
}
LEAN_NAMES = {
    "level": "region",
    "extra_strikes_per_game": "region_per_game",
    "lean_z": "region_z",
}
TRAIT_GROUPS = {
    "aal_pp": "overall accuracy",
    "false_strike_gap_pp": "direction of misses",
    "false_ball_gap_pp": "direction of misses",
    "lean_per_1000": "direction of misses",
    "bias_log_ratio": "direction of misses",
}
# Where each showcase name is written, as an offset from its dot in points.
LABEL_SPOTS = {
    "best": {"xytext": (0, 10), "ha": "center", "va": "bottom"},
    "worst": {"xytext": (0, -10), "ha": "center", "va": "top"},
    "most strike-leaning": {"xytext": (-10, 0), "ha": "right", "va": "center"},
    "most ball-leaning": {"xytext": (0, 10), "ha": "center", "va": "bottom"},
}
LEAGUE_COLUMNS = [
    "season",
    "zone",
    "split",
    "level",
    "order",
    "called",
    "share_of_called",
    "correct",
    "accuracy",
    "ci_low",
    "ci_high",
    "true_balls",
    "false_strikes",
    "false_strike_rate",
    "true_strikes",
    "false_balls",
    "false_ball_rate",
    "net_strikes_per_1000",
    "called_strikes",
    "wrong_strike_share",
    "wrong_ball_share",
    "near_line",
    "near_line_strikes",
    "near_line_true_strikes",
    "near_line_strike_rate",
    "near_line_true_share",
]
SHOWCASE_COLUMNS = [
    "season",
    "official_id",
    "official_name",
    "tier",
    "tier_name",
    "rank_in_season",
    "plate_games",
    "accuracy",
    "aal_pp",
    "false_strike_rate",
    "false_ball_rate",
    "lean_per_game",
    "lean_low_per_game",
    "lean_high_per_game",
    "lean_per_1000",
    "lean_low_per_1000",
    "lean_high_per_1000",
    "lean_side",
    "widest_region",
    "widest_region_per_game",
    "widest_region_secure",
    "tightest_region",
    "tightest_region_per_game",
    "tightest_region_secure",
    "worst_region",
    "worst_region_gap_pp",
    "worst_region_secure",
    "profile_note",
]


def counts_before(pitches: pd.DataFrame) -> pd.DataFrame:
    """Work out the count before each pitch and name the situation.

    The feed gives the count after the pitch. Taking away the final call
    (one ball, or one called strike) gives the count the umpire faced.
    """
    final_strike = pitches["call_code_raw"].eq("C")
    balls = pitches["balls_after"] - (~final_strike).astype(int)
    strikes = pitches["strikes_after"] - final_strike.astype(int)
    known = balls.between(0, MAX_BALLS) & strikes.between(0, MAX_STRIKES)
    three_balls = balls == MAX_BALLS
    two_strikes = strikes == MAX_STRIKES
    situation = np.select(
        [
            ~known,
            three_balls & two_strikes,
            two_strikes,
            three_balls,
            (balls == 0) & (strikes == 0),
        ],
        ["unknown", "full count", "two strikes", "three balls", "first pitch"],
        default="other counts",
    )
    count = balls.astype(str) + "-" + strikes.astype(str)
    return pd.DataFrame(
        {
            "count": count.where(known, "unknown"),
            "count_situation": situation,
        },
        index=pitches.index,
    )


def split_columns(pitches: pd.DataFrame) -> pd.DataFrame:
    """Return the level each pitch falls in, for every split."""
    from_line = pitches["distance_from_edge_in"] - LINE_IN
    step = np.floor(from_line).clip(DISTANCE_STEPS[0], DISTANCE_STEPS[-1])
    labels = {step: distance_label(step) for step in DISTANCE_STEPS}

    splits = counts_before(pitches)
    splits["zone_region"] = pitches["zone_region"]
    splits["pitch_group"] = (
        pitches["pitch_type_raw"].map(PITCH_GROUPS).fillna("other")
    )
    splits["batter"] = pitches["bat_side"].map(BATTER_SIDES)
    splits["line"] = np.where(
        from_line.abs() <= BORDERLINE_IN, "borderline", "clear"
    )
    splits["distance"] = step.astype(int).map(labels)
    return splits.fillna("unknown").astype("category")


def flag_columns(pitches: pd.DataFrame) -> pd.DataFrame:
    """Return the 0/1 columns that every table adds up."""
    called_strike = pitches["original_call_code"].eq("C")
    true_strike = pitches["is_strike_zone"]
    from_line = pitches["distance_from_edge_in"] - LINE_IN
    near_line = from_line.abs() <= NEAR_LINE_IN
    flags = pd.DataFrame(
        {
            "called": True,
            "correct": pitches["is_call_correct"],
            "true_balls": ~true_strike,
            "true_strikes": true_strike,
            "false_strikes": called_strike & ~true_strike,
            "false_balls": ~called_strike & true_strike,
            "called_strikes": called_strike,
            "near_line": near_line,
            "near_line_strikes": near_line & called_strike,
            "near_line_true_strikes": near_line & true_strike,
        },
        index=pitches.index,
    )
    return flags.astype("int64")


def load_calls() -> pd.DataFrame:
    """Return every graded call with its plate umpire, splits and flags."""
    pitches = pd.read_parquet(
        TABLES_DIR / "pitches_graded" / "all_seasons.parquet",
        columns=PITCH_COLUMNS,
    )
    umpires = umpire_tiers.load_plate_umpires()[["game_pk", "official_id"]]
    pitches = pitches.merge(umpires, on="game_pk", how="inner")
    keys = pitches[[*UMPIRE_SEASON, "game_pk"]]
    return pd.concat(
        [keys, split_columns(pitches), flag_columns(pitches)], axis=1
    )


def totals(calls: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Add up the calls in each group."""
    return calls.groupby(keys, observed=True)[MEASURES].sum().reset_index()


def wilson_columns(hits: pd.Series, trials: pd.Series) -> tuple[list, list]:
    """Return the low and high ends of each 95% Wilson interval."""
    bounds = [
        wilson_interval(int(hit), int(trial))
        for hit, trial in zip(hits, trials, strict=True)
    ]
    return [low for low, _ in bounds], [high for _, high in bounds]


def add_rates(table: pd.DataFrame) -> pd.DataFrame:
    """Turn the counts in a table into rates."""
    table = table.copy()
    called_balls = table["called"] - table["called_strikes"]
    net_strikes = table["false_strikes"] - table["false_balls"]
    table["accuracy"] = table["correct"] / table["called"]
    table["ci_low"], table["ci_high"] = wilson_columns(
        table["correct"], table["called"]
    )
    table["false_strike_rate"] = table["false_strikes"] / table["true_balls"]
    table["false_ball_rate"] = table["false_balls"] / table["true_strikes"]
    table["net_strikes_per_1000"] = net_strikes / table["called"] * 1000
    # Of the pitches he called a strike (or a ball), the share that were
    # wrong. The rates above are per true ball and per true strike.
    table["wrong_strike_share"] = (
        table["false_strikes"] / table["called_strikes"]
    )
    table["wrong_ball_share"] = table["false_balls"] / called_balls
    table["near_line_strike_rate"] = (
        table["near_line_strikes"] / table["near_line"]
    )
    table["near_line_true_share"] = (
        table["near_line_true_strikes"] / table["near_line"]
    )
    return table


def league_table(calls: pd.DataFrame) -> pd.DataFrame:
    """Return all umpires together in every split, season by season."""
    overall = totals(calls, ["season"])
    pieces = [overall.assign(split="all calls", level="all calls", order=0)]
    for split, levels in SPLITS.items():
        piece = totals(calls, ["season", split])
        piece = piece.rename(columns={split: "level"})
        piece["level"] = piece["level"].astype(str)
        position = {level: number for number, level in enumerate(levels)}
        piece["order"] = (
            piece["level"].map(position).fillna(len(levels)).astype(int)
        )
        pieces.append(piece.assign(split=split))

    table = add_rates(pd.concat(pieces, ignore_index=True))
    season_total = overall.set_index("season")["called"]
    table["share_of_called"] = table["called"] / table["season"].map(
        season_total
    )
    table["zone"] = np.where(
        table["season"] >= NEW_ZONE_FIRST_SEASON, "new", "old"
    )
    table["split"] = pd.Categorical(table["split"], ["all calls", *SPLITS])
    table = table.sort_values(["split", "order", "season"])
    table["split"] = table["split"].astype(str)
    return table.reset_index(drop=True)[LEAGUE_COLUMNS]


def season_totals(calls: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    """Start each profile from the umpire's tier row and error counts."""
    counts = totals(calls, UMPIRE_SEASON)
    ump = tiers[TIER_COLUMNS].merge(
        counts[[*UMPIRE_SEASON, "called", *ERROR_COUNTS]],
        on=UMPIRE_SEASON,
        how="left",
        validate="one_to_one",
    )
    if not ump["called"].eq(ump["called_pitches"]).all():
        raise ValueError("profile counts do not match the tier table")
    return ump.drop(columns="called")


def add_error_rates(ump: pd.DataFrame) -> pd.DataFrame:
    """Add the two error rates, their intervals and the league's rates."""
    ump = ump.copy()
    net_strikes = ump["false_strikes"] - ump["false_balls"]
    ump["false_strike_rate"] = ump["false_strikes"] / ump["true_balls"]
    ump["false_strike_low"], ump["false_strike_high"] = wilson_columns(
        ump["false_strikes"], ump["true_balls"]
    )
    ump["false_ball_rate"] = ump["false_balls"] / ump["true_strikes"]
    ump["false_ball_low"], ump["false_ball_high"] = wilson_columns(
        ump["false_balls"], ump["true_strikes"]
    )
    ump["false_strikes_per_game"] = ump["false_strikes"] / ump["plate_games"]
    ump["false_balls_per_game"] = ump["false_balls"] / ump["plate_games"]
    ump["net_strikes_per_1000"] = net_strikes / ump["called_pitches"] * 1000

    pool = ump[ump["qualifies"]].groupby("season")
    pool = pool[["called_pitches", *ERROR_COUNTS]].sum()
    league = pd.DataFrame(
        {
            "league_false_strike_rate": pool["false_strikes"]
            / pool["true_balls"],
            "league_false_ball_rate": pool["false_balls"]
            / pool["true_strikes"],
            "league_net_strikes_per_1000": (
                pool["false_strikes"] - pool["false_balls"]
            )
            / pool["called_pitches"]
            * 1000,
        }
    )
    ump = ump.join(league, on="season")
    ump["false_strike_gap_pp"] = (
        ump["false_strike_rate"] - ump["league_false_strike_rate"]
    ) * 100
    ump["false_ball_gap_pp"] = (
        ump["false_ball_rate"] - ump["league_false_ball_rate"]
    ) * 100
    return ump


def game_squares(ump: pd.DataFrame, by_game: pd.DataFrame) -> pd.DataFrame:
    """Measure how far each game sits from the umpire's season figures.

    The sums of squares feed the standard errors of the lean and of the
    log ratio. Each game counts as one draw, as in the tiers, so pitches
    from the same game are not treated as independent.
    """
    season = ump[
        [*UMPIRE_SEASON, *ERROR_COUNTS, "net_strikes_per_1000"]
    ].astype({name: float for name in ERROR_COUNTS})
    games = by_game.merge(season, on=UMPIRE_SEASON, suffixes=("_game", ""))
    net = games["false_strikes_game"] - games["false_balls_game"]
    expected = games["net_strikes_per_1000"] / 1000 * games["called"]
    ratio = (
        games["false_strikes_game"] / games["false_strikes"]
        - games["true_balls_game"] / games["true_balls"]
        - games["false_balls_game"] / games["false_balls"]
        + games["true_strikes_game"] / games["true_strikes"]
    )
    squares = games[UMPIRE_SEASON].assign(
        net_squares=(net - expected) ** 2, ratio_squares=ratio**2
    )
    return squares.groupby(UMPIRE_SEASON).sum().reset_index()


def add_lean(ump: pd.DataFrame, by_game: pd.DataFrame) -> pd.DataFrame:
    """Add the strike/ball lean, its interval and the log-ratio bias."""
    ump = ump.merge(game_squares(ump, by_game), on=UMPIRE_SEASON, how="left")
    plate_games = ump["plate_games"]
    correction = plate_games / (plate_games - 1).where(plate_games > 1)
    qualifies = ump["qualifies"]

    ump["lean_per_1000"] = (
        ump["net_strikes_per_1000"] - ump["league_net_strikes_per_1000"]
    )
    ump["lean_se_per_1000"] = (
        1000 * np.sqrt(correction * ump["net_squares"]) / ump["called_pitches"]
    )
    half_width = Z_95 * ump["lean_se_per_1000"]
    ump["lean_low_per_1000"] = ump["lean_per_1000"] - half_width
    ump["lean_high_per_1000"] = ump["lean_per_1000"] + half_width
    calls_per_game = ump["called_pitches"] / plate_games
    for end in ("", "_low", "_high"):
        ump[f"lean{end}_per_game"] = (
            ump[f"lean{end}_per_1000"] / 1000 * calls_per_game
        )
    to_strikes = qualifies & (ump["lean_low_per_1000"] > 0)
    to_balls = qualifies & (ump["lean_high_per_1000"] < 0)
    ump["lean_systematic"] = to_strikes | to_balls
    ump["lean_side"] = np.select(
        [~qualifies, to_strikes, to_balls],
        ["under 10 plate games", "strikes", "balls"],
        default="no clear lean",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(ump["false_strike_rate"] / ump["false_ball_rate"])
        league_log_ratio = np.log(
            ump["league_false_strike_rate"] / ump["league_false_ball_rate"]
        )
    ump["bias_log_ratio"] = log_ratio - league_log_ratio
    ump["bias_se"] = np.sqrt(correction * ump["ratio_squares"])
    bias = ["bias_log_ratio", "bias_se"]
    ump[bias] = ump[bias].replace([np.inf, -np.inf], np.nan)
    ump["bias_low"] = ump["bias_log_ratio"] - Z_95 * ump["bias_se"]
    ump["bias_high"] = ump["bias_log_ratio"] + Z_95 * ump["bias_se"]
    ump["bias_systematic"] = qualifies & (
        (ump["bias_low"] > 0) | (ump["bias_high"] < 0)
    )
    return ump.drop(columns=["net_squares", "ratio_squares"])


def league_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Return the league's figures in each cell, from qualifying umpires."""
    pool = table[table["qualifies"]].groupby(["season", "split", "level"])
    pool = pool[["called", "correct", "false_strikes", "false_balls"]].sum()
    false_strikes = pool["false_strikes"] / pool["called"]
    false_balls = pool["false_balls"] / pool["called"]
    net = false_strikes - false_balls
    return pd.DataFrame(
        {
            "league_accuracy": pool["correct"] / pool["called"],
            "league_net_strikes_per_1000": net * 1000,
            # Spread of one call's score: +1 false strike, -1 false ball.
            "league_net_spread": false_strikes + false_balls - net**2,
        }
    )


def split_table(calls: pd.DataFrame, ump: pd.DataFrame) -> pd.DataFrame:
    """Return every umpire's numbers in each split, beside the league's.

    Accuracy: gap_pp is the umpire minus the league in that cell, gap_z
    is the gap divided by the noise expected in a cell of that size, and
    extra_misses is the gap in calls (negative = fewer misses).
    Direction: extra_strikes is how many more strikes he called in that
    cell than the league's habits would give on the same pitches
    (negative = more balls), and lean_z is its size against the noise.
    """
    pieces = []
    for split in UMPIRE_SPLITS:
        piece = totals(calls, [*UMPIRE_SEASON, split])
        piece = piece.rename(columns={split: "level"})
        piece["level"] = piece["level"].astype(str)
        pieces.append(piece.assign(split=split))
    table = pd.concat(pieces, ignore_index=True)
    who = ump[[*UMPIRE_SEASON, "official_name", "qualifies", "plate_games"]]
    table = add_rates(table.merge(who, on=UMPIRE_SEASON))
    table = table.join(league_cells(table), on=["season", "split", "level"])

    table["gap_pp"] = (table["accuracy"] - table["league_accuracy"]) * 100
    noise = table["league_accuracy"] * (1 - table["league_accuracy"])
    table["gap_se_pp"] = 100 * np.sqrt(noise / table["called"])
    table["gap_z"] = table["gap_pp"] / table["gap_se_pp"]
    table["extra_misses"] = -table["gap_pp"] / 100 * table["called"]

    net_strikes = table["false_strikes"] - table["false_balls"]
    league_net = table["league_net_strikes_per_1000"] / 1000
    table["lean_per_1000"] = (
        table["net_strikes_per_1000"] - table["league_net_strikes_per_1000"]
    )
    table["extra_strikes"] = net_strikes - league_net * table["called"]
    table["extra_strikes_per_game"] = (
        table["extra_strikes"] / table["plate_games"]
    )
    table["lean_z"] = table["extra_strikes"] / np.sqrt(
        table["called"] * table["league_net_spread"]
    )
    scores = ["gap_z", "lean_z"]
    table[scores] = table[scores].replace([np.inf, -np.inf], np.nan)
    table["enough_calls"] = table["called"] >= MIN_CELL_CALLS
    return table.sort_values([*UMPIRE_SEASON, "split", "level"]).reset_index(
        drop=True
    )[SPLIT_COLUMNS]


def check_splits(ump: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Stop if a split does not add up to the umpire's called pitches."""
    summed = (
        splits.groupby([*UMPIRE_SEASON, "split"])["called"].sum().unstack()
    )
    expected = ump.set_index(UMPIRE_SEASON)["called_pitches"]
    if not summed.eq(expected, axis=0).all().all():
        raise ValueError("a split does not add up to the called pitches")


def region_extremes(splits: pd.DataFrame) -> pd.DataFrame:
    """Find the regions where each umpire stands out from the league.

    worst / best: the regions where his accuracy is furthest below and
    above the league's. widest / tightest: the regions where he calls
    the most extra strikes and the most extra balls. Each is picked on
    its size against the noise in a cell of that size (the z columns),
    and is "secure" only if it would still stand out after allowing for
    the nine regions that were looked through.
    """
    regions = splits[
        (splits["split"] == "zone_region")
        & (splits["level"] != HEART)
        & splits["enough_calls"]
    ].dropna(subset=["gap_z", "lean_z"])
    by_umpire = regions.groupby(UMPIRE_SEASON)
    picks = {
        "worst": (by_umpire["gap_z"].idxmin(), ACCURACY_NAMES),
        "best": (by_umpire["gap_z"].idxmax(), ACCURACY_NAMES),
        "widest": (by_umpire["lean_z"].idxmax(), LEAN_NAMES),
        "tightest": (by_umpire["lean_z"].idxmin(), LEAN_NAMES),
    }
    pieces = [
        regions.loc[rows]
        .set_index(UMPIRE_SEASON)[list(names)]
        .rename(columns=names)
        .add_prefix(f"{label}_")
        for label, (rows, names) in picks.items()
    ]
    table = pd.concat(pieces, axis=1)
    for label in ("best", "widest"):
        table[f"{label}_region_secure"] = (
            table[f"{label}_region_z"] >= Z_NINE_REGIONS
        )
    for label in ("worst", "tightest"):
        table[f"{label}_region_secure"] = (
            table[f"{label}_region_z"] <= -Z_NINE_REGIONS
        )
    return table.reset_index()


def wide_columns(splits: pd.DataFrame) -> pd.DataFrame:
    """Spread each umpire's cells into columns of his profile row.

    acc_ is his accuracy in the cell, gap_ is that accuracy minus the
    league's, and zone_ is the extra strikes he calls per game in a
    region (positive = a bigger zone there than the league's).
    """
    short = [
        PROFILE_LEVELS.get(pair)
        for pair in zip(splits["split"], splits["level"], strict=True)
    ]
    rows = splits.assign(short=short)
    rows = rows[rows["enough_calls"] & rows["short"].notna()]
    names = list(PROFILE_LEVELS.values())
    regions = [
        name
        for (split, level), name in PROFILE_LEVELS.items()
        if split == "zone_region" and level != HEART
    ]

    def spread(values: str, keep: list[str]) -> pd.DataFrame:
        wide = rows.pivot(index=UMPIRE_SEASON, columns="short", values=values)
        return wide.reindex(columns=keep)

    accuracy = spread("accuracy", names).add_prefix("acc_")
    gap = spread("gap_pp", names).add_prefix("gap_").add_suffix("_pp")
    zone = spread("extra_strikes_per_game", regions)
    zone = zone.add_prefix("zone_").add_suffix("_per_game")
    return pd.concat([accuracy, gap, zone], axis=1).reset_index()


def profile_note(row: tuple) -> str:
    """Describe one umpire-season in a sentence, using only firm results."""
    if not row.qualifies:
        return "Under 10 plate games: not ranked."
    per_game = abs(row.lean_per_game)
    if row.lean_side == "strikes":
        lean = (
            f"leans to strikes, {per_game:.1f} more strike calls a game "
            "than the league"
        )
    elif row.lean_side == "balls":
        lean = (
            f"leans to balls, {per_game:.1f} more ball calls a game "
            "than the league"
        )
    else:
        lean = "no clear lean to strikes or balls"

    shape = []
    if row.widest_region_secure:
        extra = row.widest_region_per_game
        shape.append(f"wide {row.widest_region} ({extra:+.1f} a game)")
    if row.tightest_region_secure:
        extra = row.tightest_region_per_game
        shape.append(f"tight {row.tightest_region} ({extra:+.1f} a game)")
    zone = " and ".join(shape) if shape else "shape close to the league's"
    return f"Tier {row.tier} ({row.tier_name}); {lean}; zone {zone}."


def profile_table(ump: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Join the pieces into one profile row per umpire per season."""
    table = ump.merge(region_extremes(splits), on=UMPIRE_SEASON, how="left")
    for label in ("worst", "best", "widest", "tightest"):
        column = f"{label}_region_secure"
        table[column] = table[column].eq(True)
    table = table.merge(wide_columns(splits), on=UMPIRE_SEASON, how="left")
    table["profile_note"] = [
        profile_note(row) for row in table.itertuples(index=False)
    ]
    return table


def carry_over(pairs: pd.DataFrame, trait: str, label: str) -> dict:
    """Return the season-to-next-season correlation of one trait."""
    both = pairs[[trait, f"{trait}_next"]].dropna()
    return {
        f"{label}_pairs": len(both),
        f"{label}_correlation": both[trait].corr(both[f"{trait}_next"]),
    }


def trait_group(trait: str) -> str:
    """Say what kind of trait a column is, for reading the table."""
    if trait in TRAIT_GROUPS:
        return TRAIT_GROUPS[trait]
    if trait.startswith("zone_"):
        return "zone shape: zone_region"
    short = trait.removeprefix("gap_").removesuffix("_pp")
    split = {name: key[0] for key, name in PROFILE_LEVELS.items()}[short]
    return f"accuracy gap: {split}"


def beyond_accuracy(pool: pd.DataFrame, trait: str) -> pd.Series:
    """Return a trait with the part explained by accuracy taken out.

    Inside each season a straight line of the trait on aal_pp is fitted
    and subtracted. What is left is how an umpire differs from umpires
    who are as accurate as he is.
    """

    def leftover(group: pd.DataFrame) -> pd.Series:
        both = group[[trait, "aal_pp"]].dropna()
        slope, intercept = np.polyfit(both["aal_pp"], both[trait], 1)
        return group[trait] - (intercept + slope * group["aal_pp"])

    pieces = [leftover(group) for _, group in pool.groupby("season")]
    return pd.concat(pieces).reindex(pool.index)


def stability_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """Check whether each trait shows up again in the next season.

    A real trait should be there a year later; noise should not. Each row
    is the correlation, across umpires who qualify in both seasons,
    between a trait in one season and the same trait in the next: for all
    pairs of seasons, for pairs inside the old zone, and for 2025 to
    2026 (old zone to new zone). aal_pp, the accuracy the tiers are built
    on, is the yardstick.

    beyond_accuracy_correlation repeats the all-pairs check after taking
    out what overall accuracy already explains. A trait that stays high
    there is a second way umpires differ; one that drops to zero was
    overall accuracy seen from another angle.
    """
    traits = [
        *TRAIT_GROUPS,
        *[name for name in profiles.columns if name.startswith("gap_")],
        *[name for name in profiles.columns if name.startswith("zone_")],
    ]
    pool = profiles.loc[profiles["qualifies"], [*UMPIRE_SEASON, *traits]]
    leftovers = pool[UMPIRE_SEASON].assign(
        **{trait: beyond_accuracy(pool, trait) for trait in traits[1:]}
    )

    def pairs_of(table: pd.DataFrame) -> pd.DataFrame:
        following = table.assign(season=table["season"] - 1)
        return table.merge(following, on=UMPIRE_SEASON, suffixes=("", "_next"))

    pairs = pairs_of(pool)
    leftover_pairs = pairs_of(leftovers)
    crosses = pairs["season"] + 1 >= NEW_ZONE_FIRST_SEASON
    rows = []
    for trait in traits:
        row = {
            "trait": trait,
            "group": trait_group(trait),
            **carry_over(pairs, trait, "all"),
            **carry_over(pairs[~crosses], trait, "old_zone"),
            **carry_over(pairs[crosses], trait, "old_to_new"),
        }
        if trait in leftovers:
            beyond = carry_over(leftover_pairs, trait, "beyond_accuracy")
            row["beyond_accuracy_correlation"] = beyond[
                "beyond_accuracy_correlation"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def showcase_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """Pick four umpires of the latest season to show as examples.

    Best and worst are the first and the last umpire in the tier ranking.
    The most strike-leaning and most ball-leaning umpires are picked on
    the cautious end of their interval (the lean they show at the very
    least), so a short season cannot win on luck.
    """
    season = profiles["season"].max()
    pool = profiles[(profiles["season"] == season) & profiles["qualifies"]]
    rank = pool["rank_in_season"].astype(float)
    picks = {
        "best": rank.idxmin(),
        "worst": rank.idxmax(),
        "most strike-leaning": pool["lean_low_per_1000"].idxmax(),
        "most ball-leaning": pool["lean_high_per_1000"].idxmin(),
    }
    table = pool.loc[list(picks.values()), SHOWCASE_COLUMNS]
    table.insert(0, "role", list(picks))
    return table.reset_index(drop=True)


def plot_lean_vs_accuracy(
    profiles: pd.DataFrame, showcase: pd.DataFrame
) -> Figure:
    """Draw every umpire of the latest season: lean against accuracy."""
    season = int(showcase["season"].iloc[0])
    rows = profiles[(profiles["season"] == season) & profiles["qualifies"]]
    fig, ax = plot_style.new_figure(height=5.0, top=0.80)
    fig.subplots_adjust(bottom=0.15)

    ax.axvline(0, color=plot_style.BASELINE, linewidth=1)
    league = rows["league_accuracy"].iloc[0] * 100
    ax.axhline(league, color=plot_style.BASELINE, linewidth=1)
    dots = {"s": 80, "color": plot_style.SERIES[0], "linewidth": 2}
    ax.scatter(
        rows["lean_per_game"],
        rows["accuracy"] * 100,
        edgecolor=plot_style.SURFACE,
        zorder=3,
        **dots,
    )
    # The four example umpires get a dark ring and their name.
    ax.scatter(
        showcase["lean_per_game"],
        showcase["accuracy"] * 100,
        edgecolor=plot_style.INK,
        zorder=4,
        **dots,
    )
    for pick in showcase.itertuples():
        ax.annotate(
            pick.official_name,
            (pick.lean_per_game, pick.accuracy * 100),
            textcoords="offset points",
            fontsize=10,
            color=plot_style.INK,
            **LABEL_SPOTS[pick.role],
        )

    reach = np.ceil(rows["lean_per_game"].abs().max() + 0.4)
    ax.set_xlim(-reach, reach)
    low, high = ax.get_ylim()
    ax.set_ylim(np.floor(low * 2) / 2, np.ceil((high + 0.15) * 2) / 2)
    ax.text(
        reach,
        league,
        f"league {league:.1f}% ",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=plot_style.INK_SECONDARY,
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}%")
    ax.xaxis.set_major_formatter(
        lambda value, _: f"{value:+.0f}" if value else "0"
    )
    ax.set_xlabel(
        "Strike calls per game compared with the league  "
        "(left: more ball calls, right: more strike calls)",
        fontsize=10,
        color=plot_style.INK_SECONDARY,
        labelpad=8,
    )
    plot_style.add_title(
        fig,
        "Two ways umpires differ: how often they miss, and which way",
        f"Each dot is one plate umpire with 10 or more games in {season}. "
        "Higher is more accurate.",
    )
    return fig


def near_line_rates(league: pd.DataFrame, new_zone: bool) -> pd.Series:
    """Return the share called a strike near the line, by count situation.

    Seasons under the same zone are added together before the share is
    taken: the old-zone seasons when new_zone is False.
    """
    is_new = league["season"] >= NEW_ZONE_FIRST_SEASON
    rows = league[
        (league["split"] == "count_situation") & (is_new == new_zone)
    ]
    summed = rows.groupby("level")[["near_line", "near_line_strikes"]].sum()
    return summed["near_line_strikes"] / summed["near_line"] * 100


def plot_borderline_by_count(league: pd.DataFrame) -> Figure:
    """Draw how often a pitch on the line is called a strike, by count."""
    seasons = sorted(league["season"].unique())
    old_seasons = [s for s in seasons if s < NEW_ZONE_FIRST_SEASON]
    labels = {
        False: f"Old zone, {old_seasons[0]}-{old_seasons[-1]}",
        True: f"New ABS zone, {seasons[-1]}",
    }
    levels = list(COUNT_LABELS)
    positions = np.arange(len(levels))
    fig, ax = plot_style.new_figure(top=0.70)
    fig.subplots_adjust(bottom=0.16)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, 70)

    handles = []
    for new_zone, color, shift in zip(
        (False, True), plot_style.SERIES, (-0.15, 0.15), strict=False
    ):
        rates = near_line_rates(league, new_zone).reindex(levels)
        plot_style.rounded_columns(
            ax, list(positions + shift), list(rates), [color] * len(levels)
        )
        plot_style.label_columns(
            ax,
            list(positions + shift),
            list(rates),
            [f"{rate:.0f}%" for rate in rates],
        )
        handles.append(Patch(color=color, label=labels[new_zone]))

    ax.set_xticks(positions, [COUNT_LABELS[level] for level in levels])
    ax.tick_params(axis="x", labelcolor=plot_style.INK_SECONDARY)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.02),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        columnspacing=1.6,
        labelcolor=plot_style.INK_SECONDARY,
    )
    plot_style.add_title(
        fig,
        "With two strikes, the same borderline pitch is a strike less often",
        "Share called a strike among pitches within 1 inch of the line "
        "where the right call\nflips from strike to ball. About half of "
        "these pitches are true strikes.",
    )
    return fig


def build() -> dict[str, pd.DataFrame]:
    """Build every table, keyed by the name of the CSV it is saved as."""
    tiers = umpire_tiers.build()["umpire_tiers"]
    calls = load_calls()
    ump = add_error_rates(season_totals(calls, tiers))
    ump = add_lean(ump, totals(calls, [*UMPIRE_SEASON, "game_pk"]))
    splits = split_table(calls, ump)
    check_splits(ump, splits)
    profiles = profile_table(ump, splits)
    return {
        "umpire_profiles": profiles,
        "umpire_split_accuracy": splits,
        "league_split_accuracy": league_table(calls),
        "profile_stability": stability_table(profiles),
        "showcase_umpires": showcase_table(profiles),
    }


def make_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Figure]:
    """Draw the charts and save them to reports/figures."""
    figures = {
        "umpire_lean_vs_accuracy": plot_lean_vs_accuracy(
            tables["umpire_profiles"], tables["showcase_umpires"]
        ),
        "borderline_calls_by_count": plot_borderline_by_count(
            tables["league_split_accuracy"]
        ),
    }
    for name, fig in figures.items():
        path = plot_style.save_figure(fig, name)
        print(f"saved {path.relative_to(PROJECT_DIR)}")
    return figures


def main() -> None:
    """Build the tables and charts, save them and print a short summary."""
    tables = build()
    umpire_tiers.save(tables)
    for fig in make_figures(tables).values():
        plt.close(fig)

    league = tables["league_split_accuracy"]
    overall = league[league["split"] == "all calls"]
    columns = [
        "season",
        "accuracy",
        "false_strike_rate",
        "false_ball_rate",
        "net_strikes_per_1000",
    ]
    print("\nAll umpires, by season")
    print(overall[columns].round(4).to_string(index=False))

    profiles = tables["umpire_profiles"]
    latest = profiles[
        (profiles["season"] == profiles["season"].max())
        & profiles["qualifies"]
    ]
    print("\nLean of the latest season's umpires")
    print(latest["lean_side"].value_counts().to_string())

    print("\nDoes a trait show up again the next season?")
    print(tables["profile_stability"].round(3).to_string(index=False))

    print("\nShowcase umpires")
    for pick in tables["showcase_umpires"].itertuples():
        print(f"{pick.role}: {pick.official_name}. {pick.profile_note}")


if __name__ == "__main__":
    main()
