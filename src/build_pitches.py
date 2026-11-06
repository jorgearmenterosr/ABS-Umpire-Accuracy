"""Build the pitches table from one saved live feed.

One row per pitch, in game order. Columns follow docs/conventions.md.
Fields we deliberately leave out (broadcast colours, pixel coordinates)
stay available in the raw JSON.

Usage:  python src/build_pitches.py 822970
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "feed_live"
TABLES_DIR = PROJECT_DIR / "data" / "tables"

CALLED_CODES = ("B", "C")  # B = ball, C = called strike
ABS_REVIEW_TYPE = "MJ"
PITCH_KEY = ["game_pk", "at_bat_index", "pitch_number"]
NULLABLE_INT_COLUMNS = [
    "event_index",
    "zone_raw",
    "balls_after",
    "strikes_after",
    "outs_after",
    "challenge_team_id",
    "challenger_id",
]


# ---------------------------------------------------------------- loading


def load_feed(game_pk: int) -> dict:
    """Read a saved feed from data/raw/feed_live/<game_pk>.json."""
    with (RAW_DIR / f"{game_pk}.json").open() as fh:
        return json.load(fh)


# ------------------------------------------------------- challenge rules


def review_for(play: dict, event: dict, is_last_pitch: bool) -> dict | None:
    """Return the ABS challenge record for a pitch, or None.

    Rule 1: a challenge is stored on the pitch event, except when it
    ended the plate appearance; then it is stored on the play and
    belongs to that play's last pitch.
    Rule 3: reviewDetails covers every replay review (tag plays, home
    runs, ...). Only reviewType "MJ" is an ABS ball/strike challenge.
    """
    review = event.get("reviewDetails")
    if review is None and is_last_pitch:
        review = play.get("reviewDetails")
    if review is None or review.get("reviewType") != ABS_REVIEW_TYPE:
        return None
    return review


def original_call(final_code: str, review: dict | None) -> str | None:
    """Return the umpire's original call code, or None if unknowable.

    Rule 2: the feed stores the final call. If a challenge overturned
    it, the original was the opposite. Only balls and called strikes
    have an original call.
    """
    if final_code not in CALLED_CODES:
        return None
    if review is None:
        return final_code
    overturned = review.get("isOverturned")
    if overturned is True:
        return "C" if final_code == "B" else "B"
    if overturned is False:
        return final_code
    return None


# ------------------------------------------------------------- one row


def pitch_row(
    game: dict, play: dict, event: dict, review: dict | None
) -> dict:
    """Return one flat row for one pitch event."""
    about = play["about"]
    matchup = play["matchup"]
    details = event["details"]
    count = event.get("count", {})
    pitch_data = event.get("pitchData", {})
    coords = pitch_data.get("coordinates", {})
    breaks = pitch_data.get("breaks", {})
    return {
        # keys
        "game_pk": game["game"]["pk"],
        "at_bat_index": about["atBatIndex"],
        "pitch_number": event["pitchNumber"],
        "play_id": event.get("playId"),
        # context
        "season": int(game["game"]["season"]),
        "game_date": game["datetime"]["officialDate"],
        "inning": about["inning"],
        "half_inning": about["halfInning"],
        "event_index": event.get("index"),
        "batter_id": matchup["batter"]["id"],
        "pitcher_id": matchup["pitcher"]["id"],
        "bat_side": matchup["batSide"]["code"],
        "pitch_hand": matchup["pitchHand"]["code"],
        # the call
        "call_code_raw": details["call"]["code"],
        "call_description_raw": details["description"],
        "original_call_code": original_call(details["call"]["code"], review),
        "is_strike": details.get("isStrike"),
        "is_ball": details.get("isBall"),
        "is_in_play": details.get("isInPlay"),
        # the pitch
        "pitch_type_raw": details.get("type", {}).get("code"),
        "pitch_type_confidence": pitch_data.get("typeConfidence"),
        "start_speed_mph": pitch_data.get("startSpeed"),
        "end_speed_mph": pitch_data.get("endSpeed"),
        "extension_ft": pitch_data.get("extension"),
        "plate_time_s": pitch_data.get("plateTime"),
        "zone_raw": pitch_data.get("zone"),
        # location at the plate (feet) and the batter's zone (feet)
        "plate_x_ft": coords.get("pX"),
        "plate_z_ft": coords.get("pZ"),
        "sz_top_ft": pitch_data.get("strikeZoneTop"),
        "sz_bot_ft": pitch_data.get("strikeZoneBottom"),
        # movement (inches) and spin
        "pfx_x_in": coords.get("pfxX"),
        "pfx_z_in": coords.get("pfxZ"),
        "spin_rate_rpm": breaks.get("spinRate"),
        "spin_direction_deg": breaks.get("spinDirection"),
        # trajectory: position (ft), velocity (ft/s), acceleration (ft/s^2)
        "x0_ft": coords.get("x0"),
        "y0_ft": coords.get("y0"),
        "z0_ft": coords.get("z0"),
        "vx0_fps": coords.get("vX0"),
        "vy0_fps": coords.get("vY0"),
        "vz0_fps": coords.get("vZ0"),
        "ax_fps2": coords.get("aX"),
        "ay_fps2": coords.get("aY"),
        "az_fps2": coords.get("aZ"),
        # count after the pitch
        "balls_after": count.get("balls"),
        "strikes_after": count.get("strikes"),
        "outs_after": count.get("outs"),
        # the challenge
        "is_challenged": review is not None,
        "is_overturned": None
        if review is None
        else review.get("isOverturned"),
        "challenge_team_id": None
        if review is None
        else review.get("challengeTeamId"),
        "challenger_id": None if review is None else review["player"]["id"],
        # timing
        "start_time": event.get("startTime"),
        "end_time": event.get("endTime"),
    }


# ----------------------------------------------------------- the table


def pitches_table(feed: dict) -> pd.DataFrame:
    """Return one row per pitch, in game order."""
    game = feed["gameData"]
    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        pitches = [e for e in play["playEvents"] if e["isPitch"]]
        for event in pitches:
            review = review_for(play, event, event is pitches[-1])
            rows.append(pitch_row(game, play, event, review))
    return pd.DataFrame(rows)


def set_types(table: pd.DataFrame) -> pd.DataFrame:
    """Give columns that can be empty a type that keeps null as null."""
    for col in NULLABLE_INT_COLUMNS:
        table[col] = table[col].astype("Int64")
    table["is_overturned"] = table["is_overturned"].astype("boolean")
    return table


# ------------------------------------------------------------- saving


def save_table(table: pd.DataFrame, name: str, season: int) -> Path:
    """Write Parquet plus a CSV copy to data/tables/<name>/season=<season>/."""
    out_dir = TABLES_DIR / name / f"season={season}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part.parquet"
    table.to_parquet(path, index=False)
    table.to_csv(path.with_suffix(".csv"), index=False)
    return path


# ---------------------------------------------------------------- main


def main() -> None:
    game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 822970
    table = set_types(pitches_table(load_feed(game_pk)))
    assert not table.duplicated(PITCH_KEY).any(), "duplicate pitch key"

    season = int(table["season"].iloc[0])
    path = save_table(table, "pitches", season)

    print("=== summary ===")
    print("saved:          ", path)
    print("rows x columns: ", table.shape)
    print(
        "called (B/C):   ",
        int(table["call_code_raw"].isin(CALLED_CODES).sum()),
    )
    print("challenged:     ", int(table["is_challenged"].sum()))
    print("overturned:     ", int((table["is_overturned"] == True).sum()))  # noqa: E712
    print("missing plate_x:", int(table["plate_x_ft"].isna().sum()))


if __name__ == "__main__":
    main()
