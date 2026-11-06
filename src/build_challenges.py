"""Build the challenges tables from saved live feeds for one season.

Two tables:
  challenges            one row per ABS challenge
  challenges_remaining  one row per pitch: challenges each team had left
                        BEFORE the pitch was thrown

Rules (MLB, 2026): each team starts with 2; a failed challenge costs 1;
a successful one is kept; a team entering an extra inning with 0 gets 1.
Our reconstructed end-of-game counts are checked against the feed's own
counters (gameData.absChallenges) for every game.

Usage:  python src/build_challenges.py 2026
"""

import sys
from pathlib import Path

import pandas as pd

from build_pitches import load_feed, original_call, review_for, save_table

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "feed_live"
TABLES_DIR = PROJECT_DIR / "data" / "tables"

KEEP_STATUSES = ("Final", "Completed Early")
STARTING_CHALLENGES = 2
EXTRA_INNING_START = 10
PROGRESS_EVERY = 200

CHALLENGE_COLUMNS = [
    "challenge_id", "game_pk", "season", "at_bat_index", "pitch_number",
    "inning", "half_inning", "record_location", "review_seq",
    "challenge_team_id", "challenger_id", "challenger_name",
    "challenger_role", "final_call_code", "original_call_code",
    "is_overturned",
]


# -------------------------------------------------------------- one row


def challenger_role(review: dict, matchup: dict) -> str:
    """Return batter, pitcher or catcher for the player who challenged."""
    challenger_id = review["player"]["id"]
    if challenger_id == matchup["batter"]["id"]:
        return "batter"
    if challenger_id == matchup["pitcher"]["id"]:
        return "pitcher"
    return "catcher"  # only other player allowed to challenge


def challenge_row(
    game: dict, play: dict, event: dict, review: dict, where: str, seq: int
) -> dict:
    """Return one flat row for one challenge."""
    about = play["about"]
    game_pk = game["game"]["pk"]
    final_code = event["details"]["call"]["code"]
    return {
        "challenge_id": f"{game_pk}-{about['atBatIndex']}-{event['pitchNumber']}-{seq}",
        "game_pk": game_pk,
        "season": int(game["game"]["season"]),
        "at_bat_index": about["atBatIndex"],
        "pitch_number": event["pitchNumber"],
        "inning": about["inning"],
        "half_inning": about["halfInning"],
        "review_seq": seq,
        "challenge_team_id": review.get("challengeTeamId"),
        "challenger_id": review["player"]["id"],
        "challenger_name": review["player"].get("fullName"),
        "challenger_role": challenger_role(review, play["matchup"]),
        "final_call_code": final_code,
        "original_call_code": original_call(final_code, review),
        "is_overturned": review.get("isOverturned"),
    }


# ----------------------------------------------------------- the tables


def challenges_table(feed: dict) -> pd.DataFrame:
    """Return one row per challenge, in game order."""
    game = feed["gameData"]
    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        pitches = [e for e in play["playEvents"] if e["isPitch"]]
        for event in pitches:
            review = review_for(play, event, event is pitches[-1])
            if review is None:
                continue
            where = "pitch" if event.get("reviewDetails") else "play"
            reviews = [review] + [
                r for r in review.get("additionalReviews", [])
                if r.get("reviewType") == "MJ"
            ]
            for seq, r in enumerate(reviews, start=1):
                rows.append(challenge_row(game, play, event, r, where, seq))
    return pd.DataFrame(rows)


def remaining_before_each_pitch(feed: dict) -> pd.DataFrame:
    """Return, per pitch, each team's challenges left BEFORE the pitch."""
    game = feed["gameData"]
    home_id = game["teams"]["home"]["id"]
    left = {"home": STARTING_CHALLENGES, "away": STARTING_CHALLENGES}
    granted_innings = set()
    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        inning = play["about"]["inning"]
        if inning >= EXTRA_INNING_START and inning not in granted_innings:
            granted_innings.add(inning)
            for side in left:
                if left[side] == 0:
                    left[side] = 1
        pitches = [e for e in play["playEvents"] if e["isPitch"]]
        for event in pitches:
            rows.append(
                {
                    "game_pk": game["game"]["pk"],
                    "at_bat_index": play["about"]["atBatIndex"],
                    "pitch_number": event["pitchNumber"],
                    "challenges_left_home": left["home"],
                    "challenges_left_away": left["away"],
                }
            )
            review = review_for(play, event, event is pitches[-1])
            if review is not None:
                reviews = [review] + [
                    r
                    for r in review.get("additionalReviews", [])
                    if r.get("reviewType") == "MJ"
                ]
                for r in reviews:
                    if r.get("isOverturned") is False:
                        side = (
                            "home"
                            if r["challengeTeamId"] == home_id
                            else "away"
                        )
                        left[side] -= 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------- checks


def check_counters(
    feed: dict, challenges: pd.DataFrame, quiet: bool = False
) -> bool:
    """Compare our won/lost counts per team with the feed's counters.

    The feed's `remaining` ignores the extra-inning grant, so we check
    `usedSuccessful` and `usedFailed` instead.
    """
    game = feed["gameData"]
    home_id = game["teams"]["home"]["id"]
    counters = game.get("absChallenges", {})
    ok = True
    for side in ("home", "away"):
        team_id = game["teams"][side]["id"]
        ours = challenges[challenges["challenge_team_id"] == team_id]
        won = int((ours["is_overturned"] == True).sum())  # noqa: E712
        lost = int((ours["is_overturned"] == False).sum())  # noqa: E712
        theirs = counters.get(side, {})
        match = won == theirs.get("usedSuccessful") and lost == theirs.get(
            "usedFailed"
        )
        ok = ok and match
        if not quiet:
            print(
                f"{side}: feed won/lost {theirs.get('usedSuccessful')}/"
                f"{theirs.get('usedFailed')}, ours {won}/{lost}"
                f"  {'OK' if match else 'MISMATCH'}"
            )
    return ok


# ---------------------------------------------------------------- main


def feeds_for(season: int) -> list[Path]:
    """Return the saved feed files for the finished games of a season."""
    games_path = TABLES_DIR / "games" / f"season={season}" / "part.parquet"
    games = pd.read_parquet(games_path)
    finished = games[games["status"].isin(KEEP_STATUSES)]
    paths = [RAW_DIR / f"{pk}.json" for pk in finished["game_pk"].astype(int)]
    return [p for p in paths if p.exists()]


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    paths = feeds_for(season)
    print(f"season {season}: {len(paths)} feeds on disk")

    challenge_frames, remaining_frames, mismatches = [], [], []
    for i, path in enumerate(paths, start=1):
        feed = load_feed(int(path.stem))
        game_challenges = challenges_table(feed)
        challenge_frames.append(game_challenges)
        remaining_frames.append(remaining_before_each_pitch(feed))
        has_abs = (
            feed["gameData"].get("absChallenges", {}).get("hasChallenges")
        )
        if has_abs and not check_counters(feed, game_challenges, quiet=True):
            mismatches.append(int(path.stem))
        if i % PROGRESS_EVERY == 0:
            print(f"  {i}/{len(paths)}")

    challenges = pd.concat(challenge_frames, ignore_index=True)
    remaining = pd.concat(remaining_frames, ignore_index=True)
    if len(challenges):
        assert challenges["challenge_id"].is_unique, "duplicate challenge_id"
        challenges["is_overturned"] = challenges["is_overturned"].astype("boolean")
        overturned = int(challenges["is_overturned"].sum())
    else:
        challenges = pd.DataFrame(columns=CHALLENGE_COLUMNS)
        overturned = 0

    print("saved", save_table(challenges, "challenges", season))
    print("saved", save_table(remaining, "challenges_remaining", season))
    print("challenges:", len(challenges), " overturned:", overturned)

    print("games with counter mismatch:", len(mismatches), mismatches[:20])


if __name__ == "__main__":
    main()
