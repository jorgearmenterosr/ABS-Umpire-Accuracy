"""Look inside one saved live feed. Scratch script, not part of the pipeline.

Usage:  python src/explore_feed.py 822970
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "feed_live"


def load_feed(game_pk: int) -> dict:
    """Read a saved feed from data/raw/feed_live/<game_pk>.json."""
    with (RAW_DIR / f"{game_pk}.json").open() as fh:
        return json.load(fh)


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n=== {title} ===")


def describe_review(review: dict, at_bat: int, pitch: dict, where: str) -> str:
    """Return one readable line for a challenge record."""
    outcome = "overturned" if review["isOverturned"] else "upheld"
    return (
        f"PA {at_bat:>3}  pitch {pitch['pitchNumber']}  "
        f"final={pitch['details']['description']:<13} "
        f"{outcome:<10} team={review['challengeTeamId']}  "
        f"by {review['player']['fullName']:<22} ({where})"
    )


def main() -> None:
    game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 822970
    feed = load_feed(game_pk)
    game = feed["gameData"]
    plays = feed["liveData"]["plays"]["allPlays"]

    section("Game")
    print("game_pk:", game_pk)
    print("date:   ", game["datetime"]["officialDate"])
    print(
        "home:   ", game["teams"]["home"]["name"], game["teams"]["home"]["id"]
    )
    print(
        "away:   ", game["teams"]["away"]["name"], game["teams"]["away"]["id"]
    )
    print("plays:  ", len(plays))

    section("Pitch events")
    n_events = sum(len(p["playEvents"]) for p in plays)
    n_pitches = sum(1 for p in plays for e in p["playEvents"] if e["isPitch"])
    print("events: ", n_events)
    print("pitches:", n_pitches)

    section("Challenges on pitch events")
    for play in plays:
        for event in play["playEvents"]:
            if event["isPitch"] and event.get("reviewDetails"):
                print(
                    describe_review(
                        event["reviewDetails"],
                        play["about"]["atBatIndex"],
                        event,
                        "pitch event",
                    )
                )

    section("Challenges on plays (last pitch of the PA)")
    for play in plays:
        if play.get("reviewDetails"):
            last_pitch = [e for e in play["playEvents"] if e["isPitch"]][-1]
            print(
                describe_review(
                    play["reviewDetails"],
                    play["about"]["atBatIndex"],
                    last_pitch,
                    "play level",
                )
            )

    section("ABS counters (end of game)")
    counters = game.get("absChallenges", {})
    for side in ("home", "away"):
        c = counters.get(side, {})
        print(
            f"{side}: won {c.get('usedSuccessful')}  "
            f"lost {c.get('usedFailed')}  remaining {c.get('remaining')}"
        )


if __name__ == "__main__":
    main()
