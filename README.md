# ABS-Umpire-Accuracy

Evaluating whether MLB's Automated Ball-Strike (ABS) challenge system improves
umpire accuracy using pitch-level and challenge data.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Step 1: schedule for one date (games + plate umpire)

    python src/fetch_schedule.py 2026-06-10

Writes data/raw/schedule/<date>.json (untouched API answer) and
data/tables/games/season=<yyyy>/part.parquet (one row per game).