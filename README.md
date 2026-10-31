# ABS Research

MLB umpire accuracy before and after the ABS challenge system.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Step 1: schedule for one date (games + plate umpire)

    python src/fetch_schedule.py 2026-06-10

Writes data/raw/schedule/<date>.json (untouched API answer) and
data/tables/games/season=<yyyy>/part.parquet (one row per game).