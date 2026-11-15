"""Grade called pitches using coordinates measured at each season's zone plane.

Run from the project root:
    python src/grade_pitches.py

For 2026, this joins Baseball Savant's midpoint-of-plate coordinates onto
the MLB Stats API pitch table by game, at-bat and pitch number. The joined
locations are cached under data/tables/pitches/.
"""

import argparse
import time
from pathlib import Path
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "tables" / "pitches" / "all_seasons.parquet"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "tables" / "pitches_graded"
DEFAULT_REPORTS = PROJECT_DIR / "reports"
SAVANT_CACHE = (
    PROJECT_DIR
    / "data"
    / "tables"
    / "pitches"
    / "savant_2026_locations.parquet"
)
SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SAVANT_PARAMS = {
    "all": "true",
    "type": "pitcher",
    "hfSea": "2026|",
    "hfGT": "R|",
}
SAVANT_USECOLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "plate_x",
    "plate_z",
]
PITCH_KEYS = ["game_pk", "at_bat_index", "pitch_number"]
HALF_WIDTH_IN = 8.5
BALL_RADIUS_IN = 1.45
UNCERTAINTY_BAND_IN = 0.5
CHUNK_SIZE = 200_000
EXPECTED_2026_CHALLENGES = 10_557

REQUIRED_COLUMNS = {
    "season",
    "game_pk",
    "at_bat_index",
    "pitch_number",
    "original_call_code",
    "plate_x_ft",
    "plate_z_ft",
    "sz_top_ft",
    "sz_bot_ft",
    "bat_side",
    "is_challenged",
    "is_overturned",
}


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson interval for a proportion."""
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(
        p * (1 - p) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return max(0, centre - half), min(1, centre + half)


def load_savant_locations(input_path, refresh=False):
    """Load or download all 2026 regular-season midpoint locations."""
    if SAVANT_CACHE.exists() and not refresh:
        locations = pd.read_parquet(SAVANT_CACHE)
        print(f"Using cached Savant locations: {SAVANT_CACHE}")
        return locations

    pitch_dates = pd.read_parquet(
        input_path,
        columns=["season", "game_date"],
    )
    dates = (
        pd.to_datetime(
            pitch_dates.loc[pitch_dates["season"] == 2026, "game_date"],
            errors="coerce",
        )
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
    )
    dates = sorted(dates.tolist())
    if not dates:
        raise ValueError("No 2026 game dates found in the pitch table.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/157.0.0.0 Safari/537.36"
        ),
        "Referer": "https://baseballsavant.mlb.com/",
    }
    daily_frames = []
    print("Downloading 2026 regular-season locations by game date...")

    for number, game_date in enumerate(dates, start=1):
        params = {
            **SAVANT_PARAMS,
            "game_date_gt": game_date,
            "game_date_lt": game_date,
        }
        url = f"{SAVANT_URL}?{urlencode(params)}"
        daily = pd.read_csv(
            url,
            usecols=SAVANT_USECOLS,
            storage_options=headers,
        )

        if len(daily) >= 25_000:
            raise ValueError(
                f"Savant returned 25,000 rows for {game_date}; "
                "that date may still be capped."
            )
        if not daily.empty:
            daily_frames.append(daily)

        if number % 10 == 0 or number == len(dates):
            print(f"Downloaded {number}/{len(dates)} game dates")
        time.sleep(0.2)

    if not daily_frames:
        raise ValueError("Baseball Savant returned no 2026 location rows.")

    locations = pd.concat(daily_frames, ignore_index=True)
    locations = locations.rename(
        columns={
            "at_bat_number": "at_bat_index",
            "plate_x": "plate_x_savant_ft",
            "plate_z": "plate_z_savant_ft",
        }
    )

    for column in ["game_pk", "at_bat_index", "pitch_number"]:
        locations[column] = pd.to_numeric(
            locations[column], errors="coerce"
        ).astype("Int64")

    # Baseball Savant numbers at-bats from 1; this project starts at 0.
    locations["at_bat_index"] = locations["at_bat_index"] - 1
    locations["plate_x_savant_ft"] = pd.to_numeric(
        locations["plate_x_savant_ft"], errors="coerce"
    )
    locations["plate_z_savant_ft"] = pd.to_numeric(
        locations["plate_z_savant_ft"], errors="coerce"
    )
    locations = locations.dropna(subset=PITCH_KEYS)

    if len(locations) < 500_000:
        raise ValueError(
            f"Only {len(locations):,} Savant rows downloaded; "
            "the 2026 season download looks incomplete."
        )

    duplicates = locations.duplicated(PITCH_KEYS)
    if duplicates.any():
        sample = locations.loc[duplicates, PITCH_KEYS].head(10)
        raise ValueError(
            "Baseball Savant has duplicate pitch keys:\n"
            f"{sample.to_string(index=False)}"
        )

    SAVANT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    locations.to_parquet(SAVANT_CACHE, index=False)
    print(f"Cached Savant locations: {SAVANT_CACHE}")
    return locations


def apply_savant_locations(frame, locations):
    """Replace 2026 feed locations with Savant midpoint locations."""
    is_2026 = frame["season"].eq(2026).fillna(False)
    rows = frame.loc[is_2026, PITCH_KEYS].copy()
    if rows.empty:
        return frame, 0

    rows["_row_index"] = rows.index
    joined = rows.merge(
        locations,
        on=PITCH_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row_index")

    missing = (
        joined["plate_x_savant_ft"].isna()
        | joined["plate_z_savant_ft"].isna()
    )

    # Do not mix front-of-plate feed locations with midpoint locations.
    frame.loc[joined["_row_index"], "plate_x_ft"] = joined[
        "plate_x_savant_ft"
    ].to_numpy()
    frame.loc[joined["_row_index"], "plate_z_ft"] = joined[
        "plate_z_savant_ft"
    ].to_numpy()

    return frame, int(missing.sum())


def zone_region(x, z, top, bot, bat_side, dx, dz):
    """Label each pitch's 3-by-3 region, or corner if outside both axes."""
    height = top - bot
    if np.any(height <= 0):
        raise ValueError("Found a pitch with sz_top_ft <= sz_bot_ft")

    vertical_position = (z - bot) / height
    vertical = np.select(
        [vertical_position < 1 / 3, vertical_position > 2 / 3],
        ["low", "high"],
        default="mid",
    )

    # Positive transformed x points toward the batter's inside edge.
    valid_side = np.isin(bat_side, ["R", "L"])
    inside_positive_x = np.where(bat_side == "R", -x, x)
    horizontal = np.select(
        [
            inside_positive_x > HALF_WIDTH_IN / 3,
            inside_positive_x < -HALF_WIDTH_IN / 3,
        ],
        ["inside", "outside"],
        default="middle",
    )

    labels = np.char.add(
        np.char.add(vertical.astype(str), "-"),
        horizontal.astype(str),
    )
    labels[(dx > 0) & (dz > 0) & valid_side] = "corner"
    return np.where(valid_side, labels, None)


def grade_batch(batch, locations):
    """Grade eligible pitches from one Arrow batch."""
    frame = batch.to_pandas().reset_index(drop=True)
    frame, savant_missing = apply_savant_locations(frame, locations)

    has_called_location = (
        frame["original_call_code"].isin(["B", "C"])
        & frame[
            ["plate_x_ft", "plate_z_ft", "sz_top_ft", "sz_bot_ft"]
        ].notna().all(axis=1)
    )

    invalid_bounds = has_called_location & (
        frame["sz_top_ft"] <= frame["sz_bot_ft"]
    ).fillna(False)

    if invalid_bounds.any():
        columns = [
            "season",
            "game_pk",
            "at_bat_index",
            "pitch_number",
            "sz_top_ft",
            "sz_bot_ft",
        ]
        print(
            "WARNING: skipping called pitches with invalid zone bounds:",
            int(invalid_bounds.sum()),
        )
        print(
            frame.loc[invalid_bounds, columns]
            .head(10)
            .to_string(index=False)
        )

    eligible = has_called_location & (
        frame["sz_top_ft"] > frame["sz_bot_ft"]
    ).fillna(False)
    graded = frame.loc[eligible].copy()

    invalid_counts = {}
    for season, count in (
        frame.loc[invalid_bounds].groupby("season").size().items()
    ):
        invalid_counts[int(season)] = int(count)

    if graded.empty:
        return graded, {
            "season": {},
            "invalid_bounds": invalid_counts,
            "validation": {},
            "savant_missing": savant_missing,
        }

    x = graded["plate_x_ft"].to_numpy(dtype=float) * 12
    z = graded["plate_z_ft"].to_numpy(dtype=float) * 12
    top = graded["sz_top_ft"].to_numpy(dtype=float) * 12
    bot = graded["sz_bot_ft"].to_numpy(dtype=float) * 12

    dx = np.maximum(np.abs(x) - HALF_WIDTH_IN, 0)
    dz = np.maximum.reduce([bot - z, z - top, np.zeros(len(graded))])
    dist_outside = np.hypot(dx, dz)
    distance = np.where(
        dist_outside > 0,
        dist_outside,
        -np.minimum.reduce(
            [HALF_WIDTH_IN - np.abs(x), z - bot, top - z]
        ),
    )

    in_zone = dist_outside <= BALL_RADIUS_IN
    original_strike = graded["original_call_code"].to_numpy() == "C"
    correct = np.where(in_zone, original_strike, ~original_strike)
    uncertain = np.abs(distance - BALL_RADIUS_IN) <= UNCERTAINTY_BAND_IN

    graded["distance_from_edge_in"] = distance
    graded["is_strike_zone"] = in_zone
    graded["is_call_correct"] = correct
    graded["is_call_correct_confident"] = pd.array(
        np.where(uncertain, None, correct),
        dtype="boolean",
    )
    graded["zone_region"] = zone_region(
        x,
        z,
        top,
        bot,
        graded["bat_side"].fillna("").to_numpy(),
        dx,
        dz,
    )

    season_counts = {}
    with_uncertain = graded.assign(_uncertain=uncertain)
    for season, group in with_uncertain.groupby("season"):
        season_counts[int(season)] = {
            "eligible": len(group),
            "uncertain": int(group["_uncertain"].sum()),
        }

    # Reconstruct the post-challenge call for validation.
    challenged = graded["is_challenged"].fillna(False).astype(bool).to_numpy()
    overturned = graded["is_overturned"].fillna(False).astype(bool).to_numpy()
    calls = graded["original_call_code"].to_numpy(dtype=object)
    final_calls = calls.copy()
    final_calls[challenged & overturned & (calls == "B")] = "C"
    final_calls[challenged & overturned & (calls == "C")] = "B"

    validation_mask = (
        (graded["season"].to_numpy() == 2026)
        & challenged
        & graded["is_overturned"].notna().to_numpy()
        & np.isin(final_calls, ["B", "C"])
    )

    validation = {}
    if validation_mask.any():
        agrees = in_zone == (final_calls == "C")
        validation = {
            "trials": int(validation_mask.sum()),
            "correct": int(agrees[validation_mask].sum()),
            "disagreement_distances": distance[validation_mask][
                ~agrees[validation_mask]
            ].tolist(),
        }

    return graded, {
        "season": season_counts,
        "invalid_bounds": invalid_counts,
        "validation": validation,
        "savant_missing": savant_missing,
    }


def output_schema(input_schema):
    """Add the five grading columns to the original Parquet schema."""
    new_columns = [
        pa.field("distance_from_edge_in", pa.float64()),
        pa.field("is_strike_zone", pa.bool_()),
        pa.field("is_call_correct", pa.bool_()),
        pa.field("is_call_correct_confident", pa.bool_()),
        pa.field("zone_region", pa.string()),
    ]
    return pa.schema(list(input_schema) + new_columns)


def write_reports(
    season_totals,
    disagreements,
    invalid_totals,
    validation_trials,
    validation_correct,
    reports_dir,
):
    """Save seasonal uncertainty, invalid-bound and validation reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seasons = sorted(set(season_totals) | set(invalid_totals))

    for season in seasons:
        counts = season_totals.get(
            season,
            {"eligible": 0, "uncertain": 0},
        )
        n = counts["eligible"]
        uncertain = counts["uncertain"]
        ci_low, ci_high = wilson_interval(uncertain, n)
        rows.append(
            {
                "season": season,
                "eligible_called_pitches": n,
                "within_uncertainty_band": uncertain,
                "uncertainty_band_share": uncertain / n if n else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "invalid_zone_bounds": invalid_totals.get(season, 0),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = reports_dir / "grading_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("=== Uncertainty band and invalid bounds by season ===")
    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")

    if validation_trials:
        ci_low, ci_high = wilson_interval(
            validation_correct,
            validation_trials,
        )
        agreement = validation_correct / validation_trials
        validation_summary = pd.DataFrame(
            [
                {
                    "season": 2026,
                    "challenged_pitches": validation_trials,
                    "agreement": agreement,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            ]
        )
        validation_path = reports_dir / "grading_validation_2026.csv"
        validation_summary.to_csv(validation_path, index=False)
        print(f"Saved: {validation_path}")

    if disagreements:
        figure_path = reports_dir / "j0_validation_disagreement_distances.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(disagreements, bins=30, edgecolor="white")
        ax.set_title("2026 challenged-pitch grading disagreements")
        ax.set_xlabel("Signed distance from strike-zone edge (inches)")
        ax.set_ylabel("Disagreements")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
        print(f"Saved: {figure_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--batch-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--refresh-savant",
        action="store_true",
        help="Redownload the 2026 Savant location cache.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input Parquet file not found: {args.input}")

    locations = load_savant_locations(
        args.input,
        refresh=args.refresh_savant,
    )
    print(f"Savant location rows: {len(locations):,}")

    parquet = pq.ParquetFile(args.input)
    missing = REQUIRED_COLUMNS.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "all_seasons.parquet"
    schema = output_schema(parquet.schema_arrow)
    all_writer = pq.ParquetWriter(all_path, schema)

    season_writers = {}
    csv_started = set()
    season_totals = {}
    invalid_totals = {}
    disagreements = []
    validation_trials = 0
    validation_correct = 0
    savant_missing_total = 0

    try:
        for batch_number, batch in enumerate(
            parquet.iter_batches(batch_size=args.batch_size),
            start=1,
        ):
            graded, info = grade_batch(batch, locations)
            savant_missing_total += info["savant_missing"]

            for season, count in info["invalid_bounds"].items():
                invalid_totals[season] = invalid_totals.get(season, 0) + count

            for season, counts in info["season"].items():
                totals = season_totals.setdefault(
                    season,
                    {"eligible": 0, "uncertain": 0},
                )
                totals["eligible"] += counts["eligible"]
                totals["uncertain"] += counts["uncertain"]

            validation = info["validation"]
            if validation:
                validation_trials += validation["trials"]
                validation_correct += validation["correct"]
                disagreements.extend(validation["disagreement_distances"])

            if graded.empty:
                continue

            all_table = pa.Table.from_pandas(
                graded,
                schema=schema,
                preserve_index=False,
                safe=False,
            )
            all_writer.write_table(all_table)

            for season, group in graded.groupby("season", sort=True):
                season = int(season)
                season_dir = args.output_dir / f"season={season}"
                season_dir.mkdir(parents=True, exist_ok=True)

                if season not in season_writers:
                    season_writers[season] = pq.ParquetWriter(
                        season_dir / "part.parquet",
                        schema,
                    )

                season_table = pa.Table.from_pandas(
                    group,
                    schema=schema,
                    preserve_index=False,
                    safe=False,
                )
                season_writers[season].write_table(season_table)

                csv_path = season_dir / "part.csv"
                group.to_csv(
                    csv_path,
                    index=False,
                    mode="a" if season in csv_started else "w",
                    header=season not in csv_started,
                )
                csv_started.add(season)

            print(
                f"Processed batch {batch_number:,}: "
                f"{len(graded):,} eligible called pitches"
            )
    finally:
        all_writer.close()
        for writer in season_writers.values():
            writer.close()

    print(f"2026 pitches without Savant coordinates: {savant_missing_total:,}")
    write_reports(
        season_totals,
        disagreements,
        invalid_totals,
        validation_trials,
        validation_correct,
        args.reports_dir,
    )

    if validation_trials:
        agreement = validation_correct / validation_trials
        ci_low, ci_high = wilson_interval(
            validation_correct,
            validation_trials,
        )
        print("=== 2026 challenged-pitch validation ===")
        print(
            f"Agreement: {validation_correct:,}/{validation_trials:,} "
            f"({agreement:.2%})"
        )
        print(f"95% Wilson interval: [{ci_low:.2%}, {ci_high:.2%}]")

        if validation_trials != EXPECTED_2026_CHALLENGES:
            print(
                "WARNING: validation count differs from the expected "
                f"{EXPECTED_2026_CHALLENGES:,} challenges."
            )
        if agreement < 0.90:
            print("STOP: agreement is below 90%; inspect the inputs.")
        elif agreement < 0.95:
            print("WARNING: agreement is below the 95% target.")
        else:
            print("Validation meets the 95% target.")
    else:
        print("WARNING: no eligible 2026 challenged pitches for validation.")

    print(f"Graded Parquet: {all_path}")


if __name__ == "__main__":
    main()