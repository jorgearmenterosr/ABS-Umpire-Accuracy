"""Compare regular-season umpire accuracy by postseason participation."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RANKINGS_PATH = PROJECT_DIR / "reports" / "tables" / "umpire_rankings.csv"
ASSIGNMENTS_PATH = (
    PROJECT_DIR
    / "data"
    / "tables"
    / "postseason_umpire_assignments"
    / "all_seasons.parquet"
)
OUTPUT_TABLE = PROJECT_DIR / "reports" / "tables" / "j4_postseason_accuracy.csv"
OUTPUT_CHART = (
    PROJECT_DIR / "reports" / "figures" / "j4_postseason_accuracy.png"
)

Z_95 = 1.959963984540054


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    """Return the 95% Wilson confidence interval for a proportion."""
    if total <= 0:
        return np.nan, np.nan

    proportion = correct / total
    denominator = 1 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2 * total)) / denominator
    margin = (
        Z_95
        * np.sqrt(
            proportion * (1 - proportion) / total
            + Z_95**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def load_inputs() -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """Load rankings and identify postseason officials by season."""
    rankings = pd.read_csv(RANKINGS_PATH)
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)

    required_rankings = {
        "season",
        "official_id",
        "official_name",
        "called_pitches",
        "correct",
    }
    required_assignments = {"season", "official_id"}

    missing_rankings = required_rankings - set(rankings.columns)
    missing_assignments = required_assignments - set(assignments.columns)

    if missing_rankings:
        raise ValueError(
            f"Rankings file is missing columns: {sorted(missing_rankings)}"
        )
    if missing_assignments:
        raise ValueError(
            "Postseason assignments file is missing columns: "
            f"{sorted(missing_assignments)}"
        )

    for frame in (rankings, assignments):
        frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
        frame["official_id"] = pd.to_numeric(
            frame["official_id"], errors="coerce"
        )
        frame.dropna(subset=["season", "official_id"], inplace=True)
        frame["season"] = frame["season"].astype(int)
        frame["official_id"] = frame["official_id"].astype(int)

    rankings["called_pitches"] = pd.to_numeric(
        rankings["called_pitches"], errors="coerce"
    )
    rankings["correct"] = pd.to_numeric(rankings["correct"], errors="coerce")
    rankings.dropna(subset=["called_pitches", "correct"], inplace=True)

    postseason_pairs = set(
        assignments[["season", "official_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    return rankings, postseason_pairs


def build_summary(
    rankings: pd.DataFrame,
    postseason_pairs: set[tuple[int, int]],
) -> pd.DataFrame:
    """Aggregate regular-season accuracy by postseason participation."""
    data = rankings.copy()
    data["postseason_umpire"] = [
        (season, official_id) in postseason_pairs
        for season, official_id in zip(data["season"], data["official_id"])
    ]

    summary_rows = []
    for (season, postseason_umpire), group in data.groupby(
        ["season", "postseason_umpire"], sort=True
    ):
        total_pitches = int(group["called_pitches"].sum())
        total_correct = int(group["correct"].sum())
        ci_low, ci_high = wilson_interval(total_correct, total_pitches)

        summary_rows.append(
            {
                "season": int(season),
                "group": (
                    "Postseason umpires"
                    if postseason_umpire
                    else "Non-postseason umpires"
                ),
                "umpires": int(group["official_id"].nunique()),
                "called_pitches": total_pitches,
                "correct_calls": total_correct,
                "accuracy": total_correct / total_pitches,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(summary_rows)


def make_chart(summary: pd.DataFrame) -> None:
    """Save a chart of accuracy by season with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "Postseason umpires": {
            "color": "#1f77b4",
            "marker": "o",
            "label": "Worked the postseason",
        },
        "Non-postseason umpires": {
            "color": "#e07a2d",
            "marker": "s",
            "label": "Did not work the postseason",
        },
    }

    for group_name, style in styles.items():
        group = summary[summary["group"] == group_name].sort_values("season")
        if group.empty:
            continue

        lower_error = group["accuracy"] - group["ci_low"]
        upper_error = group["ci_high"] - group["accuracy"]

        ax.errorbar(
            group["season"],
            group["accuracy"],
            yerr=np.vstack([lower_error, upper_error]),
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            capsize=4,
            label=style["label"],
        )

    ax.set_title("Regular-season Umpire Accuracy by Postseason Participation")
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy on Called Pitches")
    ax.set_xticks(sorted(summary["season"].unique()))
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    OUTPUT_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_CHART, dpi=200)
    plt.close(fig)


def main() -> None:
    rankings, postseason_pairs = load_inputs()
    summary = build_summary(rankings, postseason_pairs)

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_TABLE, index=False)
    make_chart(summary)

    print(f"Saved table: {OUTPUT_TABLE}")
    print(f"Saved chart: {OUTPUT_CHART}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()