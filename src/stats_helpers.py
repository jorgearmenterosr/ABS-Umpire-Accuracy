"""Small statistics helpers shared by the analysis notebooks."""

import math

import pandas as pd


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Return the 95% Wilson score interval for a proportion.

    Better than the normal approximation when the rate is near 0 or 1
    or the sample is small, which is the case for individual players.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    half = half / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(
    df: pd.DataFrame, by: list[str], success_col: str, min_n: int = 1
) -> pd.DataFrame:
    """Group by `by`, count trials and successes, add rate and interval.

    Rows with fewer than `min_n` trials are kept but their rate and
    interval are set to null, so small samples are visible, not ranked.
    """
    grouped = (
        df.groupby(by, dropna=False)[success_col]
        .agg(trials="size", successes="sum")
        .reset_index()
    )
    grouped["rate"] = grouped["successes"] / grouped["trials"]
    bounds = [
        wilson_interval(int(s), int(n))
        for s, n in zip(grouped["successes"], grouped["trials"])
    ]
    grouped["ci_low"] = [b[0] for b in bounds]
    grouped["ci_high"] = [b[1] for b in bounds]
    small = grouped["trials"] < min_n
    grouped.loc[small, ["rate", "ci_low", "ci_high"]] = float("nan")
    return grouped.sort_values("rate", ascending=False, na_position="last")