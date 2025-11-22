"""Analyze model performance across user-defined segments (age/region/etc)."""

import pandas as pd


def segment_performance(df: pd.DataFrame, *, segment_col: str, y_true: str, y_pred: str) -> pd.DataFrame:
    """Return aggregated performance metrics per segment."""

    if segment_col not in df.columns:
        raise KeyError(f"Segment column '{segment_col}' not found")

    grouped = df.groupby(segment_col)
    results = grouped.apply(
        lambda g: pd.Series(
            {
                "count": len(g),
                "accuracy": (g[y_true] == g[y_pred]).mean(),
            }
        )
    )
    return results.reset_index()
