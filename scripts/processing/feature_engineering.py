"""Feature engineering utilities for interim -> processed datasets."""

import pandas as pd


def add_time_features(df: pd.DataFrame, *, timestamp_col: str) -> pd.DataFrame:
    """Create time-based features (year, month, dow, hour, etc.)."""
    engineered = df.copy()
    if timestamp_col not in engineered.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found")

    ts = pd.to_datetime(engineered[timestamp_col], errors="coerce")
    engineered[f"{timestamp_col}_year"] = ts.dt.year
    engineered[f"{timestamp_col}_month"] = ts.dt.month
    engineered[f"{timestamp_col}_day"] = ts.dt.day
    engineered[f"{timestamp_col}_dow"] = ts.dt.dayofweek
    engineered[f"{timestamp_col}_hour"] = ts.dt.hour
    return engineered


def add_ratio_features(df: pd.DataFrame, *, numerators: list[str], denominators: list[str]) -> pd.DataFrame:
    """Create ratio/log-transform features given numerator/denominator lists."""
    engineered = df.copy()
    for num in numerators:
        for den in denominators:
            if num in engineered.columns and den in engineered.columns:
                feature_name = f"{num}_per_{den}"
                engineered[feature_name] = engineered[num] / engineered[den].replace(0, pd.NA)
    return engineered


def add_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hook for problem-specific feature engineering."""

    # Placeholder: return unchanged until domain logic is added
    return df.copy()
