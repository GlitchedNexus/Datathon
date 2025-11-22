"""Utilities to inspect high-error predictions for models."""

import pandas as pd


def highest_residuals(df: pd.DataFrame, *, y_true: str, y_pred: str, top_k: int = 20) -> pd.DataFrame:
    """Return the rows with the largest absolute residuals for regression."""
    working = df.copy()
    working["residual"] = (working[y_true] - working[y_pred]).abs()
    return working.sort_values("residual", ascending=False).head(top_k)


def misclassified_samples(df: pd.DataFrame, *, y_true: str, y_pred: str, top_k: int = 20) -> pd.DataFrame:
    """Return most confident misclassifications for classification."""

    misclassified = df[df[y_true] != df[y_pred]].copy()
    if "y_proba" in misclassified.columns:
        misclassified = misclassified.sort_values("y_proba", ascending=False)
    return misclassified.head(top_k)
