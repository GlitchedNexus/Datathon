"""Evaluation helpers for classification and regression tasks."""

from typing import Any

import numpy as np
from sklearn import metrics


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, Any]:
    """Compute accuracy/F1/AUC and return a metrics dict."""

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)

    results: dict[str, Any] = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    if y_proba is not None:
        try:
            results["roc_auc"] = metrics.roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            # AUC not defined (e.g., single class present); keep other metrics only
            results["roc_auc"] = None

    return results


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute RMSE/MAE/R2 and return a metrics dict."""

    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = metrics.r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
