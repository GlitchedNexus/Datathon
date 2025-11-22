"""Generate modeling figures (ROC/PR/confusion matrices/feature importance)."""

from pathlib import Path

import numpy as np

from scripts.plots import model_plots

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "modeling"


def generate_model_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    model_name: str,
    labels: list[str],
    feature_names: list[str] | None = None,
    feature_importances: np.ndarray | None = None,
) -> None:
    """Save standard evaluation plots for a trained model."""

    if y_proba is not None:
        model_plots.plot_roc_curve(y_true, y_proba, model_name=model_name)
        model_plots.plot_precision_recall_curve(y_true, y_proba, model_name=model_name)

    model_plots.plot_confusion_matrix(y_true, y_pred, labels=labels)

    if feature_names is not None and feature_importances is not None:
        model_plots.plot_feature_importance(feature_names, feature_importances, model_name=model_name)


if __name__ == "__main__":
    raise SystemExit("Use this module from training pipeline")
