"""Modeling-related plotting utilities saved to figs/modeling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

FIG_FORMAT = "png"
MODEL_DIR = Path(__file__).resolve().parents[2] / "figs" / "modeling"


def _save(fig, name: str | None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MODEL_DIR / f"{name}.{FIG_FORMAT}", bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, *, model_name: str, save_name: str | None = None):
    """Plot ROC curve and save as model_<modelname>_roc."""

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.roc_auc_score(y_true, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#0284c7")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {model_name}")
    ax.legend()
    save_as = save_name or f"model_{model_name}_roc"
    _save(fig, save_as)


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, *, model_name: str, save_name: str | None = None):
    """Plot precision-recall curve."""

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred_proba)
    ap = metrics.average_precision_score(y_true, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="#0369a1", label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall - {model_name}")
    ax.legend()
    save_as = save_name or f"model_{model_name}_pr"
    _save(fig, save_as)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[str], normalize: bool = True, save_name: str | None = None):
    """Plot confusion matrix with optional normalization."""

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    save_as = save_name or "model_confusion_matrix"
    _save(fig, save_as)


def plot_feature_importance(features: list[str], importances: np.ndarray, *, model_name: str, save_name: str | None = None):
    """Plot feature importances for tree/linear models."""

    order = np.argsort(importances)[::-1]
    sorted_features = np.array(features)[order]
    sorted_importances = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(features) + 1))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette="Blues_r")
    ax.set_title(f"Feature Importance - {model_name}")
    save_as = save_name or f"model_{model_name}_fi"
    _save(fig, save_as)


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray, *, save_name: str | None = None):
    """Plot learning curves across train sizes."""

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores, label="Train", color="#0ea5e9")
    ax.plot(train_sizes, val_scores, label="Validation", color="#0c4a6e")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend()
    save_as = save_name or "model_learning_curve"
    _save(fig, save_as)
