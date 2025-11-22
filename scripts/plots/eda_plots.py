"""Reusable plotting utilities for EDA figures saved to figs/eda."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIG_FORMAT = "png"  # default export format for slides
EDA_DIR = Path(__file__).resolve().parents[2] / "figs" / "eda"


def _save(fig, name: str | None) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_DIR / f"{name}.{FIG_FORMAT}", bbox_inches="tight")
    plt.close(fig)


def plot_histogram(df: pd.DataFrame, column: str, *, bins: int = 30, title: str | None = None, save_name: str | None = None):
    """Plot a histogram with consistent styling and save to figs/eda/eda_distribution_<column>."""

    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), bins=bins, ax=ax, color="#38bdf8")
    ax.set_title(title or f"Distribution of {column}")
    save_as = save_name or f"eda_distribution_{column}"
    _save(fig, save_as)


def plot_boxplot(df: pd.DataFrame, column: str, by: str | None = None, *, title: str | None = None, save_name: str | None = None):
    """Plot a boxplot optionally grouped by a column."""

    fig, ax = plt.subplots()
    if by:
        sns.boxplot(data=df, x=by, y=column, ax=ax, palette="Blues")
    else:
        sns.boxplot(y=df[column], ax=ax, color="#0284c7")
    ax.set_title(title or f"Boxplot of {column}")
    save_as = save_name or f"eda_boxplot_{column}"
    _save(fig, save_as)


def plot_time_series(df: pd.DataFrame, x: str, y: str, group: str | None = None, *, title: str | None = None, save_name: str | None = None):
    """Plot a time series with optional grouping."""

    fig, ax = plt.subplots()
    if group:
        sns.lineplot(data=df, x=x, y=y, hue=group, ax=ax)
    else:
        sns.lineplot(data=df, x=x, y=y, ax=ax, color="#0369a1")
    ax.set_title(title or f"{y} over {x}")
    save_as = save_name or f"eda_time_series_{y}_by_{x}"
    _save(fig, save_as)


def plot_correlation_heatmap(df: pd.DataFrame, *, method: str = "pearson", save_name: str | None = None):
    """Plot a correlation heatmap and save as eda_corr_heatmap."""

    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr(method=method)
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation heatmap ({method})")
    save_as = save_name or "eda_corr_heatmap"
    _save(fig, save_as)


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None, *, title: str | None = None, save_name: str | None = None):
    """Plot scatterplots for feature relationships."""

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette="Blues")
    ax.set_title(title or f"{y} vs {x}")
    save_as = save_name or f"eda_scatter_{x}_vs_{y}"
    _save(fig, save_as)


def plot_category_counts(df: pd.DataFrame, column: str, *, normalize: bool = False, save_name: str | None = None):
    """Plot category counts or proportions."""

    counts = df[column].value_counts(normalize=normalize)
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues")
    ax.set_ylabel("proportion" if normalize else "count")
    ax.set_title(f"{column} distribution")
    save_as = save_name or f"eda_category_{column}"
    _save(fig, save_as)
