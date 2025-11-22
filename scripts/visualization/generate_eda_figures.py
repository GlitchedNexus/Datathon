"""Script to orchestrate EDA figure generation into figs/eda."""

from pathlib import Path

import pandas as pd

from scripts.loading.load_data import load_processed
from scripts.plots import eda_plots


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figs" / "eda"


def generate_all_eda_figures(dataset_name: str = "train") -> None:
    """Load processed data, generate standard EDA plots, and save to OUTPUT_DIR."""

    try:
        df = load_processed(dataset_name)
    except FileNotFoundError as exc:
        raise SystemExit(f"Processed dataset not found: {exc}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns.tolist()

    for col in numeric_cols[:3]:
        eda_plots.plot_histogram(df, col)

    if len(numeric_cols) >= 2:
        eda_plots.plot_correlation_heatmap(df[numeric_cols])

    if categorical_cols:
        eda_plots.plot_category_counts(df, categorical_cols[0])


if __name__ == "__main__":
    generate_all_eda_figures()
