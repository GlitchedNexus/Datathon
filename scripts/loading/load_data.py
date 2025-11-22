"""Centralized data loading utilities.

Intended usage:
- load_raw("train") reads from data/raw/train.csv (or directory-specific variant).
- load_interim / load_processed mirror the same naming pattern.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


STANDARD_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "labels": "labels.csv",
}


def _build_path(stage: str, dataset_name: str, filename: Optional[str] = None) -> Path:
    """Build a path under data/<stage>/ using a standard filename.

    Args:
        stage: One of "raw", "interim", "processed".
        dataset_name: Logical dataset key (e.g., "train", "test", "customers").
        filename: Optional explicit filename; falls back to STANDARD_FILES mapping.
    """

    basename = filename or STANDARD_FILES.get(dataset_name, f"{dataset_name}.csv")
    return DATA_ROOT / stage / basename


def load_raw(dataset_name: str, filename: Optional[str] = None) -> pd.DataFrame:
    """Load a raw dataset from data/raw/ with a consistent filename convention."""

    path = _build_path("raw", dataset_name, filename)
    return pd.read_csv(path)


def load_interim(dataset_name: str, filename: Optional[str] = None) -> pd.DataFrame:
    """Load an interim dataset from data/interim/."""

    path = _build_path("interim", dataset_name, filename)
    return pd.read_csv(path)


def load_processed(dataset_name: str, filename: Optional[str] = None) -> pd.DataFrame:
    """Load a processed dataset from data/processed/."""

    path = _build_path("processed", dataset_name, filename)
    return pd.read_csv(path)
