"""Cleaning routines to transform raw data into interim datasets."""

import pandas as pd


def clean_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, type conversions, and basic sanity checks for the main table.

    Steps:
    - Normalize column names
    - Parse date-like columns
    - Impute missing numerics with median and categoricals with mode
    - Drop duplicate rows
    """

    cleaned = df.copy()
    cleaned.columns = [c.strip().lower() for c in cleaned.columns]

    for col in cleaned.columns:
        if "date" in col or "time" in col:
            try:
                cleaned[col] = pd.to_datetime(cleaned[col])
            except (ValueError, TypeError):
                pass

    num_cols = cleaned.select_dtypes(include=["number"]).columns
    cat_cols = [c for c in cleaned.columns if c not in num_cols]

    for col in num_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in cat_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])

    cleaned = cleaned.drop_duplicates()
    return cleaned


def clean_aux_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an auxiliary table (e.g., transactions/customers)."""

    aux = df.copy()
    aux.columns = [c.strip().lower() for c in aux.columns]
    aux = aux.drop_duplicates()
    return aux
