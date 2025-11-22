"""Scaling and encoding helpers for processed datasets."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def scale_numeric(df: pd.DataFrame, *, strategy: str = "standard") -> tuple[pd.DataFrame, BaseEstimator]:
    """Scale numeric features using standardization or min-max.

    Returns transformed DataFrame and the fitted scaler.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns
    scaler: BaseEstimator
    if strategy == "standard":
        scaler = StandardScaler()
    elif strategy == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")

    scaled_array = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)
    non_numeric = df.drop(columns=numeric_cols)
    output = pd.concat([scaled_df, non_numeric], axis=1)
    return output, scaler


def encode_categoricals(df: pd.DataFrame, *, strategy: str = "one_hot") -> tuple[pd.DataFrame, BaseEstimator]:
    """Encode categorical features (one-hot or target encoding).

    Returns transformed DataFrame and the fitted encoder.
    """
    cat_cols = df.select_dtypes(exclude=["number", "datetime64[ns]"]).columns
    if not len(cat_cols):
        return df.copy(), None

    if strategy != "one_hot":
        raise ValueError(f"Only one_hot supported currently, received: {strategy}")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    remaining = df.drop(columns=cat_cols)
    output = pd.concat([remaining, encoded_df], axis=1)
    return output, encoder
