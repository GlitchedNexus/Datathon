"""Baseline model builders."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_logistic_regression(config: dict) -> LogisticRegression:
    """Configure and return a logistic regression classifier."""
    # Use when features are mostly linear/low-dimensional; avoid for highly non-linear patterns or heavy feature interactions without feature engineering.
    params = {"max_iter": 1000, "n_jobs": -1, "penalty": "l2", "solver": "lbfgs"}
    params.update(config or {})
    return LogisticRegression(**params)


def build_random_forest_classifier(config: dict) -> RandomForestClassifier:
    """Configure and return a random forest classifier."""
    # Use for tabular data with non-linear signals and mixed feature types; avoid on very high-dimensional sparse data (e.g., large one-hots).
    params = {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": 42}
    params.update(config or {})
    return RandomForestClassifier(**params)


def build_random_forest_regressor(config: dict) -> RandomForestRegressor:
    """Configure and return a random forest regressor."""
    # Use for non-linear/tabular regression where interpretability is secondary; avoid on very wide sparse matrices or when extrapolation is required.
    params = {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": 42}
    params.update(config or {})
    return RandomForestRegressor(**params)
