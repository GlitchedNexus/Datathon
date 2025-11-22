"""Advanced model builders (boosted trees, stacking, etc.)."""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = XGBRegressor = None


def build_gradient_boosting_classifier(config: dict) -> GradientBoostingClassifier:
    """Configure and return a gradient boosting classifier."""
    # Use when you need strong non-linear decision boundaries on tabular data with modest size; avoid extremely large/high-cardinality datasets if speed is critical.
    params = {"random_state": 42}
    params.update(config or {})
    return GradientBoostingClassifier(**params)


def build_gradient_boosting_regressor(config: dict) -> GradientBoostingRegressor:
    """Configure and return a gradient boosting regressor."""
    # Use for non-linear regression on moderate-sized datasets; avoid when data volume is huge and training time is tight.
    params = {"random_state": 42}
    params.update(config or {})
    return GradientBoostingRegressor(**params)


def build_xgb_classifier(config: dict):
    """Configure and return an XGBoost classifier (if installed)."""
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")

    # Use when you need strong performance on structured/tabular data and can afford tuning; avoid if training time/compute is extremely limited or data is tiny (overfit risk).
    params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss", "random_state": 42}
    params.update(config or {})
    return XGBClassifier(**params)


def build_xgb_regressor(config: dict):
    """Configure and return an XGBoost regressor (if installed)."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")

    # Use for powerful non-linear regression on tabular data with enough rows for boosting; avoid tiny datasets or when you need very fast/simple baselines.
    params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
    params.update(config or {})
    return XGBRegressor(**params)
