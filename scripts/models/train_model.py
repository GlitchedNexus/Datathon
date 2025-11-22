"""Training entrypoint for reproducible experiments."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scripts.evaluation.evaluate_model import evaluate_classification, evaluate_regression
from scripts.models import advanced_models, baseline_models

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"


def train_and_save_model(config_path: str) -> dict[str, Any]:
    """Train a model using config, return metrics, and persist artifacts.

    Config schema (YAML/JSON):
    - task: "classification" | "regression"
    - data_path: path to processed CSV
    - target: target column name
    - model_name: logical name for saving (optional)
    - model_type: one of the builder keys below
    - model_params: dict of hyperparameters
    - test_size: float (default 0.2)
    - random_state: int (default 42)
    """

    cfg = _load_config(config_path)

    data_path = Path(cfg["data_path"]).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    target_col = cfg["target"]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if cfg.get("task") == "classification" else None
    )

    model_type = cfg.get("model_type", "logistic_regression")
    model_params = cfg.get("model_params", {})
    model = _build_model(model_type, model_params)

    pipeline = Pipeline([("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    metrics = (
        evaluate_classification(y_val, y_pred, y_proba=pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None)
        if cfg.get("task") == "classification"
        else evaluate_regression(y_val, y_pred)
    )

    model_name = cfg.get("model_name", model_type)
    model_path = _save_model(pipeline, model_name)

    return {"metrics": metrics, "model_path": str(model_path)}


def _build_model(model_type: str, params: dict):
    builders = {
        "logistic_regression": lambda: baseline_models.build_logistic_regression(params),
        "random_forest_classifier": lambda: baseline_models.build_random_forest_classifier(params),
        "random_forest_regressor": lambda: baseline_models.build_random_forest_regressor(params),
        "gbdt_classifier": lambda: advanced_models.build_gradient_boosting_classifier(params),
        "gbdt_regressor": lambda: advanced_models.build_gradient_boosting_regressor(params),
        "xgb_classifier": lambda: advanced_models.build_xgb_classifier(params),
        "xgb_regressor": lambda: advanced_models.build_xgb_regressor(params),
    }

    if model_type not in builders:
        raise ValueError(f"Unsupported model_type '{model_type}'")
    return builders[model_type]()


def _load_config(path: str) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    if path_obj.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with path_obj.open() as fp:
            return yaml.safe_load(fp)

    import json

    with path_obj.open() as fp:
        return json.load(fp)


def _save_model(model, model_name: str) -> Path:
    """Persist model artifact to disk."""

    path = ARTIFACT_DIR / f"{model_name}.joblib"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
