"""Orchestration script for end-to-end experiments."""

from pathlib import Path
from typing import Any

from scripts.models.train_model import train_and_save_model
from scripts.utils import paths as paths_utils

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"
METRICS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "metrics"


def run_experiment(config_path: str) -> dict[str, Any]:
    """Run the full pipeline: load config -> train -> evaluate -> save metrics."""
    cfg = _load_experiment_config(config_path)

    result = train_and_save_model(config_path)

    exp_name = cfg.get("name") or Path(config_path).stem
    metrics_path = _save_metrics(result["metrics"], exp_name)

    result["metrics_path"] = str(metrics_path)
    result["experiment_name"] = exp_name
    return result


def _save_metrics(metrics: dict[str, Any], name: str) -> Path:
    """Persist metrics as JSON under artifacts/metrics."""

    import json

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    with path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    return path


def _load_experiment_config(config_path: str) -> dict[str, Any]:
    path_obj = Path(config_path)
    if not path_obj.is_absolute():
        path_obj = EXPERIMENT_DIR / path_obj

    if not path_obj.exists():
        raise FileNotFoundError(f"Experiment config not found: {path_obj}")

    if path_obj.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs")
        with path_obj.open() as fp:
            return yaml.safe_load(fp)

    import json

    with path_obj.open() as fp:
        return json.load(fp)
