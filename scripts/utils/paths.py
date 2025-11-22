"""Helpers for accessing configured paths (data, figures, artifacts)."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "paths.yaml"


def load_paths() -> dict[str, Any]:
    """Load path configuration from config/paths.yaml."""

    with CONFIG_PATH.open() as fp:
        return yaml.safe_load(fp)
