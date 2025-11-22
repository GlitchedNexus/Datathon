"""Cross-validation utilities for rapid experiments."""

from typing import Callable, Iterable

import numpy as np
from sklearn.model_selection import cross_validate


def run_cv(build_model: Callable, X, y, *, cv: int = 5, scoring: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Run cross-validation with provided model builder and return scores.

    Args:
        build_model: Callable that returns an unfitted estimator when invoked with no args or a config dict.
        X: Features.
        y: Target array/Series.
        cv: Number of folds.
        scoring: Iterable of scoring strings (sklearn-style). Defaults to accuracy.
    """

    estimator = build_model()
    scoring = scoring or ["accuracy"]
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=False)

    # Collect only test_* keys for simplicity
    summary: dict[str, np.ndarray] = {}
    for key, val in results.items():
        if key.startswith("test_"):
            summary[key.replace("test_", "")] = val
    return summary
