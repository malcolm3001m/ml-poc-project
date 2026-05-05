"""Regression metrics for peak player market value prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return comparable regression metrics for all trained models."""

    clean_predictions = np.maximum(np.asarray(y_pred, dtype=float), 0)

    return {
        "mae_eur": float(mean_absolute_error(y_true, clean_predictions)),
        "rmse_eur": float(np.sqrt(mean_squared_error(y_true, clean_predictions))),
        "r2": float(r2_score(y_true, clean_predictions)),
    }
