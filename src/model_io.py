"""Helpers for loading serialized models + the CatBoost wrapper class."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class CatBoostLogWrapper(RegressorMixin, BaseEstimator):
    """Manual log1p wrapper around CatBoostRegressor.

    Why this exists: sklearn's TransformedTargetRegressor can't clone CatBoost
    because cat_features mutates state outside __init__. We replicate the
    log1p/expm1 round-trip manually and expose a sklearn-compatible
    fit/predict surface.

    Lives in src/ (not in scripts/) so joblib can unpickle it from any caller.
    Inherits from sklearn's BaseEstimator + RegressorMixin to get clone(),
    get/set_params(), and __sklearn_tags__() for free.
    """

    def __init__(self, cat_indices=None, iterations=500, depth=6, lr=0.05, l2=3, seed=42):
        from catboost import CatBoostRegressor
        # Store constructor params so get_params() can return them (sklearn clone)
        self.cat_indices = cat_indices
        self.iterations = iterations
        self.depth = depth
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self.model = CatBoostRegressor(
            iterations=iterations, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
            cat_features=cat_indices, random_state=seed, verbose=False,
        )

    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.maximum(np.expm1(self.model.predict(X)), 0)

    # ── sklearn estimator API — for use with cross_val_score / clone ──
    def get_params(self, deep=True):
        return {
            "cat_indices": self.cat_indices,
            "iterations": self.iterations,
            "depth": self.depth,
            "lr": self.lr,
            "l2": self.l2,
            "seed": self.seed,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # Rebuild the inner model with new params
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.lr, depth=self.depth,
            l2_leaf_reg=self.l2, cat_features=self.cat_indices,
            random_state=self.seed, verbose=False,
        )
        return self

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle`.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc

        return joblib.load(model_path)

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    raise ValueError(
        f"Unsupported model format for {model_path}. Use .joblib, .pkl, or .pickle."
    )
