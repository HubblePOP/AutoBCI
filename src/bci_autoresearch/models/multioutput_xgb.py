from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from xgboost import XGBRegressor


@dataclass
class MultiOutputXGBRegressor:
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    n_jobs: int = 1
    random_state: int = 0

    def __post_init__(self) -> None:
        self.estimators_: list[XGBRegressor] = []

    def _base_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "tree_method": self.tree_method,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiOutputXGBRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError("MultiOutputXGBRegressor expects y with shape (n_samples, n_outputs).")
        self.estimators_ = []
        for dim_idx in range(y.shape[1]):
            estimator = XGBRegressor(**self._base_params())
            estimator.fit(x, np.ascontiguousarray(y[:, dim_idx], dtype=np.float32))
            self.estimators_.append(estimator)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit() before predict().")
        x = np.asarray(x, dtype=np.float32)
        preds = [estimator.predict(x).astype(np.float32) for estimator in self.estimators_]
        return np.stack(preds, axis=1).astype(np.float32)
