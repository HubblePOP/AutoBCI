from __future__ import annotations

import unittest

import numpy as np

from bci_autoresearch.models.multioutput_xgb import MultiOutputXGBRegressor


class MultiOutputXGBRegressorTests(unittest.TestCase):
    def test_fit_predict_preserves_multioutput_shape(self) -> None:
        x = np.random.randn(128, 24).astype(np.float32)
        y = np.random.randn(128, 3).astype(np.float32)
        model = MultiOutputXGBRegressor(
            n_estimators=4,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=0,
        )

        model.fit(x, y)
        pred = model.predict(x[:16])

        self.assertEqual(pred.shape, (16, 3))
        self.assertEqual(len(model.estimators_), 3)


if __name__ == "__main__":
    unittest.main()
