from __future__ import annotations

import numpy as np
import pandas as pd

from src.supervised_models import fit_predict_walk_forward
from src.supervised_validation import generate_walk_forward_splits


def _toy_data(n: int = 220) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = pd.DataFrame({"portfolio_vol_21d": x1, "btc_vol_21d": x2}, index=idx)
    y_reg = pd.Series(0.5 * x1 + 0.1 * x2 + rng.normal(0, 0.1, size=n), index=idx, name="target_portfolio_vol_21d")
    y_cls = pd.Series((x1 + rng.normal(0, 0.5, size=n) > 0).astype(float), index=idx, name="target_stress_event_21d")
    return X, y_reg, y_cls


def test_fit_predict_walk_forward_regression_smoke() -> None:
    X, y_reg, _ = _toy_data()
    splits = generate_walk_forward_splits(X.index, 120, 21, 21, 100, 5)
    pred = fit_predict_walk_forward(X, y_reg, "ridge", "regression", splits)
    assert {"y_true", "y_pred", "model_name", "split_id", "target_col"}.issubset(pred.columns)
    assert len(pred) > 0


def test_fit_predict_walk_forward_classification_smoke() -> None:
    X, _, y_cls = _toy_data()
    splits = generate_walk_forward_splits(X.index, 120, 21, 21, 100, 5)
    pred = fit_predict_walk_forward(X, y_cls, "logistic", "classification", splits)
    assert {"y_true", "y_pred", "y_prob", "model_name", "split_id", "target_col"}.issubset(pred.columns)
    assert len(pred) > 0
