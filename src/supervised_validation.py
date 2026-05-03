from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_walk_forward_splits(
    index: pd.DatetimeIndex,
    train_window: int,
    test_window: int,
    step_size: int,
    min_train_size: int,
    embargo_days: int = 0,
) -> list[dict[str, Any]]:
    """Generate leakage-safe walk-forward splits with optional date embargo."""
    idx = pd.DatetimeIndex(index).sort_values()
    n = len(idx)
    if n == 0:
        return []

    train_window = int(train_window)
    test_window = int(test_window)
    step_size = int(step_size)
    min_train_size = int(min_train_size)
    embargo_days = int(embargo_days)

    train_end_pos = max(min_train_size, train_window) - 1
    split_id = 0
    splits: list[dict[str, Any]] = []

    while train_end_pos < n:
        train_start_pos = max(0, train_end_pos - train_window + 1)
        train_len = train_end_pos - train_start_pos + 1
        if train_len < min_train_size:
            train_end_pos += step_size
            continue

        min_test_date = idx[train_end_pos] + pd.Timedelta(days=embargo_days)
        test_start_pos = int(idx.searchsorted(min_test_date, side="right"))
        test_end_pos = test_start_pos + test_window - 1

        if test_end_pos >= n:
            break

        train_idx = np.arange(train_start_pos, train_end_pos + 1)
        test_idx = np.arange(test_start_pos, test_end_pos + 1)

        splits.append(
            {
                "split_id": split_id,
                "train_start": idx[train_start_pos],
                "train_end": idx[train_end_pos],
                "test_start": idx[test_start_pos],
                "test_end": idx[test_end_pos],
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

        split_id += 1
        train_end_pos += step_size

    return splits


def align_features_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and a single target column by timestamp and remove NaNs."""
    if target_col not in targets.columns:
        raise KeyError(f"Target column not found: {target_col}")

    merged = features.join(targets[[target_col]], how="inner").sort_index()
    merged = merged.loc[merged[target_col].notna()].copy()

    X = merged.drop(columns=[target_col]).copy()
    # Fully missing feature columns are non-informative and break model fitting.
    X = X.dropna(axis=1, how="all")
    y = merged[target_col].copy()
    return X, y
