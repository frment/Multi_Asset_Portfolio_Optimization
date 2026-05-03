from __future__ import annotations

import pandas as pd

from src.supervised_validation import generate_walk_forward_splits


def test_walk_forward_splits_order_and_embargo() -> None:
    idx = pd.bdate_range("2020-01-01", periods=400)
    splits = generate_walk_forward_splits(
        idx,
        train_window=120,
        test_window=21,
        step_size=21,
        min_train_size=100,
        embargo_days=5,
    )

    assert len(splits) > 0
    for split in splits:
        assert split["train_end"] < split["test_start"]
        assert len(set(split["train_idx"]).intersection(set(split["test_idx"]))) == 0
        assert split["train_start"] <= split["train_end"]
        assert split["test_start"] <= split["test_end"]
