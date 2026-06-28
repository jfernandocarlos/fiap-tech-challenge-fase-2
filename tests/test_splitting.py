"""Testes das estratégias de split (padrão Strategy)."""

import pandas as pd
import pytest

from src.recsys.data.splitting import (
    RandomSplitStrategy,
    TemporalSplitStrategy,
    get_split_strategy,
)


def test_get_split_strategy_returns_correct_type() -> None:
    assert isinstance(get_split_strategy("random"), RandomSplitStrategy)
    assert isinstance(get_split_strategy("temporal"), TemporalSplitStrategy)


def test_get_split_strategy_unknown_raises() -> None:
    with pytest.raises(ValueError, match="desconhecida"):
        get_split_strategy("inexistente")


def test_split_sizes_and_disjoint(ratings_df: pd.DataFrame) -> None:
    train, val, test = get_split_strategy("random").split(
        ratings_df, test_size=0.2, val_size=0.1, seed=42
    )
    assert len(train) + len(val) + len(test) == len(ratings_df)
    assert len(test) == int(len(ratings_df) * 0.2)


def test_temporal_split_is_ordered(ratings_df: pd.DataFrame) -> None:
    train, _, test = get_split_strategy("temporal").split(
        ratings_df, test_size=0.2, val_size=0.1, seed=42
    )
    assert train["timestamp"].max() <= test["timestamp"].min()
