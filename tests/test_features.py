"""Testes da engenharia de atributos."""

import pandas as pd

from src.recsys.features.engineering import (
    FEATURE_COLUMNS,
    TARGET,
    add_like_target,
    apply_id_mappings,
    attach_stats,
    build_id_mappings,
    build_stats,
)


def test_add_like_target(ratings_df: pd.DataFrame) -> None:
    out = add_like_target(ratings_df, threshold=4.0)
    assert TARGET in out.columns
    assert set(out[TARGET].unique()).issubset({0, 1})
    assert (out[out["rating"] >= 4.0][TARGET] == 1).all()


def test_id_mappings_are_contiguous(ratings_df: pd.DataFrame) -> None:
    user2idx, item2idx = build_id_mappings(ratings_df)
    assert sorted(user2idx.values()) == list(range(len(user2idx)))
    assert sorted(item2idx.values()) == list(range(len(item2idx)))


def test_apply_id_mappings_drops_unknown(ratings_df: pd.DataFrame) -> None:
    train = ratings_df.iloc[:100]
    user2idx, item2idx = build_id_mappings(train)
    mapped = apply_id_mappings(ratings_df, user2idx, item2idx)
    assert mapped["user_idx"].isin(range(len(user2idx))).all()
    assert mapped["item_idx"].isin(range(len(item2idx))).all()


def test_attach_stats_no_nans(ratings_df: pd.DataFrame) -> None:
    df = add_like_target(ratings_df, threshold=4.0)
    user_stats, item_stats, global_mean = build_stats(df)
    enriched = attach_stats(df, user_stats, item_stats, global_mean)
    assert enriched[FEATURE_COLUMNS].isna().sum().sum() == 0
