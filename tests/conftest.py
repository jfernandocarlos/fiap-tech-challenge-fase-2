"""Fixtures compartilhadas pelos testes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ratings_df() -> pd.DataFrame:
    """DataFrame sintético de avaliações para testes rápidos."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "userId": rng.integers(1, 21, size=n),
            "movieId": rng.integers(1, 31, size=n),
            "rating": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n),
            "timestamp": rng.integers(1_000_000, 2_000_000, size=n),
        }
    )
