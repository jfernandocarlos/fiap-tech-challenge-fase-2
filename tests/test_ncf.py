"""Testes da rede NCF e do loop de treino (smoke)."""

import numpy as np
import pytest
import torch

from src.recsys.models.ncf import NCFModel
from src.recsys.models.training import make_loader, train_ncf


def test_forward_output_shape() -> None:
    model = NCFModel(num_users=10, num_items=20, embedding_dim=8, hidden_layers=[16])
    users = torch.LongTensor([0, 1, 2])
    items = torch.LongTensor([5, 6, 7])
    out = model(users, items)
    assert out.shape == (3, 1)


@pytest.mark.smoke
def test_train_ncf_runs_and_returns_history() -> None:
    rng = np.random.default_rng(0)
    users = rng.integers(0, 10, size=200)
    items = rng.integers(0, 20, size=200)
    labels = rng.integers(0, 2, size=200).astype(float)

    loader = make_loader(users, items, labels, batch_size=32, shuffle=True)
    model = NCFModel(num_users=10, num_items=20, embedding_dim=8, hidden_layers=[16])
    train_params = {
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 32,
        "max_epochs": 3,
        "patience": 2,
    }
    trained, history = train_ncf(model, loader, loader, train_params, seed=42)
    assert len(history["train_loss"]) >= 1
    assert isinstance(trained, NCFModel)
