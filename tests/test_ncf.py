"""Testes da rede NCF e do loop de treino (smoke)."""

import torch

from src.recsys.models.ncf import NCFModel


def test_forward_output_shape() -> None:
    model = NCFModel(num_users=10, num_items=20, embedding_dim=8, hidden_layers=[16])
    users = torch.LongTensor([0, 1, 2])
    items = torch.LongTensor([5, 6, 7])
    out = model(users, items)
    assert out.shape == (3, 1)
