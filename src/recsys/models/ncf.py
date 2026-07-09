"""Rede neural de recomendação baseada em embeddings (NCF).

Neural Collaborative Filtering: aprende um embedding para cada usuário e cada
filme, concatena os dois e passa por uma MLP que estima a probabilidade do
usuário gostar do filme. Saída em logits (a sigmoid é aplicada fora do modelo).
"""

import torch
import torch.nn as nn

from src.recsys.logging_config import get_logger

logger = get_logger(__name__)


class NCFModel(nn.Module):
    """MLP sobre embeddings de usuário e filme para predição de ``like``."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 32]

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        prev_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

        logger.info(
            "NCF criada",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            total_params=sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self) -> None:
        """Inicializa embeddings com distribuição normal pequena."""
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Recebe índices de usuário e filme, retorna logits (batch, 1)."""
        user_vec = self.user_embedding(users)
        item_vec = self.item_embedding(items)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(x)
