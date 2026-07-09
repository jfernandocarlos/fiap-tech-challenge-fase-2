"""Factory para criação de modelos (padrão Factory Method).

Centraliza a construção dos modelos a partir dos parâmetros, isolando o resto
do código dos detalhes de instanciação.
"""

from typing import Any

from sklearn.base import ClassifierMixin

from src.recsys.models.baselines import get_baseline_models
from src.recsys.models.ncf import NCFModel


def create_ncf(num_users: int, num_items: int, model_params: dict[str, Any]) -> NCFModel:
    """Cria a rede neural NCF a partir dos parâmetros do ``params.yaml``.

    Args:
        num_users: Quantidade de usuários (tamanho do embedding).
        num_items: Quantidade de filmes (tamanho do embedding).
        model_params: Seção ``model`` dos parâmetros.

    Returns:
        Instância de :class:`NCFModel`.
    """
    return NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=model_params["embedding_dim"],
        hidden_layers=model_params["hidden_layers"],
        dropout=model_params["dropout"],
    )


def create_baselines(seed: int) -> dict[str, ClassifierMixin]:
    """Cria o conjunto de baselines scikit-learn.

    Args:
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário ``nome -> estimador``.
    """
    return get_baseline_models(seed)
