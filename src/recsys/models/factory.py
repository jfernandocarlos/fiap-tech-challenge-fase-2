"""Factory para criação de modelos (padrão Factory Method).

Centraliza a construção dos modelos a partir dos parâmetros, isolando o resto
do código dos detalhes de instanciação.
"""

from sklearn.base import ClassifierMixin

from src.recsys.models.baselines import get_baseline_models


def create_baselines(seed: int) -> dict[str, ClassifierMixin]:
    """Cria o conjunto de baselines scikit-learn.

    Args:
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário ``nome -> estimador``.
    """
    return get_baseline_models(seed)
