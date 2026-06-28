"""Estratégias de split treino/validação/teste (padrão Strategy).

O split de um sistema de recomendação pode ser feito de formas diferentes
(aleatório ou temporal). Em vez de espalhar ``if`` pelo código, encapsulamos
cada política numa estratégia intercambiável.
"""

from abc import ABC, abstractmethod

import pandas as pd

from src.recsys.logging_config import get_logger

logger = get_logger(__name__)

Split = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class SplitStrategy(ABC):
    """Interface das estratégias de divisão dos dados."""

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int,
    ) -> Split:
        """Divide o DataFrame em treino, validação e teste."""


class RandomSplitStrategy(SplitStrategy):
    """Embaralha as interações e divide por proporção (i.i.d.)."""

    def split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int,
    ) -> Split:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return _split_by_position(shuffled, test_size, val_size)


class TemporalSplitStrategy(SplitStrategy):
    """Ordena por ``timestamp``: treino no passado, teste no futuro.

    Mais realista para recomendação, pois evita usar o futuro para prever o
    passado (vazamento temporal).
    """

    def split(
        self,
        df: pd.DataFrame,
        test_size: float,
        val_size: float,
        seed: int,
    ) -> Split:
        ordered = df.sort_values("timestamp").reset_index(drop=True)
        return _split_by_position(ordered, test_size, val_size)


_STRATEGIES: dict[str, type[SplitStrategy]] = {
    "random": RandomSplitStrategy,
    "temporal": TemporalSplitStrategy,
}


def get_split_strategy(name: str) -> SplitStrategy:
    """Factory simples que resolve a estratégia pelo nome.

    Args:
        name: ``"random"`` ou ``"temporal"``.

    Returns:
        Instância da estratégia escolhida.

    Raises:
        ValueError: Se o nome não corresponder a uma estratégia conhecida.
    """
    try:
        strategy_cls = _STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"Estratégia '{name}' desconhecida. Use: {list(_STRATEGIES)}") from exc

    logger.info("estratégia de split selecionada", strategy=name)
    return strategy_cls()


def _split_by_position(df: pd.DataFrame, test_size: float, val_size: float) -> Split:
    """Corta o DataFrame já ordenado em três partes consecutivas."""
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info("split concluído", treino=len(train), val=len(val), teste=len(test))
    return train, val, test
