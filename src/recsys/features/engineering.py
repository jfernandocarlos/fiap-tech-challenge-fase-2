"""Engenharia de atributos para o recomendador.

Dois conjuntos de saída:

* Índices contíguos (``user_idx``, ``item_idx``) consumidos pela rede neural.
* Atributos estatísticos (médias por usuário/filme) consumidos pelos baselines.

Todas as estatísticas são calculadas **somente no treino** para evitar
vazamento de dados.
"""

import pandas as pd

from src.recsys.logging_config import get_logger

logger = get_logger(__name__)

TARGET = "liked"


def add_like_target(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Cria o alvo binário ``liked`` (1 se ``rating >= threshold``).

    Args:
        df: DataFrame de avaliações.
        threshold: Nota mínima para considerar interação positiva.

    Returns:
        Cópia do DataFrame com a coluna ``liked``.
    """
    out = df.copy()
    out[TARGET] = (out["rating"] >= threshold).astype(int)
    return out


def build_id_mappings(train: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Mapeia ``userId``/``movieId`` para índices contíguos (0..N-1).

    Args:
        train: Split de treino (define o vocabulário de ids conhecidos).

    Returns:
        Tupla ``(user2idx, item2idx)``.
    """
    user2idx = {uid: i for i, uid in enumerate(sorted(train["userId"].unique()))}
    item2idx = {iid: i for i, iid in enumerate(sorted(train["movieId"].unique()))}
    logger.info("mapeamentos criados", usuarios=len(user2idx), filmes=len(item2idx))
    return user2idx, item2idx


def apply_id_mappings(
    df: pd.DataFrame,
    user2idx: dict[int, int],
    item2idx: dict[int, int],
) -> pd.DataFrame:
    """Aplica os mapeamentos e descarta ids não vistos no treino.

    Args:
        df: Split a transformar.
        user2idx: Mapa de usuários.
        item2idx: Mapa de filmes.

    Returns:
        DataFrame com colunas ``user_idx`` e ``item_idx``.
    """
    out = df[df["userId"].isin(user2idx) & df["movieId"].isin(item2idx)].copy()
    out["user_idx"] = out["userId"].map(user2idx)
    out["item_idx"] = out["movieId"].map(item2idx)
    return out.reset_index(drop=True)


def build_stats(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calcula estatísticas de usuário e filme a partir do treino.

    Args:
        train: Split de treino com a coluna ``liked``.

    Returns:
        Tupla ``(user_stats, item_stats, global_mean)``.
    """
    global_mean = float(train["rating"].mean())

    user_stats = train.groupby("userId").agg(
        user_mean_rating=("rating", "mean"),
        user_like_rate=(TARGET, "mean"),
        user_count=("rating", "count"),
    )
    item_stats = train.groupby("movieId").agg(
        item_mean_rating=("rating", "mean"),
        item_like_rate=(TARGET, "mean"),
        item_count=("rating", "count"),
    )
    return user_stats, item_stats, global_mean


def attach_stats(
    df: pd.DataFrame,
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Anexa as estatísticas de treino a um split qualquer.

    Args:
        df: Split a enriquecer.
        user_stats: Estatísticas por usuário.
        item_stats: Estatísticas por filme.
        global_mean: Média global de rating (fallback para ids raros).

    Returns:
        DataFrame com as colunas de atributos preenchidas.
    """
    out = df.merge(user_stats, on="userId", how="left").merge(item_stats, on="movieId", how="left")
    return out.fillna(
        {
            "user_mean_rating": global_mean,
            "item_mean_rating": global_mean,
            "user_like_rate": 0.5,
            "item_like_rate": 0.5,
            "user_count": 0,
            "item_count": 0,
        }
    )


FEATURE_COLUMNS = [
    "user_mean_rating",
    "user_like_rate",
    "user_count",
    "item_mean_rating",
    "item_like_rate",
    "item_count",
]
