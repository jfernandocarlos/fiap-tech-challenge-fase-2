"""Carregamento e limpeza inicial dos dados brutos do MovieLens."""

import pandas as pd

from src.recsys.config import DATA_RAW_DIR
from src.recsys.logging_config import get_logger

logger = get_logger(__name__)

RATINGS_FILE = DATA_RAW_DIR / "ratings.csv"
MOVIES_FILE = DATA_RAW_DIR / "movies.csv"

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def load_ratings() -> pd.DataFrame:
    """Lê o arquivo de avaliações (``ratings.csv``).

    Returns:
        DataFrame com as colunas ``userId``, ``movieId``, ``rating`` e
        ``timestamp``.

    Raises:
        FileNotFoundError: Se o arquivo não existir em ``data/raw``.
    """
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"ratings.csv não encontrado em {RATINGS_FILE}. Rode `make download-data` antes."
        )

    df = pd.read_csv(RATINGS_FILE, usecols=RATING_COLUMNS)
    logger.info("ratings carregados", linhas=len(df))
    return df


def load_movies() -> pd.DataFrame:
    """Lê o catálogo de filmes (``movies.csv``).

    Returns:
        DataFrame com as colunas ``movieId``, ``title`` e ``genres``.
    """
    df = pd.read_csv(MOVIES_FILE)
    logger.info("filmes carregados", linhas=len(df))
    return df


def filter_active(
    ratings: pd.DataFrame,
    min_user: int,
    min_item: int,
) -> pd.DataFrame:
    """Remove usuários e filmes com poucas interações (reduz ruído/esparsidade).

    Args:
        ratings: DataFrame de avaliações.
        min_user: Mínimo de avaliações por usuário.
        min_item: Mínimo de avaliações por filme.

    Returns:
        DataFrame filtrado.
    """
    item_counts = ratings["movieId"].value_counts()
    keep_items = item_counts[item_counts >= min_item].index
    ratings = ratings[ratings["movieId"].isin(keep_items)]

    user_counts = ratings["userId"].value_counts()
    keep_users = user_counts[user_counts >= min_user].index
    ratings = ratings[ratings["userId"].isin(keep_users)]

    logger.info(
        "interações filtradas",
        usuarios=ratings["userId"].nunique(),
        filmes=ratings["movieId"].nunique(),
        linhas=len(ratings),
    )
    return ratings.reset_index(drop=True)
