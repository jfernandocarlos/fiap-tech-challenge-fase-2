"""Baixa o dataset MovieLens do Kaggle e salva em data/raw."""

import shutil
from pathlib import Path

from src.recsys.config import DATA_RAW_DIR
from src.recsys.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

KAGGLE_DATASET = "ayushimishra2809/movielens-dataset"
WANTED_FILES = ["ratings.csv", "movies.csv"]


def download_dataset() -> None:
    setup_logging()
    if all((DATA_RAW_DIR / name).exists() for name in WANTED_FILES):
        logger.info("dataset já presente, pulando download", destino=str(DATA_RAW_DIR))
        return

    import kagglehub

    logger.info("baixando dataset do Kaggle", dataset=KAGGLE_DATASET)
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in WANTED_FILES:
        matches = list(source.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} não encontrado em {source}")
        shutil.copy2(matches[0], DATA_RAW_DIR / name)
        logger.info("arquivo copiado", arquivo=name)
    logger.info("download concluído", destino=str(DATA_RAW_DIR))


if __name__ == "__main__":
    download_dataset()
