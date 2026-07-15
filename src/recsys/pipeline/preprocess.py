"""Stage 1 — Pré-processamento.

Carrega as avaliações brutas, filtra usuários/filmes pouco ativos, cria o alvo
binário ``liked`` e divide em treino/validação/teste.
"""

from src.recsys.config import DATA_PROCESSED_DIR, settings
from src.recsys.data.loader import filter_active, load_ratings
from src.recsys.data.splitting import get_split_strategy
from src.recsys.features.engineering import add_like_target
from src.recsys.logging_config import get_logger
from src.recsys.pipeline.base import PipelineStage

logger = get_logger(__name__)


class PreprocessStage(PipelineStage):
    """Limpeza, alvo binário e split dos dados."""

    name = "preprocess"

    def execute(self) -> None:
        data_params = self.params["data"]

        ratings = load_ratings()
        ratings = filter_active(
            ratings,
            min_user=data_params["min_user_interactions"],
            min_item=data_params["min_item_interactions"],
        )
        ratings = add_like_target(ratings, threshold=data_params["like_threshold"])

        strategy = get_split_strategy(data_params["split_strategy"])
        train, val, test = strategy.split(
            ratings,
            test_size=data_params["test_size"],
            val_size=data_params["val_size"],
            seed=settings.random_seed,
        )

        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for split_name, frame in {"train": train, "val": val, "test": test}.items():
            frame.to_parquet(DATA_PROCESSED_DIR / f"{split_name}.parquet", index=False)
        logger.info("splits salvos", destino=str(DATA_PROCESSED_DIR))


if __name__ == "__main__":
    PreprocessStage().run()
