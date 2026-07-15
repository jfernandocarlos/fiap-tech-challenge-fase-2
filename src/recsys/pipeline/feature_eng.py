"""Stage 2 — Engenharia de atributos.

Cria índices contíguos de usuário/filme (para a rede) e estatísticas de treino
(para os baselines), aplicando-os a todos os splits sem vazamento de dados.
"""

import json

import pandas as pd

from src.recsys.config import DATA_PROCESSED_DIR
from src.recsys.features.engineering import (
    apply_id_mappings,
    attach_stats,
    build_id_mappings,
    build_stats,
)
from src.recsys.logging_config import get_logger
from src.recsys.pipeline.base import PipelineStage

logger = get_logger(__name__)

MAPPINGS_FILE = DATA_PROCESSED_DIR / "mappings.json"


class FeatureEngStage(PipelineStage):
    """Mapeamento de ids + estatísticas por usuário/filme."""

    name = "feature_eng"

    def execute(self) -> None:
        train = pd.read_parquet(DATA_PROCESSED_DIR / "train.parquet")
        val = pd.read_parquet(DATA_PROCESSED_DIR / "val.parquet")
        test = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")

        user2idx, item2idx = build_id_mappings(train)
        user_stats, item_stats, global_mean = build_stats(train)

        for name, frame in {"train": train, "val": val, "test": test}.items():
            enriched = apply_id_mappings(frame, user2idx, item2idx)
            enriched = attach_stats(enriched, user_stats, item_stats, global_mean)
            enriched.to_parquet(DATA_PROCESSED_DIR / f"feat_{name}.parquet", index=False)
            logger.info("features salvas", split=name, linhas=len(enriched))

        MAPPINGS_FILE.write_text(
            json.dumps({"num_users": len(user2idx), "num_items": len(item2idx)}),
            encoding="utf-8",
        )
        logger.info("mapeamentos salvos", arquivo=str(MAPPINGS_FILE))


if __name__ == "__main__":
    FeatureEngStage().run()
