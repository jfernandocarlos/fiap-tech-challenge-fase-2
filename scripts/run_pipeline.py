"""Executa o pipeline completo sem DVC (usado dentro do container).

Ordem: preprocess -> feature_eng -> train -> evaluate.
No host, prefira ``dvc repro`` para aproveitar o cache de estágios.
"""

from src.recsys.logging_config import setup_logging
from src.recsys.pipeline.evaluate import EvaluateStage
from src.recsys.pipeline.feature_eng import FeatureEngStage
from src.recsys.pipeline.preprocess import PreprocessStage
from src.recsys.pipeline.train import TrainStage

STAGES = [PreprocessStage, FeatureEngStage, TrainStage, EvaluateStage]


def main() -> None:
    """Roda todos os stages na ordem."""
    setup_logging()
    for stage_cls in STAGES:
        stage_cls().run()


if __name__ == "__main__":
    main()
