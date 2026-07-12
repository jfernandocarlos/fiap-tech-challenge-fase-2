"""Integração com MLflow para tracking de experimentos."""

from typing import Any

import mlflow

from src.recsys.config import settings
from src.recsys.logging_config import get_logger

logger = get_logger(__name__)


def setup_mlflow() -> str:
    """Configura o tracking URI e o experimento.

    Returns:
        ``experiment_id`` do experimento ativo.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(
        "MLflow configurado",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )
    return experiment.experiment_id


def log_sklearn_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
) -> str:
    """Registra uma run de baseline scikit-learn.

    Args:
        name: Nome da run.
        params: Parâmetros logados.
        metrics: Métricas logadas.
        model: Estimador treinado.

    Returns:
        ``run_id`` da run criada.
    """
    import mlflow.sklearn

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        mlflow.set_tag("stage", "baseline")
        logger.info("baseline registrado", run=name)
        return run.info.run_id


def log_ncf_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
    register_as: str | None = None,
) -> str:
    """Registra a run da NCF e, opcionalmente, no Model Registry.

    Args:
        params: Hiperparâmetros logados.
        metrics: Métricas logadas.
        model: Modelo PyTorch treinado.
        register_as: Nome do modelo registrado (None = não registra).

    Returns:
        ``run_id`` da run criada.
    """
    import mlflow.pytorch

    with mlflow.start_run(run_name="NCF_PyTorch") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=register_as,
        )
        mlflow.set_tag("stage", "neural_network")
        logger.info("NCF registrada", run_id=run.info.run_id, registered=register_as)
        return run.info.run_id
