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
        # MLflow 3.x defaulta para 'pt2' (traced graph), que exige input_example.
        # Mantemos 'pickle' para registrar o nn.Module com forward(users, items).
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=register_as,
            serialization_format="pickle",
        )
        mlflow.set_tag("stage", "neural_network")
        logger.info("NCF registrada", run_id=run.info.run_id, registered=register_as)
        return run.info.run_id


def append_metrics_to_run(run_name: str, metrics: dict[str, float]) -> str | None:
    """Anexa métricas à run mais recente com o nome dado.

    Usado no stage ``evaluate`` para completar a run ``NCF_PyTorch`` (criada no
    treino só com ``best_val_loss``) com as métricas de teste (f1, roc_auc, etc.),
    alinhando a comparação na UI do MLflow com as runs dos baselines.

    Args:
        run_name: Nome da run a atualizar (ex.: ``NCF_PyTorch``).
        metrics: Métricas de teste a registrar.

    Returns:
        ``run_id`` atualizado, ou ``None`` se a run não for encontrada.
    """
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        logger.warning("experimento MLflow não encontrado", name=settings.mlflow_experiment_name)
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        logger.warning("run MLflow não encontrada para anexar métricas", run_name=run_name)
        return None

    run_id = str(runs.iloc[0]["run_id"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    logger.info("métricas anexadas à run", run_name=run_name, run_id=run_id)
    return run_id
