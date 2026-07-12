"""Promoção de modelos no MLflow Model Registry.

Usa *aliases* (padrão moderno do MLflow) para marcar versões como ``staging``
e ``production``, evitando os ``stages`` legados que estão deprecados.
"""

from mlflow import MlflowClient

from src.recsys.config import settings
from src.recsys.logging_config import get_logger

logger = get_logger(__name__)


def latest_version(client: MlflowClient, model_name: str) -> str:
    """Retorna a versão mais recente registrada de um modelo.

    Args:
        client: Cliente do MLflow.
        model_name: Nome do modelo no Registry.

    Returns:
        Número da versão mais recente (como string).

    Raises:
        ValueError: Se o modelo não tiver versões registradas.
    """
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise ValueError(f"Nenhuma versão encontrada para '{model_name}'.")
    return max(versions, key=lambda v: int(v.version)).version


def promote(model_name: str, alias: str, version: str | None = None) -> str:
    """Atribui um alias (ex.: ``production``) a uma versão do modelo.

    Args:
        model_name: Nome do modelo no Registry.
        alias: Alias a aplicar (``staging`` ou ``production``).
        version: Versão alvo (None = versão mais recente).

    Returns:
        A versão que recebeu o alias.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    target = version or latest_version(client, model_name)
    client.set_registered_model_alias(model_name, alias, target)
    logger.info("modelo promovido", model=model_name, alias=alias, version=target)
    return target
