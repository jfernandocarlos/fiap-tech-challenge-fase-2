"""Métricas de avaliação e tabela comparativa entre modelos."""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.recsys.logging_config import get_logger
from src.recsys.models.ncf import NCFModel

logger = get_logger(__name__)

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula as métricas de classificação binária.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos previstos (0/1).
        y_prob: Probabilidades previstas (habilita ROC-AUC e PR-AUC).

    Returns:
        Dicionário com as métricas.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def evaluate_ncf(
    model: NCFModel,
    users: np.ndarray,
    items: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Avalia a NCF no conjunto informado.

    Args:
        model: Rede treinada.
        users: Índices de usuário.
        items: Índices de filme.
        y_true: Rótulos verdadeiros.
        threshold: Limiar de decisão sobre a probabilidade.

    Returns:
        Dicionário de métricas.
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.LongTensor(users), torch.LongTensor(items))
        y_prob = torch.sigmoid(logits).numpy().flatten()
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("avaliação NCF", **{k: round(v, 4) for k, v in metrics.items()})
    return metrics


def evaluate_sklearn(
    model: Any,
    x_test: np.ndarray,
    y_true: np.ndarray,
    name: str,
) -> dict[str, float]:
    """Avalia um modelo scikit-learn no conjunto de teste.

    Args:
        model: Estimador já treinado.
        x_test: Atributos de teste.
        y_true: Rótulos verdadeiros.
        name: Nome do modelo (para log).

    Returns:
        Dicionário de métricas.
    """
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("avaliação baseline", modelo=name, **{k: round(v, 4) for k, v in metrics.items()})
    return metrics


def comparison_table(results: dict[str, dict[str, float]]) -> str:
    """Gera uma tabela Markdown comparando as métricas dos modelos.

    Args:
        results: Mapa ``modelo -> métricas``.

    Returns:
        Tabela em formato Markdown.
    """
    header = "| Modelo | " + " | ".join(m.upper() for m in METRIC_ORDER) + " |"
    separator = "|" + "|".join(["---"] * (len(METRIC_ORDER) + 1)) + "|"
    rows = [header, separator]
    for name, metrics in results.items():
        values = [
            f"{metrics[m]:.4f}" if isinstance(metrics.get(m), float) else "N/A"
            for m in METRIC_ORDER
        ]
        rows.append(f"| {name} | " + " | ".join(values) + " |")
    return "\n".join(rows)
