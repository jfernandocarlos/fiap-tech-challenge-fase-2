"""Testes das métricas de avaliação."""

import numpy as np

from src.recsys.evaluation.metrics import comparison_table, compute_metrics


def test_compute_metrics_perfect_prediction() -> None:
    y_true = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_true, y_prob=y_true.astype(float))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_without_proba_has_no_auc() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert "roc_auc" not in metrics
    assert 0.0 <= metrics["precision"] <= 1.0


def test_comparison_table_contains_models() -> None:
    results = {
        "ModelA": {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.75,
            "pr_auc": 0.7,
        },
        "ModelB": {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.85,
            "pr_auc": 0.8,
        },
    }
    table = comparison_table(results)
    assert "ModelA" in table
    assert "ModelB" in table
    assert "F1" in table
