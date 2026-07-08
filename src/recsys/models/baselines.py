"""Modelos baseline com scikit-learn.

Servem de referência: se a rede neural não superar esses modelos simples
(treinados sobre atributos estatísticos), não vale a complexidade extra.
"""

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_baseline_models(seed: int) -> dict[str, ClassifierMixin]:
    """Instancia os baselines (não treinados).

    Args:
        seed: Semente para reprodutibilidade.

    Returns:
        Dicionário ``nome -> estimador`` scikit-learn.
    """
    return {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        ),
    }
