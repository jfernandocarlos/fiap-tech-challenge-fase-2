"""Testes da factory de modelos."""

from sklearn.base import ClassifierMixin

from src.recsys.models.factory import create_baselines


def test_create_baselines_returns_estimators() -> None:
    baselines = create_baselines(seed=42)
    assert set(baselines) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForest",
        "GradientBoosting",
    }
    for model in baselines.values():
        assert isinstance(model, ClassifierMixin)
