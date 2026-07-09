"""Testes da factory de modelos."""

from sklearn.base import ClassifierMixin

from src.recsys.models.factory import create_baselines, create_ncf
from src.recsys.models.ncf import NCFModel


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


def test_create_ncf_returns_model() -> None:
    params = {"embedding_dim": 8, "hidden_layers": [16, 8], "dropout": 0.1}
    model = create_ncf(num_users=10, num_items=20, model_params=params)
    assert isinstance(model, NCFModel)
    assert model.user_embedding.num_embeddings == 10
    assert model.item_embedding.num_embeddings == 20
