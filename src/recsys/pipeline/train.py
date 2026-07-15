"""Stage 3 — Treinamento da rede neural NCF.

Treina a rede sobre os índices de usuário/filme, salva o modelo e registra a
run no MLflow (incluindo o Model Registry).
"""

import json

import numpy as np
import pandas as pd

from src.recsys.config import DATA_PROCESSED_DIR, settings
from src.recsys.features.engineering import TARGET
from src.recsys.logging_config import get_logger
from src.recsys.models.factory import create_ncf
from src.recsys.models.training import make_loader, save_model, train_ncf
from src.recsys.pipeline.base import PipelineStage
from src.recsys.tracking.mlflow_tracker import log_ncf_run, setup_mlflow

logger = get_logger(__name__)


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai arrays de índices e rótulo de um DataFrame de features."""
    return (
        df["user_idx"].to_numpy(),
        df["item_idx"].to_numpy(),
        df[TARGET].to_numpy(),
    )


class TrainStage(PipelineStage):
    """Constrói, treina e registra a NCF."""

    name = "train"

    def execute(self) -> None:
        model_params = self.params["model"]
        train_params = self.params["train"]

        train = pd.read_parquet(DATA_PROCESSED_DIR / "feat_train.parquet")
        val = pd.read_parquet(DATA_PROCESSED_DIR / "feat_val.parquet")
        mappings = json.loads((DATA_PROCESSED_DIR / "mappings.json").read_text())

        train_loader = make_loader(*_arrays(train), train_params["batch_size"], shuffle=True)
        val_loader = make_loader(*_arrays(val), train_params["batch_size"], shuffle=False)

        model = create_ncf(mappings["num_users"], mappings["num_items"], model_params)
        model, history = train_ncf(
            model, train_loader, val_loader, train_params, settings.random_seed
        )
        save_model(model)

        setup_mlflow()
        log_ncf_run(
            params={
                **model_params,
                **train_params,
                "num_users": mappings["num_users"],
                "num_items": mappings["num_items"],
                "epochs_trained": len(history["train_loss"]),
                "seed": settings.random_seed,
            },
            metrics={"best_val_loss": min(history["val_loss"])},
            model=model,
            register_as=settings.registered_model_name,
        )


if __name__ == "__main__":
    TrainStage().run()
