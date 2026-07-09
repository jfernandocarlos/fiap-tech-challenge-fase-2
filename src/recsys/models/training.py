"""Loop de treinamento da rede NCF com early stopping."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.recsys.config import MODELS_DIR
from src.recsys.logging_config import get_logger
from src.recsys.models.ncf import NCFModel

logger = get_logger(__name__)


def make_loader(
    users: np.ndarray,
    items: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Monta um ``DataLoader`` a partir dos arrays de índices e rótulos."""
    dataset = TensorDataset(
        torch.LongTensor(users),
        torch.LongTensor(items),
        torch.FloatTensor(labels).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: NCFModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Executa uma época (treino se ``optimizer`` for passado, senão validação)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    with torch.set_grad_enabled(is_train):
        for users, items, labels in loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            logits = model(users, items)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train_ncf(
    model: NCFModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_params: dict[str, Any],
    seed: int,
) -> tuple[NCFModel, dict[str, list[float]]]:
    """Treina a NCF com early stopping pela menor ``val_loss``.

    Args:
        model: Rede a treinar.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        train_params: Seção ``train`` dos parâmetros.
        seed: Semente para reprodutibilidade.

    Returns:
        Tupla ``(modelo, histórico)`` com os pesos da melhor época restaurados.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = train_params["patience"]

    for epoch in range(train_params["max_epochs"]):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info(
            "época",
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = train_params["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("early stopping", epoch=epoch + 1, best_val=round(best_val, 4))
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_model(model: NCFModel, filename: str = "ncf.pt") -> Path:
    """Salva o ``state_dict`` do modelo em ``models/``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    torch.save(model.state_dict(), path)
    logger.info("modelo salvo", path=str(path))
    return path
