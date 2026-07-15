"""Esqueleto comum dos stages do pipeline (padrão Template Method).

``run`` define o passo-a-passo invariável (setup de logging, log de início/fim
e tratamento de erro). Cada stage concreto só implementa ``execute``.
"""

from abc import ABC, abstractmethod

from src.recsys.config import load_params
from src.recsys.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class PipelineStage(ABC):
    """Classe base dos stages do pipeline DVC."""

    name: str = "stage"

    def run(self) -> None:
        """Template method: orquestra a execução do stage."""
        setup_logging()
        self.params = load_params()
        logger.info("stage iniciado", stage=self.name)
        try:
            self.execute()
        except Exception:
            logger.exception("stage falhou", stage=self.name)
            raise
        logger.info("stage concluído", stage=self.name)

    @abstractmethod
    def execute(self) -> None:
        """Implementação concreta do stage."""
