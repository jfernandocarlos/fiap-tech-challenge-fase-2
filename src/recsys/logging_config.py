"""Setup de logging estruturado usando structlog."""

import logging
import sys

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configura structlog + logging padrão do Python.

    Args:
        level: Nível mínimo de log (padrão: INFO).
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Retorna um logger estruturado para o módulo informado.

    Args:
        name: Nome do módulo (use ``__name__``).

    Returns:
        Logger estruturado pronto para uso.
    """
    return structlog.get_logger(name)
