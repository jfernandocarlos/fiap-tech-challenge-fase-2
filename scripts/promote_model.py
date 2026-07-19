"""Promove a versão mais recente do modelo no MLflow Model Registry.

Uso:
    python -m scripts.promote_model --alias staging
    python -m scripts.promote_model --alias production
"""

import argparse

from src.recsys.config import settings
from src.recsys.logging_config import setup_logging
from src.recsys.tracking.registry import promote


def main() -> None:
    """Lê os argumentos de linha de comando e aplica o alias."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Promove modelo no Registry.")
    parser.add_argument(
        "--alias",
        choices=["staging", "production"],
        required=True,
        help="Alias a aplicar à versão.",
    )
    parser.add_argument("--version", default=None, help="Versão alvo (padrão: mais recente).")
    args = parser.parse_args()

    promote(settings.registered_model_name, args.alias, args.version)


if __name__ == "__main__":
    main()
