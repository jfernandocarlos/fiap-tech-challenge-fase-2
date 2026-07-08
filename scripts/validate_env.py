"""Valida se o ambiente está pronto para rodar o projeto.

Checa versão do Python, dependências essenciais, presença do ``.env`` e do
``params.yaml``. Sai com código != 0 se algo estiver faltando.
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = ["pandas", "numpy", "pydantic_settings", "structlog", "yaml", "sklearn"]
MIN_PYTHON = (3, 10)


def _check_python() -> list[str]:
    """Verifica a versão mínima do Python."""
    if sys.version_info < MIN_PYTHON:
        return [
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ requerido (atual: {sys.version.split()[0]})"
        ]
    return []


def _check_packages() -> list[str]:
    """Verifica se as dependências essenciais estão importáveis."""
    problems = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            problems.append(f"pacote ausente: {package}")
    return problems


def _check_files() -> list[str]:
    """Verifica arquivos de configuração esperados."""
    root = Path(__file__).resolve().parents[1]
    problems = []
    if not (root / "params.yaml").exists():
        problems.append("params.yaml ausente")
    if not (root / ".env").exists() and not (root / ".env.example").exists():
        problems.append(".env / .env.example ausente")
    return problems


def main() -> int:
    """Roda todas as checagens e reporta o resultado."""
    problems = _check_python() + _check_packages() + _check_files()
    if problems:
        print("Ambiente INVÁLIDO:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Ambiente OK: Python, dependências e configs validados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
