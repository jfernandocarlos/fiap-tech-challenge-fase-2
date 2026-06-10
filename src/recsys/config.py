from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


class Settings(BaseSettings):
    random_seed: int = Field(default=42)
    mlflow_tracking_uri: str = Field(default=f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow_experiment_name: str = Field(default="movie-recommender")
    registered_model_name: str = Field(default="movielens-ncf")
    model_config = {"env_file": ".env", "extra": "ignore"}


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
