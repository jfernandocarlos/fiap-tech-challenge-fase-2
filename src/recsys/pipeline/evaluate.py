"""Stage 4 — Avaliação e comparação com baselines.

Treina os baselines, avalia NCF e baselines no teste com 6 métricas, gera o
relatório comparativo e salva ``metrics.json`` (rastreado pelo DVC).
"""

import json
from datetime import datetime

import pandas as pd
import torch

from src.recsys.config import DATA_PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT, settings
from src.recsys.evaluation.metrics import (
    comparison_table,
    evaluate_ncf,
    evaluate_sklearn,
)
from src.recsys.features.engineering import FEATURE_COLUMNS, TARGET
from src.recsys.logging_config import get_logger
from src.recsys.models.factory import create_baselines, create_ncf
from src.recsys.pipeline.base import PipelineStage
from src.recsys.tracking.mlflow_tracker import (
    append_metrics_to_run,
    log_sklearn_run,
    setup_mlflow,
)

logger = get_logger(__name__)

METRICS_FILE = PROJECT_ROOT / "metrics.json"


class EvaluateStage(PipelineStage):
    """Avalia todos os modelos e consolida os resultados."""

    name = "evaluate"

    def execute(self) -> None:
        train = pd.read_parquet(DATA_PROCESSED_DIR / "feat_train.parquet")
        test = pd.read_parquet(DATA_PROCESSED_DIR / "feat_test.parquet")
        mappings = json.loads((DATA_PROCESSED_DIR / "mappings.json").read_text())

        results = self._evaluate_baselines(train, test)
        results["NCF_PyTorch"] = self._evaluate_ncf(test, mappings)

        self._persist(results)

    def _evaluate_baselines(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        """Treina e avalia os baselines scikit-learn."""
        x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
        x_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

        setup_mlflow()
        results: dict[str, dict[str, float]] = {}
        for name, model in create_baselines(settings.random_seed).items():
            model.fit(x_train, y_train)
            metrics = evaluate_sklearn(model, x_test, y_test, name)
            results[name] = metrics
            log_sklearn_run(f"baseline_{name}", {"model": name}, metrics, model)
        return results

    def _evaluate_ncf(self, test: pd.DataFrame, mappings: dict[str, int]) -> dict[str, float]:
        """Carrega a NCF treinada, avalia no teste e anexa métricas à run MLflow."""
        model = create_ncf(mappings["num_users"], mappings["num_items"], self.params["model"])
        model.load_state_dict(torch.load(MODELS_DIR / "ncf.pt", weights_only=True))
        metrics = evaluate_ncf(
            model,
            test["user_idx"].to_numpy(),
            test["item_idx"].to_numpy(),
            test[TARGET].to_numpy(),
            threshold=self.params["train"]["threshold"],
        )
        # A run NCF_PyTorch é criada no train só com best_val_loss; aqui completamos
        # com as métricas de teste para comparar com os baselines na UI do MLflow.
        append_metrics_to_run("NCF_PyTorch", metrics)
        return metrics

    def _persist(self, results: dict[str, dict[str, float]]) -> None:
        """Salva métricas (DVC) e o relatório comparativo (docs)."""
        table = comparison_table(results)
        logger.info("comparativo:\n" + table)

        flat = {
            f"{model}.{metric}": value
            for model, m in results.items()
            for metric, value in m.items()
        }
        METRICS_FILE.write_text(json.dumps(flat, indent=2), encoding="utf-8")

        best = max(results, key=lambda name: results[name]["f1"])
        report = _build_report(results, table, best)
        report_path = PROJECT_ROOT / "docs" / "comparativo_baselines_ncf.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("relatório salvo", path=str(report_path), melhor_f1=best)


def _build_report(
    results: dict[str, dict[str, float]],
    table: str,
    best: str,
) -> str:
    """Monta o conteúdo Markdown do relatório comparativo."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ncf = results["NCF_PyTorch"]
    simple = {k: v for k, v in results.items() if k not in ("NCF_PyTorch", "DummyClassifier")}
    best_simple = max(simple, key=lambda name: simple[name]["f1"]) if simple else "N/A"
    rank = sorted(results, key=lambda name: results[name]["f1"], reverse=True)
    ncf_rank = rank.index("NCF_PyTorch") + 1
    return _REPORT_TEMPLATE.format(
        now=now,
        seed=settings.random_seed,
        table=table,
        best=best,
        best_f1=results[best]["f1"],
        ncf_rank=ncf_rank,
        total=len(results),
        ncf_f1=ncf["f1"],
        ncf_roc=ncf["roc_auc"],
        ncf_pr=ncf["pr_auc"],
        best_simple=best_simple,
    )


_REPORT_TEMPLATE = """# Relatório Comparativo — Recomendador de Filmes

> **Gerado em:** {now}
> **Dataset:** MovieLens (interações user-item)
> **Alvo:** `liked` (rating >= threshold)
> **Seed:** {seed}

---

## 1. Métricas no conjunto de teste

{table}

---

## 2. Análise

- **Melhor F1 geral:** {best} ({best_f1:.4f})
- **Posição da NCF no ranking de F1:** {ncf_rank}º de {total} modelos
- **NCF (PyTorch):** F1={ncf_f1:.4f} | ROC-AUC={ncf_roc:.4f} | PR-AUC={ncf_pr:.4f}
- **Melhor baseline (sem rede neural):** {best_simple}

A rede neural aprende embeddings latentes de usuários e filmes, capturando
padrões de afinidade que os baselines (apoiados apenas em médias) não modelam
diretamente.
"""


if __name__ == "__main__":
    EvaluateStage().run()
